from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..._utils import build_mlp


class SparseMoELayer(nn.Module):
    """
    Sparse (top-k) Mixture-of-Experts layer — the routing scheme SoftMoE was
    proposed as an alternative to. Reference: Shazeer et al. 2017 ("Outrageously
    Large Neural Networks") for top-k gating; Fedus et al. 2021 (Switch
    Transformer) for the load-balancing auxiliary loss used here.

    Unlike SoftMoELayer (which dispatches a *soft*, differentiable mixture of
    every token to every slot), this layer makes a *hard* per-token choice of
    its top-k experts out of n_experts, weighted by the router's softmax
    probability restricted to (and renormalised over) those k experts.

    Input:  X  [B × n × d_model]
    Output: Y  [B × n × d_model], aux_loss (scalar load-balancing loss)
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 2,
        d_ff: int | None = None,
        activation_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        d_ff = d_ff or d_model * 2

        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList(
            [
                build_mlp(
                    [d_model, d_ff, d_model],
                    add_final_activation=False,
                    activation_dropout=activation_dropout,
                )
                for _ in range(n_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, n, d = x.shape
        logits = self.router(x)  # B × n × E
        probs = torch.softmax(logits, dim=-1)  # B × n × E

        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)  # B × n × k
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        flat_x = x.reshape(-1, d)  # (B*n) × d
        flat_idx = topk_idx.reshape(-1, self.top_k)  # (B*n) × k
        flat_w = topk_probs.reshape(-1, self.top_k)  # (B*n) × k
        flat_y = torch.zeros_like(flat_x)

        for e, expert in enumerate(self.experts):
            sel = flat_idx == e  # (B*n) × k, True where slot k routed to expert e
            token_mask = sel.any(dim=-1)  # (B*n,)
            if not token_mask.any():
                continue
            tok_ids = token_mask.nonzero(as_tuple=True)[0]
            out = expert(flat_x[tok_ids])
            w = (flat_w[tok_ids] * sel[tok_ids].float()).sum(dim=-1, keepdim=True)
            flat_y[tok_ids] = flat_y[tok_ids] + w * out

        y = flat_y.reshape(B, n, d)

        # Switch-style load-balancing loss: n_experts * sum_e(f_e * P_e), where
        # f_e = fraction of tokens dispatched to expert e (hard) and P_e = mean
        # router probability mass on expert e (soft) — minimised when routing
        # is uniform across experts, discouraging a few experts absorbing all tokens.
        dispatch_mask = (
            F.one_hot(topk_idx, num_classes=self.n_experts)
            .float()
            .sum(dim=2)
            .clamp(max=1.0)
        )
        tokens_per_expert = dispatch_mask.reshape(-1, self.n_experts).mean(dim=0)
        prob_per_expert = probs.reshape(-1, self.n_experts).mean(dim=0)
        aux_loss = self.n_experts * (tokens_per_expert * prob_per_expert).sum()

        return y, aux_loss
