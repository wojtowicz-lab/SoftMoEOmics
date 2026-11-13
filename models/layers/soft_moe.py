from __future__ import annotations

import torch
from torch import nn

from ..._utils import build_mlp


class SoftMoELayer(nn.Module):
    """
    Soft Mixture of Experts layer.
    Reference: 'From Sparse to Soft Mixtures of Experts' (Puigcerver et al., 2023).

    Input:  X  [B × n × d_model]
    Output: Y  [B × n × d_model]

    n_slots = n_experts * slots_per_expert
    Learnable slot embeddings Φ ∈ R^{n_slots × d_model} serve as the router.
    Dispatch weights  D = softmax(X @ Φ.T, dim=1)  — how each token feeds each slot.
    Combine weights   C = softmax(X @ Φ.T, dim=2)  — how each slot contributes to each token.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        slots_per_expert: int = 1,
        d_ff: int | None = None,
        activation_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.slots_per_expert = slots_per_expert
        self.n_slots = n_experts * slots_per_expert
        d_ff = d_ff or d_model * 2

        self.slot_embeddings = nn.Parameter(torch.randn(self.n_slots, d_model) * 0.02)

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
        # x: B × n × d
        logits = torch.einsum("bnd,sd->bns", x, self.slot_embeddings)  # B × n × n_slots

        D = torch.softmax(logits, dim=1)  # dispatch: over tokens, per slot
        C = torch.softmax(logits, dim=2)  # combine: over slots, per token

        # Input to each slot: weighted sum of tokens
        x_slots = torch.einsum("bns,bnd->bsd", D, x)  # B × n_slots × d

        # Apply each expert to its assigned slots
        y_slots = torch.empty_like(x_slots)
        for k, expert in enumerate(self.experts):
            s0 = k * self.slots_per_expert
            s1 = s0 + self.slots_per_expert
            y_slots[:, s0:s1, :] = expert(x_slots[:, s0:s1, :])

        # Combine slot outputs back to token positions
        y = torch.einsum("bns,bsd->bnd", C, y_slots)  # B × n × d
        # No load-balancing loss needed: every expert sees a soft-weighted
        # combination of all tokens on every step, so there is no discrete
        # routing decision to balance.
        return y, x.new_zeros(())
