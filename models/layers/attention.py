import torch
from torch import nn


class TokenMixingAttention(nn.Module):
    """
    Lightweight self-attention across the M modality tokens, meant to sit in
    front of a SparseMoELayer. SparseMoELayer routes and processes each token
    fully independently, so a masked modality's (constant mask_emb) token can
    never see the other, varying, present tokens — this layer gives it a
    cross-token channel to do so, the same job SoftMoE's dispatch/combine gets
    "for free" from pooling over all tokens per slot.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + attn_out)
