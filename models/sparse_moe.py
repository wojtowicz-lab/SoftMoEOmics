from typing import Dict, List

from .layers import SparseMoELayer
from .multimodal import MultiModalMoE


class MultiModalSparseMoE(MultiModalMoE):
    """
    End-to-end multi-modal model with sparse (top-k) MoE shared processing
    (see MultiModalMoE for the pipeline) — the baseline SoftMoE is compared
    against. Each MoE layer is a SparseMoELayer, which routes every modality
    token independently; use `use_attn_mix=True` to give a masked modality's
    token a cross-token channel to the present ones (see TokenMixingAttention).
    After encode(), `last_aux_loss` holds the summed load-balancing loss.
    """
    moe_type = "sparse"

    def __init__(self,
                 input_dims: Dict[str, int],
                 d_model: int = 256,
                 n_experts: int = 8,
                 top_k: int = 2,
                 d_ff: int = None,
                 n_moe_layers: int = 1,
                 decoder_hidden_layers: List[int] = None,
                 activation_dropout: float = 0.0,
                 use_attn_mix: bool = False,
                 attn_heads: int = 4):
        super().__init__(
            input_dims, d_model,
            build_moe_layers=lambda: [
                SparseMoELayer(d_model, n_experts, top_k, d_ff, activation_dropout)
                for _ in range(n_moe_layers)
            ],
            decoder_hidden_layers=decoder_hidden_layers,
            activation_dropout=activation_dropout,
            use_attn_mix=use_attn_mix,
            attn_heads=attn_heads,
        )
