from typing import Dict, List

from .layers import SoftMoELayer
from .multimodal import MultiModalMoE


class MultiModalSoftMoE(MultiModalMoE):
    """
    End-to-end multi-modal model with Soft MoE shared processing
    (see MultiModalMoE for the pipeline). Each MoE layer is a SoftMoELayer:
    every expert sees a soft-weighted combination of all modality tokens, so
    no auxiliary load-balancing loss is needed (last_aux_loss stays 0).
    """
    moe_type = "soft"

    def __init__(self,
                 input_dims: Dict[str, int],
                 d_model: int = 384,
                 n_experts: int = 4,
                 slots_per_expert: int = 1,
                 d_ff: int = None,
                 n_moe_layers: int = 2,
                 decoder_hidden_layers: List[int] = None,
                 activation_dropout: float = 0.124,
                 use_attn_mix: bool = False,
                 attn_heads: int = 4):
        super().__init__(
            input_dims, d_model,
            build_moe_layers=lambda: [
                SoftMoELayer(d_model, n_experts, slots_per_expert, d_ff, activation_dropout)
                for _ in range(n_moe_layers)
            ],
            decoder_hidden_layers=decoder_hidden_layers,
            activation_dropout=activation_dropout,
            use_attn_mix=use_attn_mix,
            attn_heads=attn_heads,
        )
