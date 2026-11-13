from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import torch
from torch import nn

from .._utils import ModalityDecoder
from .layers import TokenMixingAttention


class MultiModalMoE(nn.Module):
    """
    Shared scaffolding of the end-to-end multi-modal MoE models: everything
    around the MoE stack, which the concrete subclasses (MultiModalSoftMoE,
    MultiModalSparseMoE) supply via `build_moe_layers`.
    No Phase 1 pretraining required — trains from raw inputs directly.

    Pipeline per modality:
      raw input  →  input_proj  →  token [d_model]
      (stack all modality tokens)
      →  [TokenMixingAttention]  →  MoE layer(s)  →  per-modality output token [d_model]
      →  decoder  →  reconstructed input

    Absent or deliberately masked modalities receive a learnable mask embedding
    instead of the input projection output, enabling modality dropout and
    imputation-aware training.

    `build_moe_layers` is called in the middle of construction (after the input
    projections / attention, before the decoders / mask embeddings) so that
    parameters are created — and hence randomly initialised — in the same order
    as before the split, keeping seeded runs reproducible.
    """

    moe_type: str  # "soft" | "sparse"; set by subclasses

    def __init__(
        self,
        input_dims: dict[str, int],
        d_model: int,
        build_moe_layers: Callable[[], Iterable[nn.Module]],
        decoder_hidden_layers: list[int] | None = None,
        activation_dropout: float = 0.0,
        use_attn_mix: bool = False,
        attn_heads: int = 4,
    ):
        super().__init__()
        self.modalities = sorted(input_dims.keys())  # fixed canonical order
        self.d_model = d_model
        self._mod_to_idx = {m: i for i, m in enumerate(self.modalities)}

        self.input_projs = nn.ModuleDict(
            {m: nn.Linear(input_dims[m], d_model) for m in self.modalities}
        )

        self.attn_mix = (
            TokenMixingAttention(d_model, attn_heads) if use_attn_mix else None
        )

        self.moe_layers = nn.ModuleList(build_moe_layers())

        self.last_aux_loss = torch.zeros(
            ()
        )  # populated by encode(); load-balancing loss for sparse routing

        if decoder_hidden_layers:
            self.decoders = nn.ModuleDict(
                {
                    m: ModalityDecoder(
                        [d_model] + decoder_hidden_layers + [input_dims[m]],
                        activation_dropout,
                    )
                    for m in self.modalities
                }
            )
        else:
            self.decoders = nn.ModuleDict(
                {m: nn.Linear(d_model, input_dims[m]) for m in self.modalities}
            )

        # Learnable mask embeddings (d_model) used when a modality is absent/masked
        self.mask_embs = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(d_model)) for m in self.modalities}
        )

    def encode(
        self, batch: dict[str, torch.Tensor], masked_mods: set | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Project inputs and run through MoE layers.
        Always processes all M modalities; absent/masked ones use mask_embs.
        Returns embeddings for all modalities.
        """
        B = next(iter(batch.values())).shape[0]
        masked = masked_mods or set()

        token_list = []
        for m in self.modalities:
            if m in masked or m not in batch:
                token_list.append(self.mask_embs[m].unsqueeze(0).expand(B, -1))
            else:
                token_list.append(self.input_projs[m](batch[m]))

        x = torch.stack(token_list, dim=1)  # B × M × d_model
        if self.attn_mix is not None:
            x = self.attn_mix(x)
        aux_total = x.new_zeros(())
        for layer in self.moe_layers:
            x, aux = layer(x)
            aux_total = aux_total + aux
        self.last_aux_loss = aux_total

        return {m: x[:, i, :] for i, m in enumerate(self.modalities)}

    def forward(
        self, batch: dict[str, torch.Tensor], masked_mods: set | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Returns (embeddings, reconstructions).
        Reconstructions are produced only for modalities present in batch and not in masked_mods.
        """
        masked = masked_mods or set()
        embeddings = self.encode(batch, masked)
        recon_mods = [m for m in batch if m not in masked]
        reconstructions = {m: self.decoders[m](embeddings[m]) for m in recon_mods}
        return embeddings, reconstructions
