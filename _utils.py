import random
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn


def build_mlp(dims: List[int], add_final_activation: bool = False, activation_dropout: float = 0.0) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        is_last = (i == len(dims) - 2)
        if not is_last or add_final_activation:
            layers.append(nn.ReLU())
            if activation_dropout > 0.0:
                layers.append(nn.Dropout(p=activation_dropout))
    return nn.Sequential(*layers)


class ModalityDecoder(nn.Module):
    """Decoder: hidden_dim -> ... -> input_dim (untied)."""
    def __init__(self, dims: List[int], activation_dropout: float = 0.0):
        super().__init__()
        assert len(dims) >= 2, "dims must include [hidden_dim, ..., input_dim]"
        self.net = build_mlp(dims, add_final_activation=False, activation_dropout=activation_dropout)

    def forward(self, h):
        return self.net(h)


def apply_modality_dropout(batch: Dict[str, torch.Tensor], dropout_prob: float = 0.2):
    kept = {}
    for mod, x in batch.items():
        if random.random() > dropout_prob:
            kept[mod] = x
    return kept if kept else batch


def prepare_clean_batch(batch: Dict[str, torch.Tensor],
                         mask_values: Dict[str, float]):
    """
    Replace true NaNs with modality-specific sentinel values.
    Returns:
      - batch_clean: dict[mod] -> tensor with NaNs replaced
      - orig_missing_masks: dict[mod] -> bool tensor for true NaNs
    """
    batch_clean = {}
    orig_missing_masks = {}
    for mod, x in batch.items():
        orig_missing = torch.isnan(x)
        x_clean = x.clone()
        x_clean[orig_missing] = mask_values.get(mod, 0.0)
        batch_clean[mod] = x_clean
        orig_missing_masks[mod] = orig_missing
    return batch_clean, orig_missing_masks


def apply_feature_mask_noise_with_sentinels(
    batch_clean: Dict[str, torch.Tensor],
    mask_values: Dict[str, float],
    p: float = 0.1
):
    """
    For each modality, with prob p per feature, replace with that modality's sentinel.

    Returns:
      noisy_batch: dict[mod] -> tensor (corrupted input)
      artificial_masks: dict[mod] -> bool tensor (True where we masked)
    """
    noisy_batch = {}
    artificial_masks = {}
    if p <= 0.0:
        for mod, x in batch_clean.items():
            noisy_batch[mod] = x
            artificial_masks[mod] = torch.zeros_like(x, dtype=torch.bool)
        return noisy_batch, artificial_masks

    for mod, x in batch_clean.items():
        m = torch.rand_like(x) < p  # True where we mask
        x_noisy = x.clone()
        x_noisy[m] = mask_values.get(mod, 0.0)
        noisy_batch[mod] = x_noisy
        artificial_masks[mod] = m
    return noisy_batch, artificial_masks


def reconstruction_loss_with_masks(
    target_batch: Dict[str, torch.Tensor],
    reconstructions: Dict[str, torch.Tensor],
    orig_missing_masks: Dict[str, torch.Tensor],
    artificial_masks: Dict[str, torch.Tensor] = None,
    alpha_mask: float = 0.5,
):
    """
    For each modality:
      - overall_mse: MSE over non-NaN entries
      - masked_mse: MSE over entries artificially masked (and non-NaN)
      - loss_mod = alpha_mask * masked_mse + (1 - alpha_mask) * overall_mse

    If artificial_masks is None, or p=0, this reduces to overall_mse.
    """
    per_mod_loss = {}
    total = 0.0

    for mod, target in target_batch.items():
        recon = reconstructions[mod]
        diff_sq = (recon - target) ** 2

        valid = ~orig_missing_masks[mod]
        if valid.any():
            overall_mse = diff_sq[valid].mean()
        else:
            overall_mse = diff_sq.mean()

        if artificial_masks is not None:
            mask = artificial_masks[mod] & valid
            if mask.any():
                masked_mse = diff_sq[mask].mean()
            else:
                masked_mse = overall_mse
            loss_mod = alpha_mask * masked_mse + (1.0 - alpha_mask) * overall_mse
        else:
            loss_mod = overall_mse

        per_mod_loss[mod] = loss_mod
        total = total + loss_mod

    return total, per_mod_loss


def contrastive_loss(embeddings: Dict[str, torch.Tensor], temperature: float = 0.1):
    mods = list(embeddings.keys())
    if len(mods) < 2:
        any_tensor = next(iter(embeddings.values()))
        return any_tensor.new_tensor(0.0)

    loss = 0.0
    count = 0
    for i in range(len(mods)):
        for j in range(i+1, len(mods)):
            z1 = embeddings[mods[i]]
            z2 = embeddings[mods[j]]
            sim_ab = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1) / temperature
            labels = torch.arange(z1.size(0), device=z1.device)
            loss += F.cross_entropy(sim_ab, labels)

            sim_ba = F.cosine_similarity(z2.unsqueeze(1), z1.unsqueeze(0), dim=-1) / temperature
            loss += F.cross_entropy(sim_ba, labels)
            count += 2
    return loss / max(count, 1)
