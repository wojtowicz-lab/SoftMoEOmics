from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from ._utils import (
    apply_feature_mask_noise_with_sentinels,
    apply_modality_dropout,
    contrastive_loss,
    prepare_clean_batch,
    reconstruction_loss_with_masks,
)
from .models import MultiModalMoE


def imputation_loss_smoe(
    target_batch: Dict[str, torch.Tensor],
    model: MultiModalMoE,
    orig_missing_masks: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    For each present modality T: mask its token with mask_emb[T], run encode,
    decode T from the MoE output, compute MSE against the clean target (excluding NaNs).
    """
    losses = []
    per_mod: Dict[str, torch.Tensor] = {}

    for target in target_batch:
        other_mods = {m: x for m, x in target_batch.items() if m != target}
        if not other_mods:
            continue

        embs_imp = model.encode(other_mods, masked_mods={target})
        x_imp = model.decoders[target](embs_imp[target])

        target_t = target_batch[target]
        diff_sq = (x_imp - target_t) ** 2
        valid = ~orig_missing_masks[target]

        loss_t = diff_sq[valid].mean() if valid.any() else diff_sq.mean()
        per_mod[target] = loss_t
        losses.append(loss_t)

    if not losses:
        any_t = next(iter(target_batch.values()))
        return any_t.new_tensor(0.0), per_mod

    return torch.stack(losses).mean(), per_mod



def finetune_epoch_smoe(
    model: MultiModalMoE,
    dataloader: DataLoader,
    optimizer,
    device,
    mask_values: Dict[str, float],
    lambda_contrastive: float = 1.0,
    lambda_recon: float = 1.0,
    lambda_impute: float = 1.0,
    modality_dropout_prob: float = 0.2,
    feature_mask_p: float = 0.1,
    alpha_mask_recon: float = 0.5,
    two_path_clean_for_contrast: bool = False,
):
    model.train()
    sums = {"total": 0.0, "recon": 0.0, "contrast": 0.0, "impute": 0.0}
    per_mod_recon_sums: Dict[str, float] = {}
    per_mod_impute_sums: Dict[str, float] = {}
    n = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch = apply_modality_dropout(batch, modality_dropout_prob)

        batch_clean, orig_missing_masks = prepare_clean_batch(batch, mask_values)

        if two_path_clean_for_contrast:
            _, _ = model(batch_clean)
            shared_clean, _ = model(batch_clean)

            noisy_batch, artificial_masks = apply_feature_mask_noise_with_sentinels(
                batch_clean, mask_values, feature_mask_p
            )
            _, recons_noisy = model(noisy_batch)

            rloss, per_mod_recon = reconstruction_loss_with_masks(
                target_batch=batch_clean,
                reconstructions=recons_noisy,
                orig_missing_masks=orig_missing_masks,
                artificial_masks=artificial_masks,
                alpha_mask=alpha_mask_recon,
            )
            present_embeddings = {m: shared_clean[m] for m in batch_clean}
            closs = contrastive_loss(present_embeddings)
            iloss, per_mod_impute = imputation_loss_smoe(batch_clean, model, orig_missing_masks)

        else:
            noisy_batch, artificial_masks = apply_feature_mask_noise_with_sentinels(
                batch_clean, mask_values, feature_mask_p
            )
            shared, recons = model(noisy_batch)

            rloss, per_mod_recon = reconstruction_loss_with_masks(
                target_batch=batch_clean,
                reconstructions=recons,
                orig_missing_masks=orig_missing_masks,
                artificial_masks=artificial_masks,
                alpha_mask=alpha_mask_recon,
            )
            present_embeddings = {m: shared[m] for m in batch_clean}
            closs = contrastive_loss(present_embeddings)
            iloss, per_mod_impute = imputation_loss_smoe(batch_clean, model, orig_missing_masks)

        total = lambda_recon * rloss + lambda_contrastive * closs + lambda_impute * iloss

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        sums["total"]    += total.item()
        sums["recon"]    += rloss.item()
        sums["contrast"] += closs.item()
        sums["impute"]   += iloss.item()

        for m, v in per_mod_recon.items():
            per_mod_recon_sums[m] = per_mod_recon_sums.get(m, 0.0) + v.item()
        for m, v in per_mod_impute.items():
            per_mod_impute_sums[m] = per_mod_impute_sums.get(m, 0.0) + v.item()
        n += 1

    avg = {
        "total_loss":    sums["total"]   / max(n, 1),
        "recon_loss":    sums["recon"]   / max(n, 1),
        "contrast_loss": sums["contrast"]/ max(n, 1),
        "impute_loss":   sums["impute"]  / max(n, 1),
        "modality_losses": {
            "recon":  {m: v / max(n, 1) for m, v in per_mod_recon_sums.items()},
            "impute": {m: v / max(n, 1) for m, v in per_mod_impute_sums.items()},
        }
    }
    return avg


def eval_finetune_epoch_smoe(
    model: MultiModalMoE,
    dataloader: DataLoader,
    device,
    mask_values: Dict[str, float],
    lambda_contrastive: float = 1.0,
    lambda_recon: float = 1.0,
    lambda_impute: float = 1.0,
    feature_mask_p: float = 0.0,
    alpha_mask_recon: float = 0.5,
    two_path_clean_for_contrast: bool = False,
):
    model.eval()
    sums = {"total": 0.0, "recon": 0.0, "contrast": 0.0, "impute": 0.0}
    per_mod_recon_sums: Dict[str, float] = {}
    per_mod_impute_sums: Dict[str, float] = {}
    n = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_clean, orig_missing_masks = prepare_clean_batch(batch, mask_values)

            if two_path_clean_for_contrast:
                shared_clean, _ = model(batch_clean)
                noisy_batch, artificial_masks = apply_feature_mask_noise_with_sentinels(
                    batch_clean, mask_values, feature_mask_p
                )
                _, recons_noisy = model(noisy_batch)

                rloss, per_mod_recon = reconstruction_loss_with_masks(
                    target_batch=batch_clean,
                    reconstructions=recons_noisy,
                    orig_missing_masks=orig_missing_masks,
                    artificial_masks=artificial_masks,
                    alpha_mask=alpha_mask_recon,
                )
                present_embeddings = {m: shared_clean[m] for m in batch_clean}
                closs = contrastive_loss(present_embeddings)
                iloss, per_mod_impute = imputation_loss_smoe(batch_clean, model, orig_missing_masks)

            else:
                noisy_batch, artificial_masks = apply_feature_mask_noise_with_sentinels(
                    batch_clean, mask_values, feature_mask_p
                )
                shared, recons = model(noisy_batch)

                rloss, per_mod_recon = reconstruction_loss_with_masks(
                    target_batch=batch_clean,
                    reconstructions=recons,
                    orig_missing_masks=orig_missing_masks,
                    artificial_masks=artificial_masks,
                    alpha_mask=alpha_mask_recon,
                )
                present_embeddings = {m: shared[m] for m in batch_clean}
                closs = contrastive_loss(present_embeddings)
                iloss, per_mod_impute = imputation_loss_smoe(batch_clean, model, orig_missing_masks)

            total = lambda_recon * rloss + lambda_contrastive * closs + lambda_impute * iloss

            sums["total"]    += total.item()
            sums["recon"]    += rloss.item()
            sums["contrast"] += closs.item()
            sums["impute"]   += iloss.item()

            for m, v in per_mod_recon.items():
                per_mod_recon_sums[m] = per_mod_recon_sums.get(m, 0.0) + v.item()
            for m, v in per_mod_impute.items():
                per_mod_impute_sums[m] = per_mod_impute_sums.get(m, 0.0) + v.item()
            n += 1

    avg = {
        "total_loss":    sums["total"]   / max(n, 1),
        "recon_loss":    sums["recon"]   / max(n, 1),
        "contrast_loss": sums["contrast"]/ max(n, 1),
        "impute_loss":   sums["impute"]  / max(n, 1),
        "modality_losses": {
            "recon":  {m: v / max(n, 1) for m, v in per_mod_recon_sums.items()},
            "impute": {m: v / max(n, 1) for m, v in per_mod_impute_sums.items()},
        }
    }
    return avg
