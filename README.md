# soft_moe

Soft Mixture-of-Experts multi-modal model for multi-omics imputation, in PyTorch.

Each modality (e.g. RNA, methylation, protein) is projected to a single token. The tokens pass through a shared stack of Mixture-of-Experts layers, and a per-modality decoder reconstructs the original features. A modality that is missing or deliberately masked is replaced by a learnable mask embedding, so the model can impute it from the modalities that are present.

```
raw input ─► input_proj ─► token [d_model] ─┐
                                            ├─► [TokenMixingAttention] ─► MoE layer(s) ─► decoder ─► reconstruction
missing modality ─► mask embedding ─────────┘
```

Two routing variants share this scaffolding:

| Variant | Class | Routing | Auxiliary loss |
| --- | --- | --- | --- |
| Soft (default) | `MultiModalSoftMoE` | Every slot gets a soft-weighted mix of all tokens ([Puigcerver et al., 2023](https://arxiv.org/abs/2308.00951)) | None |
| Sparse | `MultiModalSparseMoE` | Each token goes to its top-k experts ([Shazeer et al., 2017](https://arxiv.org/abs/1701.06538)) | Switch-style load balancing, in `model.last_aux_loss` |

The sparse variant routes each token independently, so a masked modality's token cannot see the others. Pass `use_attn_mix=True` to put a self-attention layer in front of the MoE stack and give it that channel.

## Installation

Requires Python 3.9+ and PyTorch 2.0+.

```bash
pip install git+https://github.com/marcin119a/SoftMoEOmics.git
```

For development, clone the repository and run `uv sync`, which also installs ruff.

## Quick start

```python
import torch
from torch.utils.data import DataLoader

from soft_moe import (
    build_smoe_model,
    eval_finetune_epoch_smoe,
    finetune_epoch_smoe,
    load_smoe_model,
    save_smoe_model,
)

# Flat config: also what gets stored in a checkpoint.
config = {
    "moe_type": "soft",  # or "sparse"
    "input_dims": {"rna": 2000, "methylation": 500, "protein": 200},
    "d_model": 128,
    "n_experts": 4,
    "n_moe_layers": 2,
}
model = build_smoe_model(**config)

# A batch is a dict of modality name -> float tensor [batch, features].
# Missing values are NaN.
samples = [
    {m: torch.randn(d) for m, d in config["input_dims"].items()} for _ in range(64)
]
loader = DataLoader(samples, batch_size=16)

# Sentinel written into the input in place of NaN and of artificially masked features.
mask_values = {"rna": 0.0, "methylation": 0.0, "protein": 0.0}

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(5):
    train = finetune_epoch_smoe(model, loader, optimizer, "cpu", mask_values)
    val = eval_finetune_epoch_smoe(model, loader, "cpu", mask_values)
    print(epoch, train["total_loss"], val["impute_loss"])

save_smoe_model(model, config, "model.pt")
model = load_smoe_model("model.pt")  # returned in eval mode
```

### Imputing a missing modality

Leave the modality out of the batch (or name it in `masked_mods`), then decode its output token:

```python
batch = next(iter(loader))
observed = {m: x for m, x in batch.items() if m != "protein"}

with torch.no_grad():
    embeddings = model.encode(observed)
    protein_hat = model.decoders["protein"](embeddings["protein"])
```

`model(batch)` returns `(embeddings, reconstructions)`. Embeddings cover every modality; reconstructions cover only the modalities that are present and not masked.

## Training objective

`finetune_epoch_smoe` minimises a weighted sum of three terms:

| Term | Weight | What it measures |
| --- | --- | --- |
| Reconstruction | `lambda_recon` | MSE of the reconstruction from a feature-masked input, blending the error on masked entries with the overall error (`alpha_mask_recon`) |
| Contrastive | `lambda_contrastive` | Symmetric InfoNCE between the embeddings of every pair of modalities from the same sample |
| Imputation | `lambda_impute` | MSE when each modality in turn is masked whole and decoded from the others |

True NaNs are excluded from every MSE. During training, whole modalities are dropped with probability `modality_dropout_prob` and individual features are masked with probability `feature_mask_p`.

Both epoch functions return a dict with `total_loss`, `recon_loss`, `contrast_loss`, `impute_loss` and per-modality `modality_losses`.

The epoch functions do not add `model.last_aux_loss` to the total. To train the sparse variant with load balancing, write your own loop and add that term.

## Configuration

`build_smoe_model(**config)` accepts:

| Key | Soft default | Sparse default | Meaning |
| --- | --- | --- | --- |
| `moe_type` | `"soft"` | | Selects the variant |
| `input_dims` | required | required | Feature count per modality |
| `d_model` | 384 | 256 | Token width |
| `n_experts` | 4 | 8 | Experts per MoE layer |
| `slots_per_expert` | 1 | ignored | Slots routed to each expert (soft only) |
| `top_k` | ignored | 2 | Experts chosen per token (sparse only) |
| `d_ff` | `2 * d_model` | `2 * d_model` | Hidden width of each expert MLP |
| `n_moe_layers` | 2 | 1 | Number of stacked MoE layers |
| `decoder_hidden_layers` | `None` | `None` | Hidden sizes of the decoder MLP; `None` gives a linear decoder |
| `activation_dropout` | 0.124 | 0.0 | Dropout after each ReLU |
| `use_attn_mix` | `False` | `False` | Self-attention across modality tokens before the MoE stack |
| `attn_heads` | 4 | 4 | Heads for that attention layer |

## Layout

```
models/
  multimodal.py     MultiModalMoE: shared scaffolding (projections, mask embeddings, decoders)
  soft_moe.py       MultiModalSoftMoE
  sparse_moe.py     MultiModalSparseMoE
  factory.py        build_smoe_model
  layers/           SoftMoELayer, SparseMoELayer, TokenMixingAttention
training.py         train / eval epoch functions and the imputation loss
checkpoint.py       save_smoe_model, load_smoe_model
_utils.py           masking, reconstruction and contrastive loss helpers
```

## Development

```bash
uv run ruff check .
uv run ruff format --check .
```

CI runs both on every push to `main` and on pull requests.
