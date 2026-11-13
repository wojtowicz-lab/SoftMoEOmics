from .multimodal import MultiModalMoE
from .soft_moe import MultiModalSoftMoE
from .sparse_moe import MultiModalSparseMoE


def build_smoe_model(**config) -> MultiModalMoE:
    """
    Build MultiModalSoftMoE or MultiModalSparseMoE from a flat config dict that
    selects the variant with `moe_type` ("soft" by default) — the format the
    training scripts and saved checkpoints use. Keys that only make sense for
    the other variant (`top_k` for soft, `slots_per_expert` for sparse) are
    ignored, as they were when both variants were one class.
    """
    config = dict(config)
    moe_type = config.pop("moe_type", "soft")
    if moe_type == "soft":
        config.pop("top_k", None)
        return MultiModalSoftMoE(**config)
    if moe_type == "sparse":
        config.pop("slots_per_expert", None)
        return MultiModalSparseMoE(**config)
    raise ValueError(f"Unknown moe_type: {moe_type!r} (expected 'soft' or 'sparse')")
