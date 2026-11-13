import torch

from .models import MultiModalMoE, build_smoe_model


def save_smoe_model(model: MultiModalMoE, config: dict, path: str):
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_smoe_model(path: str, map_location=None) -> MultiModalMoE:
    data = torch.load(path, map_location=map_location)
    model = build_smoe_model(**data["config"])
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model
