from typing import Any, Dict
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def create_model_from_config(cfg: Dict[str, Any]) -> nn.Module:
    mtype = cfg.get("type", "mlp")
    if mtype == "mlp":
        return MLP(
            d_in=int(cfg.get("d_in", 128)),
            d_hidden=int(cfg.get("d_hidden", 512)),
            n_classes=int(cfg.get("n_classes", 10)),
        )
    raise ValueError(f"Unknown model type: {mtype}")