from __future__ import annotations

from typing import Any

from torch import nn


class LinearClassifier(nn.Module):
    """Default shallow baseline."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, embeddings):
        return self.head(embeddings)


class MLPClassifier(nn.Module):
    """Simple example head, easy to replace."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embeddings):
        return self.head(embeddings)


def build_classifier(cfg: Any, encoder_dim: int, num_classes: int) -> nn.Module:
    """Classifier factory hook."""

    if cfg.type == "linear":
        return LinearClassifier(input_dim=encoder_dim, num_classes=num_classes)
    if cfg.type == "mlp":
        return MLPClassifier(
            input_dim=encoder_dim,
            hidden_dim=cfg.hidden_dim,
            num_classes=num_classes,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unsupported classifier type: {cfg.type}")
