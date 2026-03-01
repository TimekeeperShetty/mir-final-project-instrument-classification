from typing import Any, Dict, Tuple
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F


class LitClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, meta = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()

        self.log_dict(
            {"train/loss": loss, "train/acc": acc},
            on_step=True, prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, meta = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log_dict({"val/loss": loss, "val/acc": acc}, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)