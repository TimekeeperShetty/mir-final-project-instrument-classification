from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from ..evaluation.metrics import compute_epoch_metrics, init_metric_state, update_metric_state
from ..lightning_imports import pl
from .classifiers import build_classifier
from .encoders import build_encoder


class DomainTransferSystem(pl.LightningModule):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        #building the encoder here
        self.encoder = build_encoder(cfg.encoder)
        #building what will be on top of the encoder
        self.classifier = build_classifier(cfg.classifier, self.encoder.output_dim, cfg.data.num_classes)
        self.task_type = cfg.data.task_type
        #threshold is being used for predicting or not predicting a specific instrument
        self.threshold = cfg.data.threshold
        self.optimizer_cfg = cfg.optimizer
        self.val_state: Dict[str, torch.Tensor] = {}
        self.test_state: Dict[str, torch.Tensor] = {}
        self.save_hyperparameters(ignore=["cfg"])

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        #this is where we pass the inputs inside the ecoder and then into the classification head
        embeddings = self.encoder(inputs, lengths=lengths)
        return self.classifier(embeddings)

    def training_step(self, batch, batch_idx):
        #the shared step is to calculate the loss function
        logits, loss = self._shared_step(batch)
        metrics = {"train/loss": loss}
        self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch["targets"].shape[0])
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_state = init_metric_state(self.task_type, self.device)

    def validation_step(self, batch, batch_idx):
        logits, loss = self._shared_step(batch)
        update_metric_state(self.val_state, self.task_type, logits, batch["targets"], loss, self.threshold)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics(self.val_state, "val")

    def on_test_epoch_start(self) -> None:
        self.test_state = init_metric_state(self.task_type, self.device)

    def test_step(self, batch, batch_idx):
        logits, loss = self._shared_step(batch)
        update_metric_state(self.test_state, self.task_type, logits, batch["targets"], loss, self.threshold)

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics(self.test_state, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optimizer_cfg.lr, weight_decay=self.optimizer_cfg.weight_decay)
        if self.optimizer_cfg.scheduler == "none":
            return optimizer
        if self.optimizer_cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.trainer.max_epochs),
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        raise ValueError(f"Unsupported scheduler: {self.optimizer_cfg.scheduler}")

    def _shared_step(self, batch):
        #Shared forward + loss path. Good place to customize task logic.
        
        #I am implementing for now the multi label classification loss
        logits = self(batch["inputs"], batch["lengths"])

        targets = batch["targets"].float()
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return logits, loss

    def _log_epoch_metrics(self, state: Dict[str, torch.Tensor], prefix: str) -> None:
        if not state:
            return
        packed = torch.cat([tensor.reshape(1) for tensor in state.values()])
        if self.trainer.world_size > 1:
            packed = self.all_gather(packed).sum(dim=0)
        global_state = dict(zip(state.keys(), packed.unbind(dim=0)))
        metrics = compute_epoch_metrics(self.task_type, global_state)
        self.log_dict(
            {f"{prefix}/{name}": value for name, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=(prefix != "test"),
            sync_dist=False,
        )
