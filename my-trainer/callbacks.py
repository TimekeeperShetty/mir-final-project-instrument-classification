from typing import Optional
import pytorch_lightning as pl
import torch


class DemoCallback(pl.Callback):
    """
    Skeleton demo callback:
    - Fires every N train steps
    - Can use a separate demo_dl iterator (curated / deterministic / random)
    Fill in: sampling, logging, saving artifacts, etc.
    """
    def __init__(self, demo_every_n_steps: int, demo_dl=None):
        super().__init__()
        self.demo_every = int(demo_every_n_steps)
        self.demo_dl = demo_dl
        self._it = iter(demo_dl) if demo_dl is not None else None
        self._last_step = -1

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.demo_every <= 0:
            return
        if (trainer.global_step - 1) % self.demo_every != 0:
            return
        if self._last_step == trainer.global_step:
            return
        self._last_step = trainer.global_step

        pl_module.eval()
        try:
            # 1) Use the current train batch as “demo input”
            x, y, meta = batch

            # 2) Optionally also sample from demo_dl
            demo_batch = None
            if self._it is not None:
                try:
                    demo_batch = next(self._it)
                except StopIteration:
                    self._it = iter(self.demo_dl)
                    demo_batch = next(self._it)

            # TODO: put your real demo logic here:
            # - run a forward pass / sampling pass
            # - log metrics/images/audio/etc
            # - write artifacts to disk
            # Example placeholder:
            logits = pl_module(x.to(pl_module.device))
            pred = logits.argmax(dim=-1)
            trainer.logger.log_metrics(
                {"demo/pred_mean": pred.float().mean().item()},
                step=trainer.global_step,
            )

        finally:
            pl_module.train()