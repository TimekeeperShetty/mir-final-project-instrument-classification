import os
import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from .config import load_json, parse_train_config
from .data import create_dataloaders_from_config
from .model import create_model_from_config
from .lit_module import LitClassifier
from .callbacks import DemoCallback


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_json(args.config)
    train_cfg = parse_train_config(cfg["train"])

    pl.seed_everything(train_cfg.seed, workers=True)

    train_dl, val_dl, demo_dl = create_dataloaders_from_config(
        cfg["data"],
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
    )

    model = create_model_from_config(cfg["model"])
    lit = LitClassifier(model=model, lr=train_cfg.lr)

    ckpt_dir = None
    if train_cfg.save_dir:
        ckpt_dir = os.path.join(train_cfg.save_dir, "checkpoints")

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=-1,
        every_n_train_steps=train_cfg.ckpt_every_n_steps,
    )

    callbacks = [ckpt_cb]
    if train_cfg.demo_every_n_steps > 0:
        callbacks.append(DemoCallback(train_cfg.demo_every_n_steps, demo_dl=demo_dl))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        precision=train_cfg.precision,
        max_epochs=train_cfg.max_epochs,
        log_every_n_steps=train_cfg.log_every_n_steps,
        gradient_clip_val=train_cfg.grad_clip,
        accumulate_grad_batches=train_cfg.accumulate_grad_batches,
        callbacks=callbacks,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
    )

    # if val_every_n_steps == 0, Lightning will still run val at epoch end if val_dl exists;
    # you can choose to pass val_dl=None unless you want it.
    trainer.fit(lit, train_dl, val_dl)


if __name__ == "__main__":
    main()