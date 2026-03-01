from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    """
    Replace this with your real dataset.
    Returns: (x, y, meta)
    """
    def __init__(self, n: int = 10_000, d_in: int = 128, n_classes: int = 10):
        self.n = n
        self.d_in = d_in
        self.n_classes = n_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        x = torch.randn(self.d_in)
        y = torch.randint(0, self.n_classes, size=())
        meta = {"idx": idx}
        return x, y, meta


def create_dataset_from_config(cfg: Dict[str, Any]) -> Dataset:
    # minimal example: “type”-based factory
    ds_type = cfg.get("type", "dummy")
    if ds_type == "dummy":
        return DummyDataset(
            n=int(cfg.get("n", 10_000)),
            d_in=int(cfg.get("d_in", 128)),
            n_classes=int(cfg.get("n_classes", 10)),
        )
    raise ValueError(f"Unknown dataset type: {ds_type}")


def create_dataloaders_from_config(
    data_cfg: Dict[str, Any],
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Returns (train_dl, val_dl, demo_dl).
    demo_dl is intentionally a separate channel so you can feed curated examples,
    mirroring the style you already have. :contentReference[oaicite:4]{index=4}
    """
    train_ds = create_dataset_from_config(data_cfg["train"])
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    val_dl = None
    if "val" in data_cfg and data_cfg["val"] is not None:
        val_ds = create_dataset_from_config(data_cfg["val"])
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )

    demo_dl = None
    if "demo" in data_cfg and data_cfg["demo"] is not None:
        demo_ds = create_dataset_from_config(data_cfg["demo"])
        demo_bs = int(data_cfg["demo"].get("batch_size", min(8, batch_size)))
        demo_dl = DataLoader(
            demo_ds,
            batch_size=demo_bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )

    return train_dl, val_dl, demo_dl