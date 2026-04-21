from __future__ import annotations

try:  # pragma: no cover - depends on installed package
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:  # pragma: no cover - depends on installed package
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

__all__ = ["pl", "ModelCheckpoint"]
