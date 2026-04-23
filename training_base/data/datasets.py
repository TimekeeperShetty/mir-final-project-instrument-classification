from __future__ import annotations

from typing import Any, Dict

import torch
from torch.utils.data import Dataset


class SlakhDataset(Dataset):
    def __init__(
        self,
        split_cfg: Any,
        task_type: str,
        num_classes: int,
        sample_rate: int,
        clip_num_samples: int,
        train_mode: bool,
    ) -> None:
        raise NotImplementedError(
            "We need to implement the Dataset object for the slakh dataset here"
        )


class OpenMicDataset(Dataset):
    def __init__(
        self,
        split_cfg: Any,
        task_type: str,
        num_classes: int,
        sample_rate: int,
        clip_num_samples: int,
        train_mode: bool,
    ) -> None:
        raise NotImplementedError(
            "We need to implement the Dataset object for the openmic dataset here"
        )


def build_dataset(
    split_cfg: Any,
    task_type: str,
    num_classes: int,
    sample_rate: int,
    clip_num_samples: int,
    train_mode: bool,
) -> Dataset:
    ds_type = split_cfg.type
    if ds_type == "slakh":
        return SlakhDataset(
            split_cfg=split_cfg,
            task_type=task_type,
            num_classes=num_classes,
            sample_rate=sample_rate,
            clip_num_samples=clip_num_samples,
            train_mode=train_mode,
        )
    if ds_type == "openmic":
        return OpenMicDataset(
            split_cfg=split_cfg,
            task_type=task_type,
            num_classes=num_classes,
            sample_rate=sample_rate,
            clip_num_samples=clip_num_samples,
            train_mode=train_mode,
        )

    raise ValueError(f"Unsupported dataset type: {ds_type}")
