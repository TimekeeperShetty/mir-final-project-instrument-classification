from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import audio

OPENMIC_LABELS = [
    "accordion", "banjo", "bass", "cello", "clarinet",
    "cymbals", "drums", "flute", "guitar", "mallet",
    "mandolin", "organ", "piano", "saxophone", "synth",
    "trombone", "trumpet", "ukulele", "violin", "voice",
]


OPENMIC_NAME_MAP = {
    "bass_guitar":     "bass",
    "violin__fiddle":  "violin",
    "electric_guitar": "guitar",
    "acoustic_guitar": "guitar",
    "french_horn":     "trombone",  # closest brass
    "synthesizer":     "synth",
}


CORRUPTED = {
    "071826", "071827", "087435", "095253", "095259",
    "095263", "102144", "113025", "113604", "138485"
}


class OpenMicDataset(Dataset):
    """
    PyTorch Dataset for OpenMIC-2018 — multilabel instrument classification.

    Returns a multi-hot label vector of shape (num_classes,) where each
    dimension corresponds to a class in OPENMIC_LABELS.

    Expected directory layout:
        root/
            audio/
                000/
                    000046_3840.ogg
            openmic-2018-aggregated-labels.csv
            partitions/
                split01_train.csv
                split01_test.csv

    split_cfg example:
        {
            "type": "openmic",
            "domain": "real",
            "input_kind": "waveform",
            "params": {"root": "/scratch/jnl9728/datasets/mir_final_project/openmic-2018"}
        }
    """

    def __init__(
            self,
            split_cfg: Any,
            task_type: str,
            num_classes: int,
            sample_rate: int,
            clip_num_samples: int,
            train_mode: bool,
            relevance_threshold: float = 0.5,
    ) -> None:
        self.task_type        = task_type
        self.num_classes      = num_classes
        self.sample_rate      = sample_rate
        self.clip_num_samples = clip_num_samples
        self.train_mode       = train_mode
        self.threshold        = relevance_threshold

        # Root comes from split_cfg["params"]["root"]
        params = split_cfg.params if hasattr(split_cfg, "params") else split_cfg.get("params", {})
        root   = Path(params["root"])
        split  = params.get("split", "train")  # "train" | "test"

        # Label index map — must match config label_names order
        self.label_to_idx = {label: i for i, label in enumerate(OPENMIC_LABELS)}

        # Load split keys
        split_file = "split01_train.csv" if split == "train" else "split01_test.csv"
        split_keys = set((root / "partitions" / split_file).read_text().splitlines())

        # Load labels CSV
        df = pd.read_csv(root / "openmic-2018-aggregated-labels.csv")

        # Keep only samples in this split
        df = df[df["sample_key"].astype(str).isin(split_keys)]

        # Skip corrupted files
        df = df[~df["sample_key"].astype(str).isin(CORRUPTED)]

        # Normalize instrument names using our name map
        df["instrument"] = df["instrument"].apply(
            lambda x: OPENMIC_NAME_MAP.get(x, x)
        )

        # Keep only instruments in our label set
        df = df[df["instrument"].isin(self.label_to_idx)]

        # Group by sample_key — build one multi-hot vector per clip
        # A label is positive if relevance >= threshold
        self.samples: List[Dict] = []
        unique_keys = df["sample_key"].astype(str).unique()

        for key in unique_keys:
            audio_path = root / "audio" / key[:3] / f"{key}.ogg"
            if not audio_path.exists():
                continue

            clip_df   = df[df["sample_key"].astype(str) == key]
            multihot  = np.zeros(num_classes, dtype=np.float32)

            for _, row in clip_df.iterrows():
                if row["relevance"] >= self.threshold:
                    idx = self.label_to_idx.get(row["instrument"], -1)
                    if 0 <= idx < num_classes:
                        multihot[idx] = 1.0

            # Skip clips with no positive labels
            if multihot.sum() == 0:
                continue

            self.samples.append({
                "audio_path": str(audio_path),
                "label":      multihot,
                "sample_key": key,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        excerpt = audio.load_waveform(item["audio_path"], self.sample_rate)
        excerpt = audio.trim_or_pad(excerpt, self.clip_num_samples, self.train_mode)
        label   = torch.tensor(item["label"], dtype=torch.float32)  # float for multilabel

        return excerpt, label
