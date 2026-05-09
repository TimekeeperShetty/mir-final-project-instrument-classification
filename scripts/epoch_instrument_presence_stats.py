#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_base.config import load_json, parse_project_config  # noqa: E402
from training_base.data.datamodule import InstrumentDataModule  # noqa: E402
from training_base.lightning_imports import pl  # noqa: E402


@dataclass
class SplitPresenceStats:
    split: str
    dataset_type: str
    dataset_examples: int
    epoch_examples: int
    batches: int
    dropped_examples: int
    empty_target_examples: int
    avg_active_instruments_per_example: float
    instruments: Dict[str, Dict[str, float]]


def seed_everything(seed: int) -> None:
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    # Matches PyTorch's worker seed and extends it to random/numpy. This matters
    # for augmentation configs that use numpy in addition to torch.
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    datamodule: InstrumentDataModule,
    dataset: Any,
    split_cfg: Any,
    *,
    split_name: str,
    seed: int,
    num_workers_override: Optional[int],
) -> DataLoader:
    loader_cfg = split_cfg.loader
    num_workers = loader_cfg.num_workers if num_workers_override is None else num_workers_override
    persistent_workers = bool(loader_cfg.persistent_workers and num_workers > 0)
    shuffle = split_name == "train"
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=loader_cfg.pin_memory,
        persistent_workers=persistent_workers,
        drop_last=loader_cfg.drop_last if shuffle else False,
        collate_fn=datamodule._get_collate_fn(split_cfg.type),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def targets_to_binary_matrix(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    targets = targets.detach().cpu()
    if targets.ndim == 1:
        binary = torch.zeros((targets.numel(), num_classes), dtype=torch.bool)
        valid = (targets >= 0) & (targets < num_classes)
        if valid.any():
            row_ids = torch.arange(targets.numel())[valid]
            binary[row_ids, targets[valid].long()] = True
        return binary
    if targets.ndim == 2:
        return targets.float() >= 0.5
    raise ValueError(f"Expected 1D or 2D targets, got shape={tuple(targets.shape)}")


def compute_split_stats(
    *,
    split_name: str,
    dataset_type: str,
    loader: DataLoader,
    dataset_len: int,
    label_names: List[str],
) -> SplitPresenceStats:
    counts = torch.zeros(len(label_names), dtype=torch.long)
    total_examples = 0
    total_active_labels = 0
    empty_target_examples = 0
    batches = 0

    progress = tqdm(loader, desc=f"{split_name} epoch", unit="batch")
    for batch in progress:
        binary_targets = targets_to_binary_matrix(batch["targets"], len(label_names))
        batch_counts = binary_targets.sum(dim=0).long()

        counts += batch_counts
        batch_size = int(binary_targets.shape[0])
        total_examples += batch_size
        total_active_labels += int(binary_targets.sum().item())
        empty_target_examples += int((binary_targets.sum(dim=1) == 0).sum().item())
        batches += 1
        del batch

    instruments: Dict[str, Dict[str, float]] = {}
    for class_idx, class_name in enumerate(label_names):
        present = int(counts[class_idx].item())
        absent = total_examples - present
        fraction = (present / total_examples) if total_examples else 0.0
        instruments[class_name] = {
            "present_examples": present,
            "absent_examples": absent,
            "presence_fraction": fraction,
            "presence_percent": fraction * 100.0,
        }

    avg_active = (total_active_labels / total_examples) if total_examples else 0.0
    return SplitPresenceStats(
        split=split_name,
        dataset_type=dataset_type,
        dataset_examples=dataset_len,
        epoch_examples=total_examples,
        batches=batches,
        dropped_examples=max(0, dataset_len - total_examples),
        empty_target_examples=empty_target_examples,
        avg_active_instruments_per_example=avg_active,
        instruments=instruments,
    )


def print_split_stats(stats: SplitPresenceStats) -> None:
    print()
    print(f"{stats.split.upper()} ({stats.dataset_type})")
    print(
        "  "
        f"dataset_examples={stats.dataset_examples} "
        f"epoch_examples={stats.epoch_examples} "
        f"batches={stats.batches} "
        f"dropped_examples={stats.dropped_examples} "
        f"empty_targets={stats.empty_target_examples} "
        f"avg_active={stats.avg_active_instruments_per_example:.3f}"
    )
    print("  instrument       present      absent     percent")
    print("  -------------  ---------  ----------  ----------")
    for class_name, values in stats.instruments.items():
        print(
            f"  {class_name:<13}  "
            f"{int(values['present_examples']):>9}  "
            f"{int(values['absent_examples']):>10}  "
            f"{values['presence_percent']:>9.2f}%"
        )


def parse_splits(raw_splits: Iterable[str]) -> List[str]:
    splits = []
    for split in raw_splits:
        normalized = split.strip().lower()
        if normalized not in {"train", "val"}:
            raise ValueError("This script currently supports only train and val splits.")
        if normalized not in splits:
            splits.append(normalized)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Iterate one dataloader epoch without training and report how often "
            "each configured instrument label is present in emitted examples."
        )
    )
    parser.add_argument("--config", required=True, help="Path to a project JSON config")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to inspect. Supported values: train val",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override trainer.seed for this stats run",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override loader num_workers. Use 0 for the easiest reproducibility/debugging.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the same statistics as JSON",
    )
    args = parser.parse_args()

    cfg = parse_project_config(load_json(args.config))
    seed = int(cfg.trainer.seed if args.seed is None else args.seed)
    splits = parse_splits(args.splits)

    seed_everything(seed)
    datamodule = InstrumentDataModule(cfg)
    datamodule.setup("fit")
    label_names = list(cfg.data.label_names)
    if not label_names and getattr(datamodule.train_dataset, "label_names", None):
        label_names = list(datamodule.train_dataset.label_names)

    split_specs = {
        "train": (datamodule.train_dataset, cfg.data.train, seed),
        "val": (datamodule.val_dataset, cfg.data.val, seed + 1),
    }

    all_stats: List[SplitPresenceStats] = []
    print(f"Config: {Path(args.config).resolve()}")
    print(f"Seed: {seed}")
    print(f"Labels: {', '.join(label_names)}")

    for split_name in splits:
        dataset, split_cfg, split_seed = split_specs[split_name]
        if dataset is None or split_cfg is None:
            print(f"\nSkipping {split_name}: config does not define data.{split_name}")
            continue

        loader = make_loader(
            datamodule,
            dataset,
            split_cfg,
            split_name=split_name,
            seed=split_seed,
            num_workers_override=args.num_workers,
        )
        stats = compute_split_stats(
            split_name=split_name,
            dataset_type=split_cfg.type,
            loader=loader,
            dataset_len=len(dataset),
            label_names=label_names,
        )
        all_stats.append(stats)
        print_split_stats(stats)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(Path(args.config).resolve()),
            "seed": seed,
            "label_names": label_names,
            "splits": [asdict(stats) for stats in all_stats],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON stats to {output_path}")


if __name__ == "__main__":
    main()
