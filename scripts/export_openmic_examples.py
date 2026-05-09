#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import List

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_base.config import load_json, parse_project_config
from training_base.data import audio as audio_utils
from training_base.data.openmic_dataset_loader import OpenMicDataset

try:
    import torchaudio
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError("torchaudio is required to export OpenMIC waveform examples") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a small set of 10-second OpenMIC examples as WAV files."
    )
    parser.add_argument("--config", default=None, help="Optional project config JSON to read label_names/sample_rate/clip_duration/relevance_threshold from")
    parser.add_argument("--root", default=None, help="OpenMIC dataset root")
    parser.add_argument("--split-file", default=None, help="Path to a custom split file such as val.txt or test.txt")
    parser.add_argument("--split", default="val", help="Split name for metadata only (default: val)")
    parser.add_argument("--output-dir", required=True, help="Directory where WAVs and metadata will be written")
    parser.add_argument("--count", type=int, default=8, help="Number of examples to export")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--sample-rate", type=int, default=None, help="Output sample rate override")
    parser.add_argument("--clip-duration-sec", type=float, default=None, help="Clip duration override")
    parser.add_argument("--relevance-threshold", type=float, default=None, help="OpenMIC relevance threshold override")
    parser.add_argument("--label", action="append", dest="labels", default=None, help="Optional label filter, repeatable")
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> tuple[OpenMicDataset, List[str], int]:
    if args.config:
        cfg = parse_project_config(load_json(args.config))
        sample_rate = args.sample_rate or cfg.data.sample_rate
        clip_duration_sec = args.clip_duration_sec or cfg.data.clip_duration_sec
        relevance_threshold = (
            args.relevance_threshold
            if args.relevance_threshold is not None
            else cfg.data.relevance_threshold
        )
        label_names = list(args.labels or cfg.data.label_names)
        root = args.root or cfg.data.val.params.get("root")
        split_file = args.split_file or cfg.data.val.params.get("split_file")
    else:
        if args.root is None or args.split_file is None:
            raise ValueError("--root and --split-file are required when --config is not provided")
        sample_rate = args.sample_rate or 16000
        clip_duration_sec = args.clip_duration_sec or 10.0
        relevance_threshold = args.relevance_threshold if args.relevance_threshold is not None else 0.5
        label_names = list(args.labels or ["piano", "guitar", "bass", "drums", "synthesizer", "violin"])
        root = args.root
        split_file = args.split_file

    clip_num_samples = int(sample_rate * clip_duration_sec)
    split_cfg = {
        "type": "openmic",
        "domain": "real",
        "input_kind": "waveform",
        "params": {
            "root": root,
            "split": args.split,
            "split_file": split_file,
            "relevance_threshold": relevance_threshold,
        },
    }
    dataset = OpenMicDataset(
        split_cfg=split_cfg,
        task_type="multilabel",
        num_classes=len(label_names),
        sample_rate=sample_rate,
        clip_num_samples=clip_num_samples,
        train_mode=False,
        label_names=label_names,
        relevance_threshold=relevance_threshold,
    )
    return dataset, label_names, sample_rate


def main() -> None:
    args = parse_args()
    dataset, label_names, sample_rate = build_dataset(args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[: min(args.count, len(indices))]

    rows = []
    for export_idx, dataset_idx in enumerate(indices, start=1):
        item = dataset.samples[dataset_idx]
        waveform = audio_utils.load_waveform(item["audio_path"], sample_rate)
        excerpt = audio_utils.trim_or_pad(
            waveform,
            dataset.clip_num_samples,
            random_crop=False,
        )

        target = item["target"]
        positive_labels = [
            label_names[class_idx]
            for class_idx, is_active in enumerate(target.tolist())
            if is_active > 0
        ]

        stem = f"{export_idx:02d}_{item['sample_key']}"
        wav_path = output_dir / f"{stem}.wav"
        torchaudio.save(str(wav_path), excerpt.unsqueeze(0), sample_rate)

        rows.append(
            {
                "export_index": export_idx,
                "sample_key": item["sample_key"],
                "source_audio_path": item["audio_path"],
                "exported_wav_path": str(wav_path),
                "positive_labels": positive_labels,
            }
        )
        print(f"[{export_idx}/{len(indices)}] wrote {wav_path.name} labels={positive_labels}")

    metadata_json = output_dir / "metadata.json"
    metadata_csv = output_dir / "metadata.csv"
    metadata_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with metadata_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["export_index", "sample_key", "source_audio_path", "exported_wav_path", "positive_labels"],
        )
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["positive_labels"] = ",".join(row["positive_labels"])
            writer.writerow(csv_row)

    print(f"Wrote {len(rows)} examples to {output_dir}")
    print(f"Metadata JSON: {metadata_json}")
    print(f"Metadata CSV : {metadata_csv}")


if __name__ == "__main__":
    main()
