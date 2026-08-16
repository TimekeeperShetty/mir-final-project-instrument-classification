#!/usr/bin/env python3
"""Report per-class coverage of a built Slakh MIDI index.

This answers "which of my target classes does Slakh actually contain, and how
much of each?" without loading a single sample of audio. It reads the cached
index produced by scripts/build_slakh_midi_index.py and applies the *current*
SLAKH_MIDI_TO_OPENMIC_CLASS mapping, so it also doubles as a quick way to see
the effect of a mapping change before committing to a re-train.

Example:
    python scripts/slakh_index_class_coverage.py \
      --index-path /path/to/slakh_midi_index.pt \
      --label-names configs/slakh_multilabel_clap_20_classes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_base.data.utils import (  # noqa: E402
    OPENMIC_CLASS_TO_INDEX,
    program_to_openmic_class_name,
    union_intervals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-path", required=True, help="Path to slakh_midi_index.pt")
    parser.add_argument(
        "--label-names",
        default=None,
        help="Optional project JSON config; restricts the report to its data.label_names",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Only report these splits (default: every split in the index)",
    )
    parser.add_argument("--json-out", default=None, help="Optional path to dump the report as JSON")
    return parser.parse_args()


def load_target_classes(config_path: Optional[str]) -> List[str]:
    if config_path is None:
        return list(OPENMIC_CLASS_TO_INDEX)
    with open(config_path) as handle:
        config = json.load(handle)
    label_names = config.get("data", {}).get("label_names") or []
    if not label_names:
        raise ValueError(f"No data.label_names found in {config_path}")
    return list(label_names)


def summarize(index_payload: Dict[str, Any], splits: Optional[List[str]]) -> Dict[str, Dict[str, Any]]:
    # Per split, we track for every class: how many tracks contain it at all, how
    # many individual stems map to it, and how many seconds it is active for.
    # Track count is usually the most honest signal -- one track with a very long
    # stem can otherwise make a rare class look well represented.
    per_split: Dict[str, Dict[str, Any]] = {}

    for track in index_payload.get("tracks", {}).values():
        split = str(track.get("split", "unknown"))
        if splits and split not in splits:
            continue

        bucket = per_split.setdefault(
            split,
            {
                "tracks": 0,
                "stems": 0,
                "unmapped_stems": 0,
                "track_counts": defaultdict(int),
                "stem_counts": defaultdict(int),
                "active_seconds": defaultdict(float),
            },
        )
        bucket["tracks"] += 1

        classes_in_track = set()
        for stem in track.get("stems", []):
            bucket["stems"] += 1
            class_name = program_to_openmic_class_name(
                int(stem.get("program_num", -1)),
                is_drum=bool(stem.get("is_drum", False)),
            )
            if class_name is None:
                bucket["unmapped_stems"] += 1
                continue
            classes_in_track.add(class_name)
            bucket["stem_counts"][class_name] += 1
            intervals = [tuple(interval) for interval in stem.get("intervals", [])]
            for start_sec, end_sec in union_intervals(intervals):
                bucket["active_seconds"][class_name] += float(end_sec) - float(start_sec)

        for class_name in classes_in_track:
            bucket["track_counts"][class_name] += 1

    return per_split


def print_report(per_split: Dict[str, Dict[str, Any]], target_classes: List[str]) -> None:
    for split in sorted(per_split):
        bucket = per_split[split]
        total_tracks = bucket["tracks"]
        print(f"\n=== split: {split} ({total_tracks} tracks, {bucket['stems']} stems, "
              f"{bucket['unmapped_stems']} unmapped stems) ===")
        print(f"{'class':<20}{'tracks':>8}{'% tracks':>10}{'stems':>8}{'active hrs':>12}")
        print("-" * 58)

        for class_name in target_classes:
            track_count = bucket["track_counts"].get(class_name, 0)
            stem_count = bucket["stem_counts"].get(class_name, 0)
            hours = bucket["active_seconds"].get(class_name, 0.0) / 3600.0
            pct = (100.0 * track_count / total_tracks) if total_tracks else 0.0
            flag = "   <-- NO DATA" if track_count == 0 else ""
            print(f"{class_name:<20}{track_count:>8}{pct:>9.1f}%{stem_count:>8}{hours:>12.2f}{flag}")

        dead = [name for name in target_classes if bucket["track_counts"].get(name, 0) == 0]
        if dead:
            print(f"\n  {len(dead)} class(es) with zero positive examples: {', '.join(dead)}")


def main() -> None:
    args = parse_args()
    target_classes = load_target_classes(args.label_names)

    payload = torch.load(args.index_path, map_location="cpu", weights_only=False)
    per_split = summarize(payload, args.splits)
    if not per_split:
        raise SystemExit("No tracks matched the requested splits.")

    print_report(per_split, target_classes)

    if args.json_out:
        serializable = {
            split: {
                "tracks": bucket["tracks"],
                "stems": bucket["stems"],
                "unmapped_stems": bucket["unmapped_stems"],
                "track_counts": dict(bucket["track_counts"]),
                "stem_counts": dict(bucket["stem_counts"]),
                "active_seconds": dict(bucket["active_seconds"]),
            }
            for split, bucket in per_split.items()
        }
        Path(args.json_out).write_text(json.dumps(serializable, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
