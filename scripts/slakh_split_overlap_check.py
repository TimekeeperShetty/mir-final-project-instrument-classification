#!/usr/bin/env python3
"""Detect duplicate-content tracks shared between Slakh splits.

Slakh2100-redux keeps 390 tracks in an `omitted` split precisely because their
musical content duplicates tracks that live elsewhere in the dataset. Our training
configs ask for split ["train", "omitted"], so answering "is that merge safe?"
means answering "does any omitted track duplicate a test or validation track?".

The index-level duplicate guard in build_slakh_midi_index.py only compares track
*ids*, which never collide across split folders. This script compares track
*content* instead, using two fingerprints built from the cached index:

  exact  sorted (program_num, is_drum, note intervals @1ms) over every stem.
         Identical arrangement, identical timing.
  loose  duration @0.1s + the multiset of (program_num, is_drum) + per-stem note
         counts. Catches the same arrangement re-rendered with timing jitter or
         a different patch assignment, which `exact` would miss.

Reads only the .pt index: no audio, no MIDI files, no GPU. Seconds to run.

Example:
    python scripts/slakh_split_overlap_check.py \
      --index-path /path/to/slakh_midi_index.pt \
      --candidate-split omitted \
      --holdout-splits test validation \
      --label-names configs/slakh_multilabel_clap_20_classes.json \
      --json-out slakh_split_overlap.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        "--candidate-split",
        default="omitted",
        help="The split we are considering adding to training (default: omitted)",
    )
    parser.add_argument(
        "--holdout-splits",
        nargs="*",
        default=["test", "validation"],
        help="Splits that must stay clean of training material (default: test validation)",
    )
    parser.add_argument(
        "--label-names",
        default=None,
        help="Optional project JSON config; restricts the per-class gain table to its data.label_names",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=1,
        help="Rounding applied to note intervals for the exact fingerprint (default: 1ms)",
    )
    parser.add_argument("--json-out", default=None, help="Optional path to dump the full report as JSON")
    parser.add_argument(
        "--show",
        type=int,
        default=10,
        help="How many colliding pairs to print per category (default: 10, use 0 for all)",
    )
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


def _digest(payload: Any) -> str:
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]


def fingerprint_track(track: Dict[str, Any], interval_ms: int) -> Tuple[str, str]:
    # Both fingerprints are built only from facts the index already caches, so this
    # never has to touch a MIDI file or a sample of audio.
    quantum = max(interval_ms, 1) / 1000.0

    exact_stems = []
    loose_programs = []
    loose_counts = []
    for stem in track.get("stems", []):
        program = int(stem.get("program_num", -1))
        is_drum = bool(stem.get("is_drum", False))
        intervals = [
            (round(float(start) / quantum), round(float(end) / quantum))
            for start, end in (tuple(interval) for interval in stem.get("intervals", []))
        ]
        exact_stems.append((program, is_drum, tuple(sorted(intervals))))
        loose_programs.append((program, is_drum))
        loose_counts.append(len(intervals))

    exact = _digest(tuple(sorted(exact_stems)))

    # The loose fingerprint deliberately drops exact note timings: a re-render of the
    # same arrangement keeps its instrumentation and note counts but may shift timings.
    duration = round(float(track.get("duration_sec", 0.0)), 1)
    loose = _digest((duration, tuple(sorted(loose_programs)), tuple(sorted(loose_counts))))
    return exact, loose


def group_by_fingerprint(
    tracks: List[Dict[str, Any]], interval_ms: int
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    by_exact: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_loose: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for track in tracks:
        exact, loose = fingerprint_track(track, interval_ms)
        by_exact[exact].append(track)
        by_loose[loose].append(track)
    return by_exact, by_loose


def find_cross_split_collisions(
    groups: Dict[str, List[Dict[str, Any]]],
    left_split: str,
    right_splits: List[str],
) -> List[Dict[str, Any]]:
    # A collision is a fingerprint group holding at least one track from left_split and
    # at least one from any of right_splits. Those are the pairs that would leak.
    collisions = []
    for fingerprint, members in groups.items():
        left = [t for t in members if str(t.get("split")) == left_split]
        right = [t for t in members if str(t.get("split")) in right_splits]
        if left and right:
            collisions.append(
                {
                    "fingerprint": fingerprint,
                    "candidates": sorted(str(t.get("track_id")) for t in left),
                    "holdout": sorted(f"{t.get('track_id')} ({t.get('split')})" for t in right),
                }
            )
    return sorted(collisions, key=lambda c: c["candidates"][0])


def class_hours(tracks: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, int]]:
    hours: Dict[str, float] = defaultdict(float)
    track_counts: Dict[str, int] = defaultdict(int)
    for track in tracks:
        seen = set()
        for stem in track.get("stems", []):
            class_name = program_to_openmic_class_name(
                int(stem.get("program_num", -1)),
                is_drum=bool(stem.get("is_drum", False)),
            )
            if class_name is None:
                continue
            seen.add(class_name)
            intervals = [tuple(interval) for interval in stem.get("intervals", [])]
            for start_sec, end_sec in union_intervals(intervals):
                hours[class_name] += (float(end_sec) - float(start_sec)) / 3600.0
        for class_name in seen:
            track_counts[class_name] += 1
    return hours, track_counts


def print_collisions(title: str, collisions: List[Dict[str, Any]], show: int) -> None:
    print(f"\n{title}: {len(collisions)} colliding fingerprint group(s)")
    if not collisions:
        print("  none")
        return
    limit = len(collisions) if show == 0 else min(show, len(collisions))
    for entry in collisions[:limit]:
        print(f"  {', '.join(entry['candidates'])}  ==  {', '.join(entry['holdout'])}")
    if limit < len(collisions):
        print(f"  ... and {len(collisions) - limit} more (use --show 0 to list all)")


def main() -> None:
    args = parse_args()
    target_classes = load_target_classes(args.label_names)

    payload = torch.load(args.index_path, map_location="cpu", weights_only=False)
    tracks = list(payload.get("tracks", {}).values())
    if not tracks:
        raise SystemExit(f"No tracks found in {args.index_path}")

    split_sizes: Dict[str, int] = defaultdict(int)
    for track in tracks:
        split_sizes[str(track.get("split"))] += 1

    print(f"index: {args.index_path}")
    print(f"tracks: {len(tracks)}  splits: " + ", ".join(f"{s}={n}" for s, n in sorted(split_sizes.items())))

    candidate = args.candidate_split
    if candidate not in split_sizes:
        raise SystemExit(
            f"Candidate split {candidate!r} is not in the index. Available: {sorted(split_sizes)}"
        )
    holdouts = [s for s in args.holdout_splits if s in split_sizes]
    missing = sorted(set(args.holdout_splits) - set(holdouts))
    if missing:
        print(f"warning: holdout split(s) not in index, ignored: {', '.join(missing)}")
    if not holdouts:
        raise SystemExit("None of the requested holdout splits exist in this index.")

    by_exact, by_loose = group_by_fingerprint(tracks, args.interval_ms)

    print(f"\n=== {candidate} vs {'/'.join(holdouts)} — would this merge leak? ===")
    exact_hits = find_cross_split_collisions(by_exact, candidate, holdouts)
    loose_hits = find_cross_split_collisions(by_loose, candidate, holdouts)
    print_collisions("exact content match", exact_hits, args.show)
    print_collisions("loose content match (same instrumentation + note counts)", loose_hits, args.show)

    # Pre-existing leakage is worth knowing regardless of the merge decision: if train
    # already duplicates test, that is a separate problem this script happens to detect.
    print(f"\n=== train vs {'/'.join(holdouts)} — pre-existing leakage, independent of the merge ===")
    print_collisions("exact content match", find_cross_split_collisions(by_exact, "train", holdouts), args.show)
    print_collisions(
        "loose content match",
        find_cross_split_collisions(by_loose, "train", holdouts),
        args.show,
    )

    tainted = {tid for entry in exact_hits for tid in entry["candidates"]}
    tainted_loose = {tid for entry in loose_hits for tid in entry["candidates"]}
    candidate_tracks = [t for t in tracks if str(t.get("split")) == candidate]
    clean_strict = [t for t in candidate_tracks if str(t.get("track_id")) not in (tainted | tainted_loose)]
    clean_exact_only = [t for t in candidate_tracks if str(t.get("track_id")) not in tainted]

    print(f"\n=== verdict for merging {candidate!r} into train ===")
    total = len(candidate_tracks)
    print(f"  {total} tracks in {candidate}")
    print(f"  {len(tainted)} collide with a holdout track by exact content")
    print(f"  {len(tainted_loose - tainted)} more collide only by loose content (likely re-renders)")
    print(f"  {len(clean_strict)} safe under both fingerprints "
          f"({100.0 * len(clean_strict) / total:.1f}% of {candidate})")

    hours, track_counts = class_hours(clean_strict)
    if hours:
        print(f"\n=== per-class gain from the {len(clean_strict)} safe {candidate} tracks ===")
        print(f"{'class':<20}{'tracks':>8}{'active hrs':>12}")
        print("-" * 40)
        for class_name in target_classes:
            count = track_counts.get(class_name, 0)
            if count == 0:
                continue
            print(f"{class_name:<20}{count:>8}{hours.get(class_name, 0.0):>12.2f}")

    if args.json_out:
        report = {
            "index_path": str(args.index_path),
            "split_sizes": dict(split_sizes),
            "candidate_split": candidate,
            "holdout_splits": holdouts,
            "interval_ms": args.interval_ms,
            "collisions": {
                f"{candidate}_vs_holdout_exact": exact_hits,
                f"{candidate}_vs_holdout_loose": loose_hits,
                "train_vs_holdout_exact": find_cross_split_collisions(by_exact, "train", holdouts),
                "train_vs_holdout_loose": find_cross_split_collisions(by_loose, "train", holdouts),
            },
            "candidate_tracks": total,
            "tainted_exact": sorted(tainted),
            "tainted_loose_only": sorted(tainted_loose - tainted),
            "safe_tracks": sorted(str(t.get("track_id")) for t in clean_strict),
            "safe_tracks_exact_fingerprint_only": sorted(str(t.get("track_id")) for t in clean_exact_only),
            "safe_class_hours": dict(hours),
            "safe_class_track_counts": dict(track_counts),
        }
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
