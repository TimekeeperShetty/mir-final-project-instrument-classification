#!/usr/bin/env python3
"""Load a config's training set and report what the merged pool will feed the model.

Run this before spending GPU time on a config that merges several index files (for
example Slakh plus SynthSOD). It answers:

- how many tracks each source contributes after the current class mapping,
- how many tracks contain each class, and the keep probability the remix gives it,
- what a sample of real training clips looks like: labels per source, how often the
  fallback clip is used, and how long one clip takes to build,
- whether reading only the clip window from a stem gives the same audio as decoding the
  whole stem,
- how long and how much memory one clip from the longest track takes.

Example:
    python scripts/check_training_pool.py \
      --config configs/slakh_synthsod_to_openmic_clap_20_classes.json \
      --num-clips 200
"""
from __future__ import annotations

import argparse
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_base.config import load_json, parse_project_config  # noqa: E402
from training_base.data import audio  # noqa: E402
from training_base.data.datasets import build_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Project config JSON")
    parser.add_argument("--num-clips", type=int, default=200, help="Random training clips to build")
    parser.add_argument("--seed", type=int, default=42, help="Seed for picking tracks and crops")
    return parser.parse_args()


def track_source(item: Dict[str, Any]) -> str:
    # The SynthSOD builder prefixes its track ids; Slakh ids are plain "TrackNNNNN".
    return "synthsod" if str(item.get("track_id", "")).startswith("synthsod/") else "slakh"


def peak_memory_mb() -> float:
    # ru_maxrss is in kilobytes on Linux and in bytes on macOS.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def main() -> None:
    args = parse_args()
    cfg = parse_project_config(load_json(args.config))
    dataset = build_dataset(
        split_cfg=cfg.data.train,
        task_type=cfg.data.task_type,
        num_classes=cfg.data.num_classes,
        sample_rate=cfg.data.sample_rate,
        clip_num_samples=cfg.data.clip_num_samples,
        train_mode=True,
        min_activity_ratio=cfg.data.min_activity_ratio,
        label_names=cfg.data.label_names,
        relevance_threshold=cfg.data.relevance_threshold,
    )
    samples = dataset.samples
    sources = sorted({track_source(item) for item in samples})

    print(f"\nTraining pool: {len(samples)} tracks")
    for source in sources:
        count = sum(track_source(item) == source for item in samples)
        print(f"  {source:<10} {count:>6}  ({count / len(samples):.0%})")

    # Track presence per class and source, next to the keep probability the remix uses.
    presence: Dict[str, Counter] = defaultdict(Counter)
    for item in samples:
        for class_name in item.get("class_intervals", {}):
            presence[class_name][track_source(item)] += 1
    print(f"\n  {'class':<18}" + "".join(f"{source:>10}" for source in sources) + f"{'keep prob':>11}")
    for class_name in dataset.label_names:
        counts = "".join(f"{presence[class_name][source]:>10}" for source in sources)
        print(f"  {class_name:<18}{counts}{dataset.class_keep_probabilities.get(class_name, 1.0):>11.2f}")

    # Build real training clips the same way the dataloader does.
    generator = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    picks = torch.randint(0, len(samples), size=(args.num_clips,), generator=generator).tolist()
    modes: Counter = Counter()
    clips_per_source: Counter = Counter()
    labels_per_source: Dict[str, Counter] = defaultdict(Counter)
    seconds_per_source: Dict[str, float] = defaultdict(float)
    for index in picks:
        item = samples[index]
        source = track_source(item)
        start = time.perf_counter()
        _, target, debug = dataset._build_multilabel_excerpt_and_target(item, index)
        seconds_per_source[source] += time.perf_counter() - start
        clips_per_source[source] += 1
        modes[debug["mode"]] += 1
        for class_index, value in enumerate(target.tolist()):
            if value >= 0.5:
                labels_per_source[source][dataset.label_names[class_index]] += 1

    print(f"\n{args.num_clips} random training clips")
    print("  clip modes: " + ", ".join(f"{mode} {count}" for mode, count in modes.most_common()))
    for source in sources:
        clips = clips_per_source[source]
        if not clips:
            continue
        print(f"  {source}: {clips} clips, {seconds_per_source[source] / clips:.2f} s per clip")
        print("    labelled: " + ", ".join(f"{name} {count}" for name, count in labels_per_source[source].most_common()))

    # Clips read only their window from each stem (audio.load_waveform_excerpt). Check that
    # it returns the same samples as decoding the whole stem with load_waveform and slicing,
    # on a few short stems from each source.
    worst_difference = 0.0
    compared = 0
    for source in sources:
        short_items = sorted(
            (item for item in samples if track_source(item) == source and item.get("stems")),
            key=lambda item: float(item.get("duration_sec", 0.0)),
        )[:3]
        for item in short_items:
            stem = item["stems"][0]
            full = audio.load_waveform(str(stem["audio_path"]), dataset.sample_rate)
            offset = max(0, (len(full) - dataset.clip_num_samples) // 2)
            expected = full[offset : offset + dataset.clip_num_samples]
            excerpt = audio.load_waveform_excerpt(
                str(stem["audio_path"]), dataset.sample_rate, offset, dataset.clip_num_samples
            )
            if excerpt.shape != expected.shape:
                raise SystemExit(f"Excerpt length differs for {stem['audio_path']}: {excerpt.shape} vs {expected.shape}")
            worst_difference = max(worst_difference, float((excerpt - expected).abs().max()))
            compared += 1
    print(f"\nExcerpt loader vs full decode on {compared} stems: worst sample difference {worst_difference:.1e}")

    # The longest track is the worst case for time and memory per dataloader worker.
    longest = max(samples, key=lambda item: float(item.get("duration_sec", 0.0)))
    memory_before = peak_memory_mb()
    start = time.perf_counter()
    _, _, debug = dataset._build_multilabel_excerpt_and_target(longest, 0)
    elapsed = time.perf_counter() - start
    print(
        f"\nLongest track: {longest['track_id']} ({float(longest['duration_sec']) / 60:.1f} min), "
        f"{len(debug['selected_stems'])} stems in the clip"
    )
    print(f"  one clip took {elapsed:.1f} s; peak memory {memory_before:.0f} MB before, {peak_memory_mb():.0f} MB after")


if __name__ == "__main__":
    main()
