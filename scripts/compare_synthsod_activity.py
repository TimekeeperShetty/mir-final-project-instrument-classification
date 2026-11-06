#!/usr/bin/env python3
"""Compare cached SynthSOD RMS activity with aligned scores without rebuilding the index."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_base.data.utils import OPENMIC_CLASS_TO_INDEX, SYNTHSOD_INSTRUMENT_TO_OPENMIC_CLASS, union_intervals


# Physical instruments from the aligned-score release, not the Slakh/OpenMIC mapping.
# Program 40 contains both violin sections; section identity is not in these TXT files.
PROGRAM_TO_INSTRUMENT = {
    0: "untunedpercussion", 40: "Violin", 41: "Viola", 42: "Cello", 43: "Bass",
    46: "Harp", 47: "Timpani", 56: "Trumpet", 57: "Trombone", 58: "Tuba",
    60: "Horn", 68: "Oboe", 69: "coranglais", 70: "Bassoon", 71: "Clarinet",
    72: "Piccolo", 73: "Flute",
}
SUM_FIELDS = (
    "pieces", "score_active_sec", "rms_active_sec", "overlap_sec", "score_only_sec", "rms_only_sec",
    "crops", "both_active_crops", "score_only_crops", "rms_only_crops", "both_inactive_crops",
)


def instrument_name(name):
    return "Violin" if name in {"Violin", "Violin_1", "Violin_2"} else name


def read_score(path):
    intervals = defaultdict(list)
    rests = 0
    with path.open(encoding="utf-8-sig") as handle:
        header = next(handle, "").split()
        if header != ["Start(secs)", "End(secs)", "MIDI_Pitch", "MIDI_Instrument(0-127)"]:
            raise ValueError(f"Unexpected aligned-score header: {path}")
        for line_number, line in enumerate(handle, 2):
            if not line.strip():
                continue
            start, end, pitch, program = line.split()
            start, end, pitch, program = float(start), float(end), int(pitch), int(program)
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError(f"Invalid score interval at {path}:{line_number}")
            if program == -1:
                rests += 1
                continue
            if program not in PROGRAM_TO_INSTRUMENT or not 0 <= pitch <= 127:
                raise ValueError(f"Unknown instrument/pitch at {path}:{line_number}")
            intervals[PROGRAM_TO_INSTRUMENT[program]].append((start, end))
    return {name: union_intervals(spans) for name, spans in intervals.items()}, rests


def clipped(intervals, duration):
    spans = []
    for start, end in intervals:
        start, end = float(start), float(end)
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"Invalid activity interval: {(start, end)}")
        if start < duration:
            spans.append((start, min(end, duration)))
    return union_intervals(spans)


class Activity:
    """Exact interval integrals, so crop queries do not rescan every note."""

    def __init__(self, intervals):
        self.intervals = union_intervals(intervals)
        self.starts = [start for start, _ in self.intervals]
        self.prefix = [0.0]
        for start, end in self.intervals:
            self.prefix.append(self.prefix[-1] + end - start)

    def until(self, time):
        index = bisect_right(self.starts, time) - 1
        if index < 0:
            return 0.0
        start, end = self.intervals[index]
        return self.prefix[index] + min(time, end) - start

    def within(self, start, end):
        return max(0.0, self.until(end) - self.until(start))


def intersection_seconds(left, right):
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def derived_metrics(stats):
    result = dict(stats)
    result.update(
        active_time_iou=ratio(stats["overlap_sec"], stats["score_active_sec"] + stats["rms_only_sec"]),
        rms_time_precision_vs_score=ratio(stats["overlap_sec"], stats["rms_active_sec"]),
        rms_time_recall_vs_score=ratio(stats["overlap_sec"], stats["score_active_sec"]),
        crop_disagreement_rate=ratio(stats["score_only_crops"] + stats["rms_only_crops"], stats["crops"]),
        positive_crop_disagreement_rate=ratio(
            stats["score_only_crops"] + stats["rms_only_crops"],
            stats["both_active_crops"] + stats["score_only_crops"] + stats["rms_only_crops"],
        ),
        rms_crop_precision_vs_score=ratio(stats["both_active_crops"], stats["both_active_crops"] + stats["rms_only_crops"]),
        rms_crop_recall_vs_score=ratio(stats["both_active_crops"], stats["both_active_crops"] + stats["score_only_crops"]),
    )
    return result


def compare(score_intervals, rms_intervals, crops, min_ratio, rms_stems=None):
    score, rms = Activity(score_intervals), Activity(rms_intervals)
    stems = [Activity(spans) for spans in rms_stems] if rms_stems is not None else None
    overlap = intersection_seconds(score.intervals, rms.intervals)
    stats = {key: 0 for key in SUM_FIELDS}
    stats.update(pieces=1, score_active_sec=score.prefix[-1], rms_active_sec=rms.prefix[-1], overlap_sec=overlap,
                 score_only_sec=max(0.0, score.prefix[-1] - overlap), rms_only_sec=max(0.0, rms.prefix[-1] - overlap))
    disagreements = []
    for start, end in crops:
        score_sec, rms_sec = score.within(start, end), rms.within(start, end)
        required_sec = (end - start) * min_ratio
        score_label = score_sec >= required_sec
        # Remix requires an individual stem to pass, rather than only its class union.
        rms_label = any(stem.within(start, end) >= required_sec for stem in stems) if stems is not None else rms_sec >= required_sec
        stats["crops"] += 1
        field = ("both_active_crops" if score_label and rms_label else
                 "score_only_crops" if score_label else "rms_only_crops" if rms_label else "both_inactive_crops")
        stats[field] += 1
        if score_label != rms_label:
            disagreements.append(dict(start_sec=start, end_sec=end, score_active_sec=score_sec,
                                      rms_union_active_sec=rms_sec, score_label=score_label, rms_label=rms_label))
    return stats, disagreements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rms-index", type=Path, default=Path("/scratch/axs10538/datasets/synthsod/synthsod_index.pt"))
    parser.add_argument("--aligned-root", type=Path, default=Path("/scratch/mg8253/mir_instrument_classification/SynthSOD_aligned_scores"))
    parser.add_argument("--synthsod-root", type=Path, default=Path("/scratch/axs10538/datasets/synthsod/SynthSOD"))
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="train")
    parser.add_argument("--config", type=Path, help="Read clip duration, sample rate, minimum activity ratio and labels from this config")
    parser.add_argument("--crops-per-piece", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Random subset of matched pieces; 0 compares all")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "synthsod_activity_comparison")
    args = parser.parse_args()
    if args.crops_per_piece < 1 or args.limit < 0:
        parser.error("--crops-per-piece must be positive and --limit nonnegative")
    data = {}
    if args.config:
        data = json.loads(args.config.read_text(encoding="utf-8"))["data"]
    sample_rate = int(data.get("sample_rate", 22050))
    clip_sec = float(data.get("clip_duration_sec", 10.0))
    min_ratio = float(data.get("min_activity_ratio", 0.2))
    labels = set(data.get("label_names", OPENMIC_CLASS_TO_INDEX))
    if sample_rate <= 0 or not math.isfinite(clip_sec) or clip_sec <= 0 or not 0 < min_ratio <= 1:
        parser.error("Config must specify a positive sample rate/duration and 0 < min_activity_ratio <= 1")
    clip_samples = int(sample_rate * clip_sec)
    if clip_samples < 1:
        parser.error("Clip duration must contain at least one sample")
    clip_sec = clip_samples / sample_rate
    selected_splits = [args.split] if args.split != "all" else ["train", "validation", "test"]
    aligned, original = {}, {}
    for split in selected_splits:
        raw_split = "evaluation" if split == "validation" else split
        for target, root, filename in (
            (aligned, args.aligned_root, f"SynthSOD_metadata_aligned_{raw_split}.json"),
            (original, args.synthsod_root, f"SynthSOD_metadata_{raw_split}.json"),
        ):
            metadata = json.loads((root.expanduser() / filename).read_text(encoding="utf-8"))
            for song in metadata["songs"].values():
                name = song["song_name"]
                if name in target:
                    raise ValueError(f"Duplicate song across selected metadata splits: {name}")
                target[name] = dict(song, split=split)
    payload = torch.load(args.rms_index.expanduser(), map_location="cpu", weights_only=True)
    if payload.get("source") != "synthsod" or not isinstance(payload.get("tracks"), dict):
        raise ValueError("--rms-index must be a SynthSOD index built by build_synthsod_index.py")
    cached = {}
    for key, track in payload["tracks"].items():
        name = str(track.get("track_id", key)).removeprefix("synthsod/")
        if name in cached:
            raise ValueError(f"Duplicate cached song: {name}")
        cached[name] = track
    matched = sorted(set(aligned) & set(cached))
    if not matched:
        raise ValueError("No aligned songs match the RMS index")
    for name in matched:
        if cached[name]["split"] != aligned[name]["split"]:
            raise ValueError(f"Index and aligned metadata disagree on the split for {name}")
    if args.limit:
        matched = sorted(random.Random(args.seed).sample(matched, min(args.limit, len(matched))))

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    aggregates = defaultdict(lambda: {field: 0 for field in SUM_FIELDS})
    rows, missing_stems, duration_differences, clipped_scores = [], [], [], []
    ignored_rests = 0
    with (output / "crop_disagreements.jsonl").open("w", encoding="utf-8") as detail:
        for number, name in enumerate(matched, 1):
            item, song = cached[name], aligned[name]
            duration = float(item["duration_sec"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"Invalid cached duration for {name}: {duration}")
            score_path = args.aligned_root.expanduser() / "SynthSOD_score_data" / f"{name}.txt"
            score, rests = read_score(score_path)
            ignored_rests += rests
            if any(end > duration for spans in score.values() for _, end in spans):
                clipped_scores.append(name)
            score = {inst: clipped(spans, duration) for inst, spans in score.items()}
            if abs(duration - float(song["duration"])) > 0.1:
                duration_differences.append(dict(song=name, rms_duration_sec=duration, aligned_metadata_duration_sec=song["duration"]))
            rms, rms_stems = defaultdict(list), defaultdict(list)
            for stem in item.get("stems", []):
                inst = instrument_name(stem["instrument"])
                if inst not in PROGRAM_TO_INSTRUMENT.values():
                    raise ValueError(f"Unknown cached instrument {inst!r} in {name}")
                spans = clipped(stem["intervals"], duration)
                rms[inst].extend(spans)
                rms_stems[inst].append(spans)
            for inst in score:
                if inst not in rms:
                    missing_stems.append(dict(song=name, instrument=inst))
            rms = {inst: union_intervals(spans) for inst, spans in rms.items()}
            seed = int.from_bytes(hashlib.sha256(f"{args.seed}|{name}".encode()).digest()[:8], "big")
            rng = random.Random(seed)
            max_offset = max(0, round(duration * sample_rate) - clip_samples)
            offsets = [rng.randrange(max_offset + 1) for _ in range(args.crops_per_piece)] if max_offset else [0]
            crops = [(offset / sample_rate, offset / sample_rate + clip_sec) for offset in offsets]
            class_score, class_rms, class_stems = defaultdict(list), defaultdict(list), defaultdict(list)
            comparisons = []
            for inst in sorted(set(score) | set(rms)):
                comparisons.append(("instrument_union", inst, score.get(inst, []), rms.get(inst, []), None))
                mapped = SYNTHSOD_INSTRUMENT_TO_OPENMIC_CLASS.get("Violin_1" if inst == "Violin" else inst)
                if mapped in labels:
                    class_score[mapped].extend(score.get(inst, []))
                    class_rms[mapped].extend(rms.get(inst, []))
                    class_stems[mapped].extend(rms_stems.get(inst, []))
            for label in sorted(set(class_score) | set(class_rms)):
                comparisons.append(("class_union", label, class_score[label], class_rms[label], None))
                comparisons.append(("remix_eligibility", label, class_score[label], class_rms[label], class_stems[label]))
            for scope, inst, score_spans, rms_spans, stems in comparisons:
                stats, disagreements = compare(score_spans, rms_spans, crops, min_ratio, rms_stems=stems)
                for split in ("all_selected", song["split"]):
                    total = aggregates[(scope, split, inst)]
                    for field in SUM_FIELDS:
                        total[field] += stats[field]
                rows.append(dict(song=name, split=song["split"], scope=scope, instrument=inst, **derived_metrics(stats)))
                for disagreement in disagreements:
                    detail.write(json.dumps(dict(song=name, split=song["split"], scope=scope, instrument=inst,
                                                 violin_section_ambiguous=scope == "remix_eligibility" and inst == "violin",
                                                 **disagreement)) + "\n")
            if number == 1 or number % 25 == 0 or number == len(matched):
                print(f"[{number}/{len(matched)}] {name}", flush=True)
    with (output / "per_piece.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        rms_index=str(args.rms_index), aligned_root=str(args.aligned_root), selected_splits=selected_splits,
        rms_index_activity_settings=payload.get("activity"), training_audio_mic=payload.get("mic"),
        config=str(args.config) if args.config else None,
        clip_duration_sec=clip_sec, sample_rate=sample_rate, min_activity_ratio=min_ratio,
        crops_per_piece=args.crops_per_piece, seed=args.seed, limit=args.limit,
        coverage=dict(original_pieces=len(original), aligned_pieces=len(aligned), compared_pieces=len(matched),
                      pieces_without_aligned_scores=sorted(set(original) - set(aligned)),
                      aligned_pieces_missing_from_rms_index=sorted(set(aligned) - set(cached))),
        score_instruments_missing_from_cached_stems=missing_stems,
        duration_differences_over_100ms=duration_differences, score_pieces_clipped_to_audio_duration=clipped_scores,
        ignored_silence_rows=ignored_rests,
        interpretation=[
            "Scores are an alignment reference, not exact ground truth for acoustic decay or audibility.",
            "Time metrics use exact unions/intersections of intervals in seconds; no frame rounding or shift correction.",
            "Crop metrics use uniform sample-grid offsets before remix class dropping, matching loader crop/activity rules.",
            "Only instruments/classes present in either source for a piece contribute crop comparisons.",
            "Instrument/class union tests use pooled intervals. Remix eligibility requires an individual RMS stem to pass.",
            "Violin score notes combine both sections. Violin remix eligibility uses a combined reference and is not section-exact.",
            "For remix_eligibility, time metrics still describe the union; only crop metrics use the individual-stem rule.",
            "Missing cached stems on matched pieces count as empty RMS activity; missing entire pieces are coverage gaps.",
            "Aligned metadata JSON activity totals are not substituted for TXT note timings. Use aligned release v2.",
        ],
        results=[dict(scope=scope, split=split, instrument=inst, **derived_metrics(stats))
                 for (scope, split, inst), stats in sorted(aggregates.items())],
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\nTraining-class crop disagreement (RMS union vs aligned-score union):")
    for row in summary["results"]:
        if row["scope"] == "class_union" and row["split"] == "all_selected":
            print(f"  {row['instrument']:<18} {row['crop_disagreement_rate']:>7.2%}  "
                  f"score-only={row['score_only_crops']} rms-only={row['rms_only_crops']}")
    print(f"\nCompared {len(matched)} pieces. Reports: {output}")


if __name__ == "__main__":
    main()
