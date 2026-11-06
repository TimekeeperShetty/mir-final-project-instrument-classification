#!/usr/bin/env python3
"""Build a loader-compatible SynthSOD index from aligned-score pieces only.

Official split assignments are preserved. Violin sections are one logical stem:
their audio is summed together during loading because the score merges program 40.
Use the v2 aligned-score release. The extracted files do not embed a version marker.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

import torch
from pedalboard.io import AudioFile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_synthsod_index import SYNTHSOD_SPLIT_NAMES, resolve_synthsod_root, write_mono_mix
from scripts.compare_synthsod_activity import PROGRAM_TO_INSTRUMENT, clipped, read_score
from training_base.data.utils import (
    OPENMIC_CLASS_TO_INDEX,
    SYNTHSOD_INSTRUMENT_TO_OPENMIC_CLASS,
    materialize_slakh_label_view,
)


DEFAULT_OUTPUT_ROOT = Path("/scratch/mg8253/mir_instrument_classification")


def read_songs(root, filename):
    payload = json.loads((root / filename).read_text(encoding="utf-8"))
    songs = {}
    for song in payload["songs"].values():
        name = song["song_name"]
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid song_name in {filename}: {name!r}")
        if name in songs:
            raise ValueError(f"Duplicate song_name in {filename}: {name}")
        songs[name] = song
    return songs


def read_partitions(root, aligned_root):
    original, aligned = {}, {}
    for raw_split, split in SYNTHSOD_SPLIT_NAMES.items():
        for destination, directory, filename in (
            (original, root, f"SynthSOD_metadata_{raw_split}.json"),
            (aligned, aligned_root, f"SynthSOD_metadata_aligned_{raw_split}.json"),
        ):
            for name, song in read_songs(directory, filename).items():
                if name in destination:
                    raise ValueError(f"Song appears in multiple official splits: {name}")
                destination[name] = dict(song, split=split)
    for name, song in aligned.items():
        if name not in original or original[name]["split"] != song["split"]:
            raise ValueError(f"Aligned metadata disagrees with original split membership: {name}")
        score_path = aligned_root / "SynthSOD_score_data" / f"{name}.txt"
        if not score_path.is_file():
            raise FileNotFoundError(f"Aligned score missing: {score_path}")
    return original, aligned


def index_piece(song, data_root, aligned_root, mix_root, mic):
    name = song["song_name"]
    score_path = aligned_root / "SynthSOD_score_data" / f"{name}.txt"
    score, silence_rows = read_score(score_path)
    audio_dir = data_root / name / mic
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"SynthSOD audio directory missing: {audio_dir}")
    programs = {instrument: program for program, instrument in PROGRAM_TO_INSTRUMENT.items()}
    stems = []
    for instrument, intervals in sorted(score.items()):
        if not intervals:
            continue
        if instrument == "Violin":
            sections = ["Violin_1", "Violin_2"]
            # Require all sections named by the release metadata; keep available
            # sections together even when the original naming is just Violin.
            required = [section for section in sections if section in song.get("instruments_time", {})]
            for section in required:
                if not (audio_dir / f"{section}.flac").is_file():
                    raise FileNotFoundError(f"Violin section missing: {audio_dir / (section + '.flac')}")
            paths = [audio_dir / f"{section}.flac" for section in sections
                     if (audio_dir / f"{section}.flac").is_file()]
            if not paths and (audio_dir / "Violin.flac").is_file():
                paths = [audio_dir / "Violin.flac"]
        else:
            # Ambiguous percussion cannot provide separate drums/cymbals labels.
            if SYNTHSOD_INSTRUMENT_TO_OPENMIC_CLASS.get(instrument) is None:
                continue
            paths = [audio_dir / f"{instrument}.flac"]
        if not paths or any(not path.is_file() for path in paths):
            raise FileNotFoundError(f"Scored instrument {instrument} has missing audio in {audio_dir}")
        stems.append(dict(
            stem_id=instrument, instrument=instrument,
            audio_path=str(paths[0]), audio_paths=[str(path) for path in paths],
            midi_path=None, program_num=programs[instrument], is_drum=False,
            intervals=intervals, score_path=str(score_path), activity_source="aligned_score",
        ))
    if not stems:
        raise ValueError(f"No supported scored stems remain in {name}")

    all_paths = sorted({Path(path) for stem in stems for path in stem["audio_paths"]})
    sample_rates, durations = set(), []
    for path in all_paths:
        with AudioFile(str(path)) as handle:
            if handle.frames <= 0 or handle.samplerate <= 0:
                raise ValueError(f"Empty or invalid audio: {path}")
            sample_rates.add(int(handle.samplerate))
            durations.append(handle.frames / handle.samplerate)
    if len(sample_rates) != 1:
        raise ValueError(f"Sample rates differ within {name}: {sample_rates}")
    duration = max(durations)
    cropped_intervals = 0
    for stem in stems:
        stem_duration = min(durations[all_paths.index(Path(path))] for path in stem["audio_paths"])
        # A grouped violin may contain a shorter section padded with silence. Its
        # combined annotation is checked against the longest member's timeline.
        if len(stem["audio_paths"]) > 1:
            stem_duration = max(durations[all_paths.index(Path(path))] for path in stem["audio_paths"])
        for start, end in stem["intervals"]:
            if end > stem_duration + 0.1:
                raise ValueError(f"Score extends beyond {stem['instrument']} audio in {name}: {end:.3f} > {stem_duration:.3f}s")
            cropped_intervals += end > duration
        stem["intervals"] = clipped(stem["intervals"], duration)

    # This mix contains exactly the retained score-backed stems, including context.
    # It therefore also excludes untunedpercussion when stem_remix is off.
    mix_path = mix_root / name / "mix.flac"
    mix_duration, sample_rate = write_mono_mix(all_paths, mix_path)
    _, class_intervals = materialize_slakh_label_view(stems)
    track_id = f"synthsod/{name}"
    track = dict(
        track_id=track_id, sample_id=track_id, split=song["split"],
        audio_path=str(mix_path), duration_sec=float(mix_duration), sample_rate=sample_rate,
        class_intervals=class_intervals, stems=stems,
        score_path=str(score_path), activity_source="aligned_score",
    )
    audit = dict(
        ignored_silence_rows=silence_rows, intervals_clipped_at_track_end=cropped_intervals,
        grouped_violin_pieces=int(any(stem["instrument"] == "Violin" for stem in stems)),
        untuned_percussion_pieces_excluded_from_audio=int("untunedpercussion" in score),
    )
    return track, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/scratch/axs10538/datasets/synthsod/SynthSOD"))
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "SynthSOD_aligned_scores")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "synthsod_aligned_index.pt")
    parser.add_argument("--mix-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "synthsod_aligned_mixes")
    parser.add_argument("--mic", choices=["Tree", "Close Mic"], default="Tree")
    parser.add_argument("--workers", type=int, default=min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    parser.add_argument("--limit", type=int, default=0, help="Maximum pieces per split for a trial; 0 indexes all aligned pieces")
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        parser.error("--workers must be positive and --limit nonnegative")
    root = resolve_synthsod_root(args.root.expanduser().resolve())
    aligned_root = args.aligned_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    mix_root = args.mix_root.expanduser().resolve()
    report_path = output.with_suffix(".report.json")
    if output.exists() or report_path.exists() or mix_root.exists():
        parser.error("--output, its .report.json companion, and --mix-root must be new paths")
    original, aligned = read_partitions(root, aligned_root)
    songs, selected_counts = [], Counter()
    for name, song in sorted(aligned.items()):
        if not args.limit or selected_counts[song["split"]] < args.limit:
            songs.append(song)
            selected_counts[song["split"]] += 1
    if not songs:
        raise ValueError("Aligned metadata contains no pieces")
    output.parent.mkdir(parents=True, exist_ok=True)
    mix_root.mkdir(parents=True, exist_ok=False)
    tracks, audit_totals = {}, Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(index_piece, song, root / "SynthSOD_data", aligned_root, mix_root, args.mic): song["song_name"]
                   for song in songs}
        for done, future in enumerate(as_completed(futures), 1):
            track, audit = future.result()
            tracks[track["track_id"]] = track
            audit_totals.update(audit)
            if done == 1 or done % 25 == 0 or done == len(futures):
                print(f"[{done}/{len(futures)}] indexed {futures[future]}", flush=True)
    split_counts = Counter(track["split"] for track in tracks.values())
    target_counts, context_only = defaultdict(Counter), Counter()
    for track in tracks.values():
        targets = set(track["class_intervals"]) & set(OPENMIC_CLASS_TO_INDEX)
        target_counts[track["split"]].update(targets)
        context_only[track["split"]] += not targets
    payload = dict(
        root=str(root), source="synthsod", mic=args.mic, split_counts=dict(split_counts),
        activity=dict(method="aligned_scores", aligned_root=str(aligned_root),
                      violin_handling="sections_loaded_together", note_tail_sec=0.0),
        tracks=dict(sorted(tracks.items())),
    )
    torch.save(payload, output)
    report = dict(
        output=str(output), mix_root=str(mix_root), mic=args.mic, trial_limit_per_split=args.limit,
        original_split_counts=dict(Counter(song["split"] for song in original.values())),
        aligned_split_counts=dict(Counter(song["split"] for song in aligned.values())),
        indexed_split_counts=dict(split_counts), target_class_track_counts=dict(target_counts),
        context_only_pieces_filtered_by_20_class_loader=dict(context_only),
        unannotated_pieces_excluded=sorted(set(original) - set(aligned)), audit=dict(audit_totals),
        notes=["Official splits preserved; no RMS fallback and no added note tails.",
               "Violin sections are a single remix unit with union score activity.",
               "Untuned percussion is excluded from both remix stems and generated full mixes.",
               "Use aligned-score release v2; extracted files do not embed a version marker."],
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {len(tracks)} aligned-only pieces to {output}")
    for split in SYNTHSOD_SPLIT_NAMES.values():
        print(f"  {split}: {split_counts[split]} pieces ({context_only[split]} have no 20-class targets)")
    print(f"Excluded {len(set(original) - set(aligned))} pieces without aligned scores.")
    print(f"Coverage/audit report: {report_path}")


if __name__ == "__main__":
    main()
