#!/usr/bin/env python3
"""Build a SynthSOD index in the same format as the Slakh MIDI index.

SlakhDataset reads cached index files, not dataset folders, and it accepts a list of them
in `index_paths`. This script turns SynthSOD into one more index file, so its pieces join
the same training pool as the Slakh tracks.

Two things differ from the Slakh builder:

- Labels: SynthSOD stems are named by instrument (Cello.flac, Violin_1.flac), not by GM
  program. Each stem stores that name under "instrument", and the mapping to OpenMIC
  classes (SYNTHSOD_INSTRUMENT_TO_OPENMIC_CLASS) is applied at load time.
- Activity intervals: SynthSOD ships no MIDI, so intervals come from the stem audio by
  energy thresholding. The metadata JSONs do record how many seconds each instrument is
  active in each piece, and the script compares its intervals against those totals at the
  end so the threshold can be checked.

SynthSOD also has no mix file, so the script sums the stems of each piece into a mono
mix.flac under --mix-root. Training with stem_remix only needs that mix for its rare
fallback path, but the index format requires one per track.

Example:
    python scripts/build_synthsod_index.py \
      --root /scratch/axs10538/datasets/synthsod/SynthSOD \
      --output /scratch/axs10538/datasets/synthsod/synthsod_index.pt \
      --mix-root /scratch/axs10538/datasets/synthsod/mixes \
      --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pedalboard.io import AudioFile

# When this file is run as `python scripts/build_synthsod_index.py`, Python puts
# `scripts/` on sys.path, not the repo root. We add the repo root explicitly so the
# `training_base` package can be imported reliably.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_base.data.utils import (  # noqa: E402
    SYNTHSOD_INSTRUMENT_TO_OPENMIC_CLASS,
    materialize_slakh_label_view,
)


# SynthSOD calls its validation split "evaluation". We store the name SlakhDataset uses.
SYNTHSOD_SPLIT_NAMES = {
    "train": "train",
    "evaluation": "validation",
    "test": "test",
}

# Frames are read in blocks of this many hops, so memory stays flat for long pieces.
READ_BLOCK_HOPS = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        required=True,
        help="Folder holding SynthSOD_data/ and the SynthSOD_metadata_*.json files",
    )
    parser.add_argument("--output", required=True, help="Output .pt path")
    parser.add_argument("--mix-root", required=True, help="Folder to write one mix.flac per piece into")
    parser.add_argument(
        "--mic",
        default="Tree",
        choices=["Tree", "Close Mic"],
        help="Which microphone version the training audio comes from",
    )
    parser.add_argument(
        "--activity-mic",
        default="Close Mic",
        choices=["Tree", "Close Mic"],
        help="Which microphone version the activity intervals are measured on. Close Mic has "
        "less reverb tail, so its note ends are sharper; both versions are time-aligned.",
    )
    parser.add_argument("--hop-sec", type=float, default=0.05, help="RMS frame length in seconds")
    parser.add_argument(
        "--threshold-db",
        type=float,
        default=-40.0,
        help="A frame is active if its RMS is within this many dB of the stem's loudest frame",
    )
    parser.add_argument(
        "--floor-db",
        type=float,
        default=-70.0,
        help="Frames quieter than this absolute level (dBFS) are never active",
    )
    parser.add_argument(
        "--min-gap-sec",
        type=float,
        default=0.2,
        help="Silent gaps shorter than this are merged into the surrounding activity",
    )
    parser.add_argument(
        "--min-segment-sec",
        type=float,
        default=0.1,
        help="Active segments shorter than this, after gap merging, are dropped",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
        help="Pieces processed in parallel (default: SLURM_CPUS_PER_TASK, else 4)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only index this many pieces per split, for a quick trial run (0 = all)",
    )
    parser.add_argument("--print-every", type=int, default=25, help="Progress print frequency")
    return parser.parse_args()


def resolve_synthsod_root(root: Path) -> Path:
    # Accept either the folder the zip was extracted into or the SynthSOD/ folder inside it.
    for candidate in (root, root / "SynthSOD"):
        if (candidate / "SynthSOD_data").is_dir():
            return candidate
    raise FileNotFoundError(f"No SynthSOD_data folder found under {root} or {root / 'SynthSOD'}")


def read_split_songs(root: Path) -> List[Tuple[str, Dict[str, Any]]]:
    # Each split JSON lists its songs under "songs". The song folder on disk is named
    # after song_name; the JSON's own "folder" field does not match the zip layout.
    songs: List[Tuple[str, Dict[str, Any]]] = []
    for raw_split, split_name in SYNTHSOD_SPLIT_NAMES.items():
        meta_path = root / f"SynthSOD_metadata_{raw_split}.json"
        with open(meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        for song in metadata["songs"].values():
            songs.append((split_name, song))
    return songs


def frame_rms_db(path: Path, hop_sec: float) -> Tuple[np.ndarray, float, int]:
    # Returns per-frame RMS in dBFS (mono), the hop length actually used in seconds,
    # and the file's length in samples.
    with AudioFile(str(path)) as handle:
        sample_rate = int(handle.samplerate)
        hop = max(1, int(round(hop_sec * sample_rate)))
        num_frames = int(handle.frames)
        rms_blocks: List[np.ndarray] = []
        while handle.tell() < num_frames:
            block = handle.read(hop * READ_BLOCK_HOPS).mean(axis=0)
            usable = (len(block) // hop) * hop
            if usable < len(block):
                # Only the very last block can be partial; pad it to a whole frame.
                block = np.pad(block, (0, usable + hop - len(block)))
            frames = block.reshape(-1, hop)
            rms_blocks.append(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)))
    rms = np.concatenate(rms_blocks) if rms_blocks else np.zeros(0)
    return 20.0 * np.log10(rms + 1e-10), hop / sample_rate, num_frames


def energy_activity_intervals(
    rms_db: np.ndarray,
    hop_sec: float,
    *,
    threshold_db: float,
    floor_db: float,
    min_gap_sec: float,
    min_segment_sec: float,
) -> List[Tuple[float, float]]:
    # This plays the role midi_note_intervals plays for Slakh: it turns one stem into the
    # list of (start_sec, end_sec) spans where the instrument is sounding.
    if rms_db.size == 0:
        return []
    level = max(float(rms_db.max()) + threshold_db, floor_db)
    active = rms_db > level
    if not active.any():
        return []

    # Find runs of active frames: +1 marks a run start, -1 the frame after a run ends.
    edges = np.diff(np.concatenate(([0], active.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    merged: List[List[float]] = []
    for start, end in zip(starts, ends):
        start_sec, end_sec = start * hop_sec, end * hop_sec
        if merged and start_sec - merged[-1][1] < min_gap_sec:
            merged[-1][1] = end_sec
        else:
            merged.append([start_sec, end_sec])
    return [(start_sec, end_sec) for start_sec, end_sec in merged if end_sec - start_sec >= min_segment_sec]


def write_mono_mix(stem_paths: List[Path], mix_path: Path) -> Tuple[float, int]:
    # Sum the stems one at a time into a mono buffer, peak-limit it, and write it out.
    # Returns the mix duration in seconds and the sample rate.
    mix = np.zeros(0, dtype=np.float32)
    sample_rate = 0
    for stem_path in stem_paths:
        with AudioFile(str(stem_path)) as handle:
            if sample_rate and int(handle.samplerate) != sample_rate:
                raise ValueError(f"Sample rate mismatch inside one piece: {stem_path}")
            sample_rate = int(handle.samplerate)
            num_frames = int(handle.frames)
            if num_frames > len(mix):
                # Stems of one piece should be equally long; pad in case one is not.
                mix = np.pad(mix, (0, num_frames - len(mix)))
            # Read in blocks so a long stereo stem never sits in memory all at once.
            block_frames = sample_rate * 30
            while handle.tell() < num_frames:
                position = handle.tell()
                block = handle.read(block_frames).mean(axis=0)
                mix[position : position + len(block)] += block

    peak = float(np.abs(mix).max())
    if peak > 0.99:
        # Summed orchestral stems can clip. The dataset peak-normalizes crops anyway, so
        # scaling the whole mix down only matters for the 16-bit file.
        mix *= 0.99 / peak

    mix_path.parent.mkdir(parents=True, exist_ok=True)
    with AudioFile(str(mix_path), "w", samplerate=sample_rate, num_channels=1, bit_depth=16) as handle:
        handle.write(mix[np.newaxis, :])
    return len(mix) / sample_rate, sample_rate


def index_one_piece(
    split_name: str,
    song: Dict[str, Any],
    data_dir: Path,
    mix_root: Path,
    args_dict: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    song_name = str(song["song_name"])
    audio_dir = data_dir / song_name / args_dict["mic"]
    activity_dir = data_dir / song_name / args_dict["activity_mic"]
    stem_paths = sorted(audio_dir.glob("*.flac"))
    if not stem_paths:
        return None

    metadata_active_sec = song.get("instruments_time", {})
    stems: List[Dict[str, Any]] = []
    for stem_path in stem_paths:
        instrument = stem_path.stem
        activity_path = activity_dir / stem_path.name
        if not activity_path.exists():
            # Every Tree stem has a Close Mic twin in the released zip; fall back to the
            # audio mic rather than dropping the stem if one is ever missing.
            activity_path = stem_path
        rms_db, hop_sec, _ = frame_rms_db(activity_path, args_dict["hop_sec"])
        intervals = energy_activity_intervals(
            rms_db,
            hop_sec,
            threshold_db=args_dict["threshold_db"],
            floor_db=args_dict["floor_db"],
            min_gap_sec=args_dict["min_gap_sec"],
            min_segment_sec=args_dict["min_segment_sec"],
        )
        if not intervals:
            continue
        stems.append(
            {
                "stem_id": instrument,
                "instrument": instrument,
                "audio_path": str(stem_path),
                "midi_path": None,
                "program_num": -1,
                "is_drum": False,
                "intervals": [(float(start_sec), float(end_sec)) for start_sec, end_sec in intervals],
                # Kept so the activity threshold can be re-checked later without the JSONs.
                "metadata_active_sec": float(metadata_active_sec.get(instrument, 0.0)),
            }
        )
    if not stems:
        return None

    mix_path = mix_root / song_name / "mix.flac"
    duration_sec, _ = write_mono_mix(stem_paths, mix_path)

    # As in the Slakh builder, class_intervals is only for reporting; the dataset rebuilds
    # it from the raw stems with whatever mapping is current at load time.
    _, class_intervals = materialize_slakh_label_view(stems)
    track_id = f"synthsod/{song_name}"
    return {
        "track_id": track_id,
        "sample_id": track_id,
        "split": split_name,
        "audio_path": str(mix_path),
        "duration_sec": float(duration_sec),
        "class_intervals": {
            str(class_name): [(float(start_sec), float(end_sec)) for start_sec, end_sec in intervals]
            for class_name, intervals in class_intervals.items()
        },
        "stems": stems,
    }


def print_activity_report(tracks: Dict[str, Dict[str, Any]]) -> None:
    # Compare the derived activity against SynthSOD's own per-instrument totals. A ratio
    # near 1.0 means the threshold agrees with the dataset; well above 1.0 means reverb
    # tails or quiet bleed are counted as playing, well below means quiet passages are missed.
    derived: Dict[str, float] = defaultdict(float)
    reference: Dict[str, float] = defaultdict(float)
    outliers: Dict[str, int] = defaultdict(int)
    counts: Dict[str, int] = defaultdict(int)
    for track in tracks.values():
        for stem in track["stems"]:
            instrument = stem["instrument"]
            active = sum(end_sec - start_sec for start_sec, end_sec in stem["intervals"])
            expected = stem["metadata_active_sec"]
            derived[instrument] += active
            reference[instrument] += expected
            counts[instrument] += 1
            if expected > 0 and not 0.8 <= active / expected <= 1.25:
                outliers[instrument] += 1

    print("\nActivity check against SynthSOD metadata (hours):")
    print(f"  {'instrument':<20} {'class':<20} {'derived':>8} {'metadata':>9} {'ratio':>6} {'stems off >25%':>15}")
    for instrument in sorted(reference, key=lambda name: -reference[name]):
        ratio = derived[instrument] / reference[instrument] if reference[instrument] > 0 else float("nan")
        class_name = SYNTHSOD_INSTRUMENT_TO_OPENMIC_CLASS.get(instrument, "(unmapped)")
        print(
            f"  {instrument:<20} {class_name:<20} {derived[instrument] / 3600:>8.2f} "
            f"{reference[instrument] / 3600:>9.2f} {ratio:>6.2f} "
            f"{outliers[instrument]:>6} of {counts[instrument]:<5}"
        )
    unmapped = sorted(name for name in reference if name not in SYNTHSOD_INSTRUMENT_TO_OPENMIC_CLASS)
    if unmapped:
        print(f"\nWARNING: instruments missing from SYNTHSOD_INSTRUMENT_TO_OPENMIC_CLASS: {unmapped}")


def main() -> None:
    args = parse_args()

    root = resolve_synthsod_root(Path(args.root).expanduser())
    data_dir = root / "SynthSOD_data"
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mix_root = Path(args.mix_root).expanduser()

    songs = read_split_songs(root)
    if args.limit > 0:
        per_split: Dict[str, int] = defaultdict(int)
        limited = []
        for split_name, song in songs:
            if per_split[split_name] < args.limit:
                limited.append((split_name, song))
                per_split[split_name] += 1
        songs = limited

    args_dict = {
        "mic": args.mic,
        "activity_mic": args.activity_mic,
        "hop_sec": args.hop_sec,
        "threshold_db": args.threshold_db,
        "floor_db": args.floor_db,
        "min_gap_sec": args.min_gap_sec,
        "min_segment_sec": args.min_segment_sec,
    }

    all_tracks: Dict[str, Dict[str, Any]] = {}
    skipped: List[str] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(index_one_piece, split_name, song, data_dir, mix_root, args_dict): song["song_name"]
            for split_name, song in songs
        }
        for done, future in enumerate(as_completed(futures), start=1):
            song_name = futures[future]
            track = future.result()
            if track is None:
                skipped.append(song_name)
            else:
                all_tracks[track["track_id"]] = track
            if done == 1 or done % args.print_every == 0 or done == len(futures):
                print(f"[{done}/{len(futures)}] indexed {song_name}", flush=True)

    split_counts: Dict[str, int] = defaultdict(int)
    for track in all_tracks.values():
        split_counts[track["split"]] += 1

    payload = {
        "root": str(root),
        "source": "synthsod",
        "mic": args.mic,
        "activity": dict(args_dict),
        "split_counts": dict(split_counts),
        "tracks": all_tracks,
    }
    torch.save(payload, output_path)

    print(f"\nSaved SynthSOD index with {len(all_tracks)} pieces to {output_path}")
    for split_name, count in split_counts.items():
        print(f"  {split_name}: {count}")
    if skipped:
        print(f"Skipped {len(skipped)} pieces with no audio or no detected activity: {skipped[:10]}")
    print_activity_report(all_tracks)


if __name__ == "__main__":
    main()
