#!/usr/bin/env python3
"""Export random SynthSOD stem crops with labels from the original RMS index."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_base.data import audio
from training_base.data.utils import (
    OPENMIC_CLASS_TO_INDEX,
    overlap_duration_sec,
    stem_to_openmic_class_name,
    union_intervals,
)


def write_mp3(ffmpeg, waveform, sample_rate, path):
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
         "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
         "-c:a", "libmp3lame", "-q:a", "2", str(path)],
        input=waveform.detach().cpu().numpy().astype("<f4", copy=False).tobytes(),
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rms-index", type=Path, default=Path("/scratch/axs10538/datasets/synthsod/synthsod_index.pt"))
    parser.add_argument("--synthsod-root", type=Path, help="Override the audio root stored in the index")
    parser.add_argument("--config", type=Path, help="Read sample rate, clip duration, minimum activity ratio and label names")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "rendered_stem_pairs" / "synthsod_rms")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="train")
    parser.add_argument("--count", type=int, default=20, help="Number of distinct random pieces")
    parser.add_argument("--duration-sec", type=float, help="Override the clip duration (default: config value or 10 seconds)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg with libmp3lame must be installed/on PATH")
    data = json.loads(args.config.read_text(encoding="utf-8"))["data"] if args.config else {}
    sample_rate = int(data.get("sample_rate", 22050))
    duration = args.duration_sec if args.duration_sec is not None else float(data.get("clip_duration_sec", 10.0))
    min_ratio = float(data.get("min_activity_ratio", 0.2))
    label_names = list(data.get("label_names", OPENMIC_CLASS_TO_INDEX))
    allowed_labels = set(label_names)
    if args.count < 1 or sample_rate <= 0 or not math.isfinite(duration) or duration <= 0 or not 0 < min_ratio <= 1:
        parser.error("Count, sample rate and duration must be positive; 0 < min_activity_ratio <= 1")
    if allowed_labels - set(OPENMIC_CLASS_TO_INDEX):
        parser.error("Config contains unsupported target label names")
    num_samples = int(duration * sample_rate)
    if num_samples < 1:
        parser.error("Duration must contain at least one audio sample")
    duration = num_samples / sample_rate
    payload = torch.load(args.rms_index.expanduser(), map_location="cpu", weights_only=True)
    if payload.get("source") != "synthsod" or not isinstance(payload.get("tracks"), dict):
        raise ValueError("Use the original SynthSOD RMS index")
    activity_settings = payload.get("activity") or {}
    if activity_settings.get("method") not in (None, "rms"):
        raise ValueError("This preview script requires an RMS index, not the aligned-score index")
    root = (args.synthsod_root or Path(payload["root"])).expanduser().resolve()
    if not (root / "SynthSOD_data").is_dir() and (root / "SynthSOD" / "SynthSOD_data").is_dir():
        root = root / "SynthSOD"
    mic = payload.get("mic", "Tree")
    if mic not in {"Tree", "Close Mic"}:
        raise ValueError(f"Unknown audio microphone perspective in index: {mic!r}")
    candidates = [(key, item) for key, item in sorted(payload["tracks"].items())
                  if args.split == "all" or item["split"] == args.split]
    if len(candidates) < args.count:
        raise ValueError(f"Requested {args.count} pieces, but only {len(candidates)} match split={args.split}")
    rng = random.Random(args.seed)
    chosen = rng.sample(candidates, args.count)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        rms_index=str(args.rms_index), audio_root=str(root), audio_mic=mic,
        rms_activity_mic=activity_settings.get("activity_mic"), rms_activity_settings=activity_settings,
        seed=args.seed, split=args.split, clip_duration_sec=duration, sample_rate=sample_rate,
        min_activity_ratio=min_ratio, target_label_order=label_names,
        note="Random crops before balancing, stem dropping, context sampling or augmentation. Inactive stems are exported too.",
        examples=[],
    )
    for number, (key, item) in enumerate(chosen, 1):
        song = str(item.get("track_id", key)).removeprefix("synthsod/")
        if not song or Path(song).name != song or song in {".", ".."}:
            raise ValueError(f"Invalid song name: {song!r}")
        track_duration = float(item["duration_sec"])
        if not math.isfinite(track_duration) or track_duration <= 0:
            raise ValueError(f"Invalid cached track duration for {song}")
        max_offset = max(0, round(track_duration * sample_rate) - num_samples)
        offset = rng.randrange(max_offset + 1)
        start, end = offset / sample_rate, offset / sample_rate + duration
        audio_dir = root / "SynthSOD_data" / song / mic
        paths = sorted(audio_dir.glob("*.flac"))
        if not paths:
            raise FileNotFoundError(f"No stems found: {audio_dir}")
        cached = {}
        for stem in item.get("stems", []):
            instrument = stem["instrument"]
            if instrument in cached:
                raise ValueError(f"Duplicate cached instrument {instrument!r} in {song}")
            cached[instrument] = stem
        absent_files = set(cached) - {path.stem for path in paths}
        if absent_files:
            raise FileNotFoundError(f"Indexed stems missing under {audio_dir}: {sorted(absent_files)}")
        example_dir = output / f"{number:02d}_{song}_{start:.3f}s"
        example_dir.mkdir()
        class_intervals = defaultdict(list)
        for instrument, stem in cached.items():
            mapped = stem_to_openmic_class_name(stem)
            if mapped in allowed_labels:
                class_intervals[mapped].extend(stem.get("intervals", []))

        def passes(mapped, active_sec):
            return active_sec > 0 if mapped == "drums" else active_sec >= duration * min_ratio

        active_labels = [label for label in label_names if label in class_intervals and passes(
            label, overlap_duration_sec(union_intervals(class_intervals[label]), start, end)
        )]
        full_mix = torch.zeros(num_samples, dtype=torch.float32)
        eligible_mix = torch.zeros_like(full_mix)
        stem_rows, waveforms, eligible_labels = [], [], set()
        for path in paths:
            instrument = path.stem
            cached_stem = cached.get(instrument)
            mapped = stem_to_openmic_class_name({"instrument": instrument})
            spans = union_intervals(cached_stem.get("intervals", [])) if cached_stem else []
            active_sec = overlap_duration_sec(spans, start, end)
            stem_active = passes(mapped, active_sec)
            role = "target" if mapped in allowed_labels else "context" if mapped is not None else "dropped"
            status = "not_in_rms_index" if cached_stem is None else "rms_active" if stem_active else "rms_inactive"
            waveform = audio.trim_or_pad(
                audio.load_waveform_excerpt(str(path), sample_rate, offset, num_samples), num_samples, random_crop=False
            )
            if not torch.isfinite(waveform).all():
                raise ValueError(f"Non-finite audio in {path} at {start:.3f}s")
            full_mix += waveform
            if role == "target" and stem_active:
                eligible_mix += waveform
                eligible_labels.add(mapped)
            mp3_name = f"{instrument}__{status}__{role}.mp3"
            waveforms.append((waveform, mp3_name))
            stem_rows.append(dict(
                instrument=instrument, mapped_class=mapped, role=role, rms_status=status,
                in_rms_index=cached_stem is not None, rms_active_sec=active_sec,
                rms_active_ratio=active_sec / duration, passes_stem_activity_rule=stem_active,
                source_audio=str(path), mp3=mp3_name,
                rms_intervals_in_clip_sec=[(max(a, start) - start, min(b, end) - start)
                                           for a, b in spans if b > start and a < end],
            ))
        # Share one attenuation across all exports, preserving relative stem levels.
        peak = max([1.0, float(full_mix.abs().max()), float(eligible_mix.abs().max())]
                   + [float(waveform.abs().max()) for waveform, _ in waveforms])
        gain = 1.0 / peak
        for waveform, filename in waveforms:
            write_mp3(ffmpeg, waveform * gain, sample_rate, example_dir / filename)
        write_mp3(ffmpeg, full_mix * gain, sample_rate, example_dir / "full_mix.mp3")
        write_mp3(ffmpeg, eligible_mix * gain, sample_rate, example_dir / "rms_eligible_target_mix.mp3")
        example = dict(
            song=song, split=item["split"], directory=example_dir.name,
            crop_start_sample=offset, crop_start_sec=start, crop_end_sec=end, audio_mic=mic,
            min_active_sec=duration * min_ratio, active_labels=active_labels,
            stem_eligible_labels=[label for label in label_names if label in eligible_labels],
            shared_export_gain=gain, stems=stem_rows,
            full_mix="full_mix.mp3", eligible_target_mix="rms_eligible_target_mix.mp3",
        )
        (example_dir / "labels.json").write_text(json.dumps(example, indent=2) + "\n", encoding="utf-8")
        lines = [
            f"Song: {song} ({item['split']}); audio: {mic}",
            f"Original timeline: {start:.6f} to {end:.6f} seconds",
            f"Minimum activity: {duration * min_ratio:.3f}s (drums: any overlap)",
            f"RMS class-union labels: {', '.join(active_labels) or '(none)'}",
            f"RMS stem-eligible target labels: {', '.join(example['stem_eligible_labels']) or '(none)'}",
            "full_mix.mp3: all original stems, including inactive/context/dropped instruments.",
            "rms_eligible_target_mix.mp3: only target stems passing RMS, before balancing/augmentation.",
            "Listen to rms_inactive and not_in_rms_index stems to check missed activity.",
            "Both violin sections contribute to the class union; stem eligibility is checked separately.",
            "", "Instrument | mapped class | role | RMS status | active seconds",
        ]
        lines.extend(f"{row['instrument']} | {row['mapped_class']} | {row['role']} | {row['rms_status']} | {row['rms_active_sec']:.3f}"
                     for row in stem_rows)
        (example_dir / "labels.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest["examples"].append(example)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[{number}/{len(chosen)}] {song} {start:.3f}-{end:.3f}s: {', '.join(active_labels) or '(no active labels)'}", flush=True)
    print(f"\nExported {len(chosen)} RMS preview crops to {output}")


if __name__ == "__main__":
    main()
