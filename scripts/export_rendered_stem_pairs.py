#!/usr/bin/env python3
"""Export random rendered/original Slakh stems as time-preserving listening pairs."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_base.data.rendered_stems import DEFAULT_RENDER_ROOTS, discover_rendered_stems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", action="append", metavar="INSTRUMENT=PATH",
                        help="Repeat to override the default banjo/mandolin/ukulele roots")
    parser.add_argument("--slakh-root", type=Path,
                        default=Path("/scratch/jnl9728/datasets/mir_final_project/slakh2100_flac_redux"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "rendered_stem_pairs")
    parser.add_argument("--count", type=int, default=20, help="Number of distinct Slakh tracks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instrument", choices=list(DEFAULT_RENDER_ROOTS),
                        help="Only export renders of this instrument")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg with the libmp3lame encoder must be installed/on PATH")

    roots = dict(DEFAULT_RENDER_ROOTS)
    if args.render_root:
        roots = {}
        for entry in args.render_root:
            instrument, separator, path = entry.partition("=")
            if not separator or instrument not in DEFAULT_RENDER_ROOTS or not path:
                parser.error("--render-root must be banjo=PATH, mandolin=PATH, or ukulele=PATH")
            roots[instrument] = path
    if args.instrument:
        if args.instrument not in roots:
            parser.error("--instrument must have a configured --render-root")
        roots = {args.instrument: roots[args.instrument]}

    # First choose distinct tracks, then choose one eligible render per track.
    by_track = {}
    for record in discover_rendered_stems(roots):
        split, track, stem = record["key"]
        source_track = args.slakh_root.expanduser() / split / track
        if not source_track.is_dir() and split == "omitted":
            source_track = args.slakh_root.expanduser() / "ommited" / track
        original = source_track / "stems" / f"{stem}.flac"
        midi = source_track / "MIDI" / f"{stem}.mid"
        if not original.is_file() or not midi.is_file():
            raise FileNotFoundError(f"Original Slakh audio/MIDI missing: {original}, {midi}")
        record = dict(record, original_path=original, local_source_midi=midi)
        by_track.setdefault((split, track), []).append(record)
    if len(by_track) < args.count:
        raise ValueError(f"Requested {args.count} distinct tracks, but only {len(by_track)} have matching renders")
    rng = random.Random(args.seed)
    chosen = [rng.choice(by_track[key]) for key in rng.sample(sorted(by_track), args.count)]
    output_dir = args.output_dir.expanduser().resolve()
    # A separate directory makes each seeded listening set self-contained.
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = []
    manifest_path = output_dir / "manifest.json"
    for number, record in enumerate(chosen, 1):
        split, track, stem = record["key"]
        instrument = record["class_name"]
        pair_dir = output_dir / f"{number:02d}_{split}_{track}_{stem}_{instrument}"
        pair_dir.mkdir()
        base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n"]
        original_mp3 = pair_dir / "original.mp3"
        rendered_mp3 = pair_dir / f"{instrument}.mp3"
        for source, destination in ((record["original_path"], original_mp3),
                                    (record["audio_path"], rendered_mp3)):
            subprocess.run(base + ["-i", str(source), "-map", "0:a:0", "-ac", "1", "-ar", "44100",
                                  "-c:a", "libmp3lame", "-q:a", "2", str(destination)], check=True)
        comparison = pair_dir / "comparison_left_original_right_rendered.mp3"
        # Keep t=0 and leading silence. The shorter input is padded by amix.
        filters = (
            "[0:a:0]aformat=sample_rates=44100:channel_layouts=mono,"
            "pan=stereo|c0=c0|c1=0*c0[left];"
            "[1:a:0]aformat=sample_rates=44100:channel_layouts=mono,"
            "pan=stereo|c0=0*c0|c1=c0[right];"
            "[left][right]amix=inputs=2:duration=longest:normalize=0[out]"
        )
        subprocess.run(base + ["-i", str(record["original_path"]), "-i", str(record["audio_path"]),
                              "-filter_complex", filters, "-map", "[out]", "-c:a", "libmp3lame",
                              "-q:a", "2", str(comparison)], check=True)
        entry = {key: str(value) if isinstance(value, Path) else value for key, value in record.items()}
        entry.update(original_mp3=str(original_mp3), rendered_mp3=str(rendered_mp3), comparison_mp3=str(comparison))
        manifest.append(entry)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[{number}/{len(chosen)}] {pair_dir.name}", flush=True)
    print(f"Exported full-length pairs to {output_dir}; original LEFT, rendered RIGHT.")


if __name__ == "__main__":
    main()
