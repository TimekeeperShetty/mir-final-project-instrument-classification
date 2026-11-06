"""Discover rendered Slakh stems using their provenance, without audio dependencies."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


DEFAULT_RENDER_ROOTS = {
    name: f"/scratch/jnl9728/synth_{name}"
    for name in ("banjo", "mandolin", "ukulele")
}


def source_midi_key(path: Path) -> Tuple[str, str, str]:
    """Identify a stem independently of the machine's Slakh root directory."""
    if path.parent.name != "MIDI" or not path.parent.parent.name.startswith("Track"):
        raise ValueError(f"Expected .../<split>/TrackNNNNN/MIDI/SNN.mid: {path}")
    split = path.parent.parent.parent.name.lower()
    split = {"ommited": "omitted", "val": "validation", "vallidation": "validation"}.get(split, split)
    return split, path.parent.parent.name, path.stem


def discover_rendered_stems(roots: Mapping[str, str]) -> list[Dict[str, Any]]:
    records = []
    seen = set()
    for instrument, raw_root in sorted(roots.items()):
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Rendered stem directory not found: {root}")
        sidecars = sorted(root.rglob("*.provenance.json"))
        if not sidecars:
            raise ValueError(f"No rendered stem provenance files found under {root}")
        for provenance_path in sidecars:
            with provenance_path.open(encoding="utf-8") as handle:
                provenance = json.load(handle)
            if provenance.get("target_instrument") != instrument:
                raise ValueError(f"Expected target_instrument={instrument!r}: {provenance_path}")
            source_midi = Path(provenance["source_midi"])
            key = source_midi_key(source_midi)
            # Use the actual neighboring file, even if the render directory was moved.
            rendered_path = provenance_path.with_name(
                provenance_path.name.removesuffix(".provenance.json") + ".flac"
            )
            expected_name = f"{key[1]}_{key[2]}.{instrument}.flac"
            if rendered_path.name != expected_name:
                raise ValueError(f"Filename disagrees with source MIDI: {provenance_path}")
            if not rendered_path.is_file():
                raise FileNotFoundError(f"Rendered audio missing for {provenance_path}: {rendered_path}")
            if (key, instrument) in seen:
                raise ValueError(f"Duplicate {instrument} render for {key}: {provenance_path}")
            seen.add((key, instrument))
            records.append({
                "key": key,
                "class_name": instrument,
                "audio_path": rendered_path,
                "source_midi": source_midi,
                "provenance_path": provenance_path,
                "remap_stats": provenance.get("remap_stats", {}),
            })
    return records
