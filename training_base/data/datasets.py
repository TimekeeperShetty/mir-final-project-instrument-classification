from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
import yaml

from . import audio

# Future OpenMIC target space.
OPENMIC_CLASS_TO_INDEX = {
    "accordion": 0,
    "banjo": 1,
    "bass": 2,
    "cello": 3,
    "clarinet": 4,
    "cymbals": 5,
    "drums": 6,
    "flute": 7,
    "guitar": 8,
    "mallet_percussion": 9,
    "mandolin": 10,
    "organ": 11,
    "piano": 12,
    "saxophone": 13,
    "synthesizer": 14,
    "trombone": 15,
    "trumpet": 16,
    "ukulele": 17,
    "violin": 18,
    "voice": 19,
}

COMMON_OPENMIC_CLASSES = [
    "accordion",
    "banjo",
    "bass",
    "cello",
    "clarinet",
    "drums",
    "flute",
    "guitar",
    "mallet_percussion",
    "organ",
    "piano",
    "saxophone",
    "synthesizer",
    "trombone",
    "trumpet",
    "violin",
    "voice",
]

# Explicit Slakh MIDI-program mapping into the future OpenMIC label indices.
# This is intentionally conservative:
# - we only keep classes that overlap meaningfully with OpenMIC
# - we map by MIDI program number instead of relying on broader GM families
# - classes with no clean Slakh counterpart (cymbals, mandolin, ukulele) are omitted
SLAKH_MIDI_TO_OPENMIC_CLASS = {
    0: "piano",
    1: "piano",
    2: "piano",
    3: "piano",
    4: "piano",
    5: "piano",
    7: "piano",
    11: "mallet_percussion",
    16: "organ",
    17: "organ",
    18: "organ",
    19: "organ",
    20: "organ",
    21: "accordion",
    23: "accordion",
    24: "guitar",
    25: "guitar",
    26: "guitar",
    27: "guitar",
    28: "guitar",
    29: "guitar",
    30: "guitar",
    31: "guitar",
    32: "bass",
    33: "bass",
    34: "bass",
    35: "bass",
    36: "bass",
    37: "bass",
    38: "bass",
    39: "bass",
    42: "cello",
    43: "bass",
    52: "voice",
    53: "voice",
    54: "voice",
    56: "trumpet",
    57: "trombone",
    64: "saxophone",
    65: "saxophone",
    66: "saxophone",
    67: "saxophone",
    71: "clarinet",
    73: "flute",
    75: "flute",
    81: "synthesizer",
    82: "synthesizer",
    83: "synthesizer",
    84: "synthesizer",
    85: "voice",
    86: "synthesizer",
    87: "synthesizer",
    88: "synthesizer",
    89: "synthesizer",
    90: "synthesizer",
    91: "voice",
    92: "synthesizer",
    93: "synthesizer",
    94: "synthesizer",
    95: "synthesizer",
    96: "synthesizer",
    97: "synthesizer",
    98: "synthesizer",
    99: "synthesizer",
    100: "synthesizer",
    101: "synthesizer",
    102: "synthesizer",
    103: "synthesizer",
    105: "banjo",
    110: "violin",
    128: "drums",
}


def program_to_openmic_class_id(program: int, is_drum: bool) -> int:
    if is_drum:
        return OPENMIC_CLASS_TO_INDEX["drums"]
    class_name = SLAKH_MIDI_TO_OPENMIC_CLASS.get(program)
    if class_name is None:
        return -1
    return OPENMIC_CLASS_TO_INDEX[class_name]


def _get_split_cfg_value(split_cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(split_cfg, dict):
        return split_cfg.get(key, default)
    return getattr(split_cfg, key, default)


def _get_split_cfg_params(split_cfg: Any) -> Dict[str, Any]:
    params = _get_split_cfg_value(split_cfg, "params", {}) or {}
    if not isinstance(params, dict):
        raise TypeError(f"Expected split_cfg.params to be a dict, got {type(params)!r}")
    return params


class SlakhDataset(Dataset):
    """
    Slakh dataset aligned to the future OpenMIC label space.

    Expected config:
        split_cfg.params.root: path to the Slakh root directory
        split_cfg.params.split: one of train / validation / val / vallidation / test

    Supported modes:
        - multiclass: one example per mapped stem
        - multilabel: one example per mix with a multi-hot OpenMIC-aligned target
    """

    def __init__(
        self,
        split_cfg: Any,
        task_type: str,
        num_classes: int,
        sample_rate: int,
        clip_num_samples: int,
        train_mode: bool,
    ) -> None:
        if task_type not in {"multiclass", "multilabel", "instrument_classification"}:
            raise ValueError(
                "SlakhDataset currently supports only multiclass or multilabel classification "
                f"(got task_type={task_type!r})"
            )

        params = _get_split_cfg_params(split_cfg)
        root_value = params.get("root")
        if not root_value:
            raise ValueError("SlakhDataset requires split_cfg.params.root")

        self.domain = _get_split_cfg_value(split_cfg, "domain", "synthetic")
        self.task_type = task_type
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.clip_num_samples = clip_num_samples
        self.train_mode = train_mode

        root = Path(root_value).expanduser()
        split = str(params.get("split", "train")).lower()
        split_dir = self._resolve_split_dir(root, split)

        self.samples: List[Dict[str, Any]] = []
        self._build_index(split_dir)

    @staticmethod
    def _resolve_split_dir(root: Path, split: str) -> Path:
        normalized = {
            "train": "train",
            "validation": "validation",
            "val": "validation",
            "vallidation": "vallidation",
            "test": "test",
        }.get(split, split)

        candidate_names = [
            normalized,
            "validation" if normalized == "vallidation" else "vallidation",
            normalized.capitalize(),
            normalized.upper(),
            split,
            split.capitalize(),
            split.upper(),
        ]

        seen = set()
        for candidate_name in candidate_names:
            if candidate_name in seen:
                continue
            seen.add(candidate_name)
            candidate = root / candidate_name
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not find Slakh split directory for split={split!r} under {root}"
        )

    def _build_index(self, split_dir: Path) -> None:
        for track_dir in sorted(split_dir.iterdir()):
            if not track_dir.is_dir():
                continue

            meta_path = track_dir / "metadata.yaml"
            mix_path = track_dir / "mix.flac"
            stems_dir = track_dir / "stems"
            if not (meta_path.exists() and stems_dir.exists()):
                continue

            with open(meta_path, "r", encoding="utf-8") as handle:
                metadata = yaml.safe_load(handle) or {}

            stems_metadata = metadata.get("stems", {})
            if self.task_type in {"multiclass", "instrument_classification"}:
                self._add_multiclass_samples(stems_dir, stems_metadata)
                continue

            if not mix_path.exists():
                continue

            class_ids = set()
            for stem_info in stems_metadata.values():
                program = int(stem_info.get("program_num", -1))
                is_drum = bool(stem_info.get("is_drum", False))
                class_id = program_to_openmic_class_id(program, is_drum=is_drum)
                if 0 <= class_id < self.num_classes:
                    class_ids.add(class_id)

            if not class_ids:
                continue

            target = torch.zeros(self.num_classes, dtype=torch.float32)
            for class_id in class_ids:
                target[class_id] = 1.0

            self.samples.append(
                {
                    "audio_path": mix_path,
                    "label": target,
                }
            )

    def _add_multiclass_samples(self, stems_dir: Path, stems_metadata: Dict[str, Any]) -> None:
        for stem_id, stem_info in stems_metadata.items():
            stem_path = stems_dir / f"{stem_id}.flac"
            if not stem_path.exists():
                continue

            program = int(stem_info.get("program_num", -1))
            is_drum = bool(stem_info.get("is_drum", False))
            class_id = program_to_openmic_class_id(program, is_drum=is_drum)
            if class_id < 0 or class_id >= self.num_classes:
                continue

            self.samples.append(
                {
                    "audio_path": stem_path,
                    "label": class_id,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        waveform = audio.load_waveform(str(item["audio_path"]), self.sample_rate)
        excerpt = audio.trim_or_pad(waveform, self.clip_num_samples, self.train_mode)

        target = item["label"]
        if isinstance(target, torch.Tensor):
            target = target.clone()
        else:
            target = torch.tensor(target, dtype=torch.long)

        return {
            "inputs": excerpt,
            "target": target,
            "domain": self.domain,
        }


class OpenMicDataset(Dataset):
    def __init__(
        self,
        split_cfg: Any,
        task_type: str,
        num_classes: int,
        sample_rate: int,
        clip_num_samples: int,
        train_mode: bool,
    ) -> None:
        raise NotImplementedError(
            "We need to implement the Dataset object for the openmic dataset here"
        )


def build_dataset(
    split_cfg: Any,
    task_type: str,
    num_classes: int,
    sample_rate: int,
    clip_num_samples: int,
    train_mode: bool,
) -> Dataset:
    ds_type = _get_split_cfg_value(split_cfg, "type")
    if ds_type == "slakh":
        return SlakhDataset(
            split_cfg=split_cfg,
            task_type=task_type,
            num_classes=num_classes,
            sample_rate=sample_rate,
            clip_num_samples=clip_num_samples,
            train_mode=train_mode,
        )

    if ds_type == "openmic":
        return OpenMicDataset(
            split_cfg=split_cfg,
            task_type=task_type,
            num_classes=num_classes,
            sample_rate=sample_rate,
            clip_num_samples=clip_num_samples,
            train_mode=train_mode,
        )

    raise ValueError(f"Unsupported dataset type: {ds_type}")
