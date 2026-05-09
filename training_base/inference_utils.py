from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .config import load_json, parse_project_config
from .data import audio as audio_utils
from .models.system import DomainTransferSystem


def load_model_bundle(config_path: str | Path, checkpoint_path: str | Path):
    cfg = parse_project_config(load_json(str(config_path)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DomainTransferSystem.load_from_checkpoint(
        str(checkpoint_path),
        cfg=cfg,
        map_location=device,
    )
    model = model.to(device).eval()
    return model, cfg, device


def prepare_waveform(audio_path: str | Path, cfg) -> torch.Tensor:
    waveform = audio_utils.load_waveform(str(audio_path), cfg.data.sample_rate)
    waveform = audio_utils.trim_or_pad(
        waveform,
        cfg.data.clip_num_samples,
        random_crop=False,
    )
    return waveform


@torch.no_grad()
def predict_file(audio_path: str | Path, model, cfg, device) -> Dict[str, Any]:
    waveform = prepare_waveform(audio_path, cfg)
    inputs = waveform.unsqueeze(0).to(device)
    lengths = torch.tensor([inputs.shape[-1]], dtype=torch.long, device=device)

    logits = model(inputs, lengths)
    if cfg.data.task_type == "multiclass":
        probs = torch.softmax(logits, dim=-1)[0].cpu()
    else:
        probs = torch.sigmoid(logits)[0].cpu()

    rows = [
        {"label": label, "probability": float(prob)}
        for label, prob in zip(cfg.data.label_names, probs.tolist())
    ]
    rows = sorted(rows, key=lambda row: row["probability"], reverse=True)
    return {
        "waveform": waveform,
        "predictions": rows,
    }


def load_sample_metadata(samples_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    samples_dir = Path(samples_dir)
    metadata_path = samples_dir / "metadata.json"
    if not metadata_path.exists():
        return {}

    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    by_name: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        exported_path = Path(row["exported_wav_path"])
        key = exported_path.name
        by_name[key] = row
    return by_name


def list_audio_samples(samples_dir: str | Path) -> List[Path]:
    samples_dir = Path(samples_dir)
    audio_paths = sorted(
        path
        for path in samples_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".wav", ".ogg", ".mp3", ".flac"}
    )
    return audio_paths


def build_prediction_table_rows(
    predictions: List[Dict[str, Any]],
    ground_truth_labels: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    gt_set = set(ground_truth_labels or [])
    rows: List[Dict[str, Any]] = []
    for row in predictions:
        probability = float(row["probability"])
        label = str(row["label"])
        rows.append(
            {
                "label": label,
                "probability": probability,
                "predicted": probability >= threshold,
                "ground_truth": label in gt_set,
            }
        )
    return rows
