from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import torchaudio
except ImportError:  # pragma: no cover - depends on local environment
    torchaudio = None


def require_torchaudio() -> None:
    if torchaudio is None:
        raise ImportError(
            "torchaudio is required for waveform datasets. "
            "Install torchaudio or switch the split to input_kind='embedding'."
        )


def load_waveform(path: str, target_sample_rate: int) -> torch.Tensor:
    require_torchaudio()
    waveform, sample_rate = torchaudio.load(path)
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform.squeeze(0)


def trim_or_pad(waveform: torch.Tensor, target_num_samples: int, random_crop: bool) -> torch.Tensor:
    if waveform.shape[-1] == target_num_samples:
        return waveform

    if waveform.shape[-1] > target_num_samples:
        max_offset = waveform.shape[-1] - target_num_samples
        offset = torch.randint(0, max_offset + 1, size=(1,)).item() if random_crop else 0
        return waveform[offset : offset + target_num_samples]

    pad_amount = target_num_samples - waveform.shape[-1]
    return F.pad(waveform, (0, pad_amount))


def load_embedding(path: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("embedding", "embeddings", "x"):
            if key in payload:
                payload = payload[key]
                break
    tensor = torch.as_tensor(payload, dtype=torch.float32)
    if tensor.ndim != 1:
        tensor = tensor.reshape(-1)
    return tensor


def resolve_data_path(root: Optional[str], raw_path: str) -> str:
    candidate = Path(raw_path)
    if candidate.is_absolute() or root is None:
        return str(candidate)
    return str((Path(root) / raw_path).resolve())
