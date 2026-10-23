from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import torchaudio
except ImportError:  # pragma: no cover - depends on local environment
    torchaudio = None

try:
    from pedalboard.io import AudioFile
except ImportError:  # pragma: no cover - depends on local environment
    AudioFile = None

# Extra audio decoded on each side of an excerpt, so the resampling filter's edge effects
# fall outside the samples we keep.
EXCERPT_MARGIN_SEC = 0.1


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


def load_waveform_excerpt(path: str, target_sample_rate: int, offset: int, num_samples: int) -> torch.Tensor:
    # Same samples as load_waveform(path, target_sample_rate)[offset : offset + num_samples],
    # but only the audio around the excerpt is decoded and resampled. load_waveform reads
    # the whole file, which for an hour-long SynthSOD stem costs gigabytes per training clip.
    if AudioFile is None:
        return load_waveform(path, target_sample_rate)[offset : offset + num_samples]
    require_torchaudio()

    with AudioFile(str(path)) as handle:
        source_rate = int(handle.samplerate)
        total_frames = int(handle.frames)
        # Resampled sample j of a read that starts at source frame n0 sits at time
        # n0 / source_rate + j / target_rate. Starting the read on a whole number of
        # `source_step` frames keeps that grid identical to the full-file resample.
        divisor = math.gcd(source_rate, target_sample_rate)
        source_step = source_rate // divisor
        target_step = target_sample_rate // divisor
        margin_steps = math.ceil(EXCERPT_MARGIN_SEC * target_sample_rate / target_step)
        first_step = max(0, offset // target_step - margin_steps)
        last_step = -(-(offset + num_samples) // target_step) + margin_steps
        start_frame = min(first_step * source_step, total_frames)
        end_frame = min(last_step * source_step, total_frames)
        if end_frame <= start_frame:
            # The excerpt starts past the end of the file; the caller pads it with silence.
            return torch.zeros(0, dtype=torch.float32)
        handle.seek(start_frame)
        data = handle.read(end_frame - start_frame)

    waveform = torch.from_numpy(data).float()
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, target_sample_rate)
    start = offset - first_step * target_step
    return waveform.squeeze(0)[start : start + num_samples]


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
