from __future__ import annotations

from typing import Any, Dict

import torch


class IdentityAugmenter:
    """Default no-op augmenter."""

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform.float()


class ExampleWaveformAugmenter:
    """
    Very small example only.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.normalize = bool(cfg.get("normalize", False))
        self.random_gain_db = float(cfg.get("random_gain_db", 0.0))
        self.gaussian_noise_std = float(cfg.get("gaussian_noise_std", 0.0))

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        x = waveform.float()
        if self.random_gain_db > 0:
            x = self._random_gain(x)
        if self.gaussian_noise_std > 0:
            x = x + torch.randn_like(x) * self.gaussian_noise_std
        if self.normalize:
            peak = x.abs().max().clamp_min(1e-6)
            x = x / peak
        return x

    def _random_gain(self, waveform: torch.Tensor) -> torch.Tensor:
        gain_db = torch.empty(1).uniform_(-self.random_gain_db, self.random_gain_db).item()
        gain = 10 ** (gain_db / 20.0)
        return waveform * gain


def build_augmenter(cfg: Dict[str, Any] | None):
    if not cfg:
        return IdentityAugmenter()
    aug_type = cfg.get("type", "identity")
    if aug_type in {"identity", "none"}:
        return IdentityAugmenter()
    if aug_type == "example_waveform":
        return ExampleWaveformAugmenter(cfg)
    raise ValueError(f"Unsupported augmentation type: {aug_type}")
