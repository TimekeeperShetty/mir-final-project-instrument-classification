from __future__ import annotations

from typing import Any, Dict

import torch

try:
    import numpy as np
    from pedalboard import Chorus, Gain, HighpassFilter, LowpassFilter, Pedalboard, Reverb
except ImportError:  # pragma: no cover - depends on local environment
    np = None
    Chorus = None
    Gain = None
    HighpassFilter = None
    LowpassFilter = None
    Pedalboard = None
    Reverb = None


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


class ReverbNoiseAugmenter:
    """
    Stem-level room simulation augmenter.

    This is the clean version of the older branch idea:
    - apply a random reverb
    - apply a random gain shift
    - add low-level Gaussian noise

    The important part is *where* this runs in the current pipeline:
    - it is applied to each selected stem excerpt independently
    - then the augmented stems are summed into the synthetic submix
    - it is not applied to stems that were dropped before remixing
    """

    def __init__(self, cfg: Dict[str, Any]):
        if (
            Pedalboard is None
            or Reverb is None
            or Gain is None
            or HighpassFilter is None
            or LowpassFilter is None
            or Chorus is None
            or np is None
        ):
            raise ImportError(
                "The 'reverb_noise' augmenter requires pedalboard and numpy. "
                "Install pedalboard to use this augmentation type."
            )

        self.sample_rate = int(cfg.get("sample_rate", 22050))
        self.noise_std = float(cfg.get("noise_std", 0.001))
        self.noise_prob = float(cfg.get("noise_prob", 1.0))
        self.gain_db_min = float(cfg.get("gain_db_min", -3.0))
        self.gain_db_max = float(cfg.get("gain_db_max", 3.0))
        self.room_size_min = float(cfg.get("room_size_min", 0.1))
        self.room_size_max = float(cfg.get("room_size_max", 0.6))
        self.damping_min = float(cfg.get("damping_min", 0.3))
        self.damping_max = float(cfg.get("damping_max", 0.7))
        self.wet_level_min = float(cfg.get("wet_level_min", 0.1))
        self.wet_level_max = float(cfg.get("wet_level_max", 0.4))
        self.dry_level_min = float(cfg.get("dry_level_min", 0.6))
        self.dry_level_max = float(cfg.get("dry_level_max", 0.9))
        self.width_min = float(cfg.get("width_min", 0.5))
        self.width_max = float(cfg.get("width_max", 1.0))

        # Optional extra color effects. They are all off by default, but if we enable
        # them in a config they run per stem before the final submix is formed.
        self.highpass_prob = float(cfg.get("highpass_prob", 0.0))
        self.highpass_hz_min = float(cfg.get("highpass_hz_min", 40.0))
        self.highpass_hz_max = float(cfg.get("highpass_hz_max", 180.0))
        self.lowpass_prob = float(cfg.get("lowpass_prob", 0.0))
        self.lowpass_hz_min = float(cfg.get("lowpass_hz_min", 2500.0))
        self.lowpass_hz_max = float(cfg.get("lowpass_hz_max", 9000.0))
        self.chorus_prob = float(cfg.get("chorus_prob", 0.0))
        self.chorus_rate_hz_min = float(cfg.get("chorus_rate_hz_min", 0.2))
        self.chorus_rate_hz_max = float(cfg.get("chorus_rate_hz_max", 1.2))
        self.chorus_depth_min = float(cfg.get("chorus_depth_min", 0.1))
        self.chorus_depth_max = float(cfg.get("chorus_depth_max", 0.4))
        self.chorus_centre_delay_ms_min = float(cfg.get("chorus_centre_delay_ms_min", 4.0))
        self.chorus_centre_delay_ms_max = float(cfg.get("chorus_centre_delay_ms_max", 10.0))
        self.chorus_feedback_min = float(cfg.get("chorus_feedback_min", 0.0))
        self.chorus_feedback_max = float(cfg.get("chorus_feedback_max", 0.15))
        self.chorus_mix_min = float(cfg.get("chorus_mix_min", 0.05))
        self.chorus_mix_max = float(cfg.get("chorus_mix_max", 0.25))

    def _build_effect_chain(self) -> Pedalboard:
        effects = []

        if self.highpass_prob > 0 and np.random.uniform(0.0, 1.0) <= self.highpass_prob:
            effects.append(
                HighpassFilter(cutoff_frequency_hz=np.random.uniform(self.highpass_hz_min, self.highpass_hz_max))
            )

        if self.lowpass_prob > 0 and np.random.uniform(0.0, 1.0) <= self.lowpass_prob:
            effects.append(
                LowpassFilter(cutoff_frequency_hz=np.random.uniform(self.lowpass_hz_min, self.lowpass_hz_max))
            )

        effects.append(
            Reverb(
                room_size=np.random.uniform(self.room_size_min, self.room_size_max),
                damping=np.random.uniform(self.damping_min, self.damping_max),
                wet_level=np.random.uniform(self.wet_level_min, self.wet_level_max),
                dry_level=np.random.uniform(self.dry_level_min, self.dry_level_max),
                width=np.random.uniform(self.width_min, self.width_max),
            )
        )

        if self.chorus_prob > 0 and np.random.uniform(0.0, 1.0) <= self.chorus_prob:
            effects.append(
                Chorus(
                    rate_hz=np.random.uniform(self.chorus_rate_hz_min, self.chorus_rate_hz_max),
                    depth=np.random.uniform(self.chorus_depth_min, self.chorus_depth_max),
                    centre_delay_ms=np.random.uniform(
                        self.chorus_centre_delay_ms_min, self.chorus_centre_delay_ms_max
                    ),
                    feedback=np.random.uniform(self.chorus_feedback_min, self.chorus_feedback_max),
                    mix=np.random.uniform(self.chorus_mix_min, self.chorus_mix_max),
                )
            )

        effects.append(Gain(gain_db=np.random.uniform(self.gain_db_min, self.gain_db_max)))
        return Pedalboard(effects)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        x = waveform.detach().cpu().float().numpy().astype(np.float32)
        squeeze = False
        if x.ndim == 1:
            x = x[np.newaxis, :]
            squeeze = True

        board = self._build_effect_chain()
        effected = board(x, self.sample_rate)
        if self.noise_std > 0 and np.random.uniform(0.0, 1.0) <= self.noise_prob:
            noise = np.random.normal(0.0, self.noise_std, effected.shape).astype(np.float32)
            effected = effected + noise
        effected = np.clip(effected, -1.0, 1.0)

        if squeeze:
            effected = effected.squeeze(0)
        return torch.from_numpy(effected).float()


def build_augmenter(cfg: Dict[str, Any] | None):
    if not cfg:
        return IdentityAugmenter()
    aug_type = cfg.get("type", "identity")
    if aug_type in {"identity", "none"}:
        return IdentityAugmenter()
    if aug_type == "example_waveform":
        return ExampleWaveformAugmenter(cfg)
    if aug_type == "reverb_noise_highpass_lowpass_chorus":
        return ReverbNoiseAugmenter(cfg)
    raise ValueError(f"Unsupported augmentation type: {aug_type}")