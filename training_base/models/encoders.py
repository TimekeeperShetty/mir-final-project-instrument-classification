from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

import torch
from torch import nn


def _freeze_module(module: nn.Module, freeze: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = not freeze


def _load_factory(path: str):
    module_name, attr_name = path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


class BaseEncoder(nn.Module):
    def __init__(self, output_dim: int, input_kind: str) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_kind = input_kind

    def forward(self, inputs: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class IdentityEncoder(BaseEncoder):
    def __init__(self, output_dim: int) -> None:
        super().__init__(output_dim=output_dim, input_kind="embedding")

    def forward(self, inputs: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        return inputs.float()


class ConvWaveformEncoder(BaseEncoder):
    """
    Tiny example waveform encoder.
    This is only here so the scaffold can run without a full pretrained model.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__(output_dim=output_dim, input_kind="waveform")
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
        )
        self.proj = nn.Linear(128, output_dim)

    def forward(self, inputs: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = inputs.float().unsqueeze(1)
        feats = self.net(x)
        pooled = feats.mean(dim=-1)
        return self.proj(pooled)



class ExternalEncoder(BaseEncoder):
    """
    Adapter for encoders created elsewhere, e.g. CLAP or AudioMAE.

    Expected factory contract:
    - config supplies encoder.factory = "your_module:build_encoder"
    - the factory returns an nn.Module
    - the module forward accepts either:
      1. forward(inputs)
      2. forward(inputs, lengths=lengths)
    """

    def __init__(
        self,
        output_dim: int,
        input_kind: str,
        factory: str,
        factory_kwargs: Optional[Dict[str, Any]] = None,
        freeze: bool = True,
    ) -> None:
        super().__init__(output_dim=output_dim, input_kind=input_kind)
        builder = _load_factory(factory)
        self.module = builder(**(factory_kwargs or {}))
        if not isinstance(self.module, nn.Module):
            raise TypeError(f"Encoder factory {factory} did not return an nn.Module")
        _freeze_module(self.module, freeze)

    def forward(self, inputs: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            outputs = self.module(inputs, lengths=lengths)
        except TypeError:
            outputs = self.module(inputs)
        if isinstance(outputs, dict):
            for key in ("embedding", "embeddings", "x"):
                if key in outputs:
                    outputs = outputs[key]
                    break
        return outputs.float()


def build_encoder(cfg: Any) -> BaseEncoder:
    #here depending on our configuration file we are building the corresponding encoder
    if cfg.type == "identity":
        #this branch is for the case where we are loading saved embeddings
        return IdentityEncoder(output_dim=cfg.output_dim)
    if cfg.type == "conv_waveform":
        #this is an example encoder
        return ConvWaveformEncoder(output_dim=cfg.output_dim)
    if cfg.type == "external":
        if not cfg.factory:
            raise ValueError("External encoder requires encoder.factory")
        return ExternalEncoder(
            output_dim=cfg.output_dim,
            input_kind=cfg.input_kind,
            factory=cfg.factory,
            factory_kwargs=cfg.factory_kwargs,
            freeze=cfg.freeze,
        )
    raise ValueError(f"Unsupported encoder type: {cfg.type}")
