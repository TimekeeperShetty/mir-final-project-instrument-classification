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


import torchaudio

class MelSpectrogramEncoder(BaseEncoder):
    """
    A strong baseline 2D CNN operating on log-Mel Spectrograms.
    Similar in architecture to VGGish or PANNs CNN10.
    """
    def __init__(
        self, 
        output_dim: int, 
        sample_rate: int = 22050, 
        n_fft: int = 1024, 
        hop_length: int = 256, 
        n_mels: int = 64
    ) -> None:
        super().__init__(output_dim=output_dim, input_kind="waveform")
        
        # 1. On-the-fly Spectrogram Extraction
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=50.0,
            f_max=sample_rate / 2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        
        # 2. 2D CNN Backbone
        # Input shape: [Batch, 1, n_mels, time_frames]
        self.features = nn.Sequential(
            self._conv_block(1, 32, pool=True),      # [B, 32, n_mels/2, time/2]
            self._conv_block(32, 64, pool=True),     # [B, 64, n_mels/4, time/4]
            self._conv_block(64, 128, pool=True),    # [B, 128, n_mels/8, time/8]
            self._conv_block(128, 256, pool=True),   # [B, 256, n_mels/16, time/16]
            self._conv_block(256, 512, pool=False),  # [B, 512, n_mels/16, time/16]
        )
        
        # 3. Projection Head
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )

    def _conv_block(self, in_channels: int, out_channels: int, pool: bool) -> nn.Module:
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # inputs shape: [Batch, Samples]
        
        # 1. Create Spectrogram
        x = self.mel_transform(inputs)          # [Batch, n_mels, time]
        x = self.amplitude_to_db(x)             # Log-mel scale [Batch, n_mels, time]
        
        # 2. Per-Instance Z-Score Normalization
        # We calculate the mean and std for EACH item in the batch independently
        # Keep dims=True so we can broadcast the subtraction/division back to the original shape
        mean = x.mean(dim=[1, 2], keepdim=True)
        std = x.std(dim=[1, 2], keepdim=True)
        
        # Add a tiny epsilon (1e-5) to prevent division by zero in pure silence
        x = (x - mean) / (std + 1e-5)
        
        # Add channel dimension for 2D Conv
        x = x.unsqueeze(1)                      # [Batch, 1, n_mels, time]
        
        # 3. Extract features
        x = self.features(x)                    # [Batch, 512, freq, time]
        
        # 4. Global Pooling (mean across time and frequency)
        x = x.mean(dim=[2, 3])                  # [Batch, 512]
        
        # 5. Project
        return self.proj(x)                     # [Batch, output_dim]

class AudioMAEEncoder(BaseEncoder):
    """
    AudioMAE (Masked Autoencoders that Listen) encoder.
    Loads the HuggingFace `facebook/audiomae-base` checkpoint (ViT-Base,
    pre-trained on AudioSet with masked autoencoding) and projects its
    patch-token mean-pool to `output_dim`.
 
    Input  : raw waveform tensor [B, T] at `sample_rate` Hz (default 16 kHz).
    Output : embedding tensor    [B, output_dim].
 
    The backbone is frozen by default; only the projection head is trained.
 
    Notes
    -----
    - AudioMAE was pre-trained with 128 mel bins, 16 kHz, 25 ms / 10 ms
      windows, so keep `sample_rate=16000` unless you retrain the backbone.
    - Requires `transformers >= 4.35.0`:  pip install transformers
    - The model will be downloaded automatically from HuggingFace Hub on first
      use (~340 MB for ViT-Base).
    """
     # AudioMAE pre-training hyper-parameters (do not change unless you are
     # using a different checkpoint)
    _N_FFT      = 400
    _HOP_LENGTH = 160
    _N_MELS     = 128
    _F_MIN      = 0.0
    _F_MAX      = 8000.0
    _HIDDEN     = 768

    def __init__(self, output_dim, model_name="facebook/audiomae-base", sample_rate=16000, freeze=True, pooling="mean"):
        super().__init__(output_dim=output_dim, input_kind="waveform")
        from transformers import AudioMAEModel
        self.backbone = AudioMAEModel.from_pretrained(model_name)
        _freeze_module(self.backbone, freeze)
        self.pooling = pooling
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=self._N_FFT, hop_length=self._HOP_LENGTH,
            n_mels=self._N_MELS, f_min=self._F_MIN, f_max=self._F_MAX,
            window_fn=torch.hann_window, normalized=False,
        )
        self.proj = nn.Sequential(nn.LayerNorm(self._HIDDEN), nn.Linear(self._HIDDEN, output_dim))

    def forward(self, inputs, lengths=None):
        mel = self.mel_transform(inputs.float())
        log_mel = (mel + 1e-6).log()
        mean = log_mel.mean(dim=[1,2], keepdim=True)
        std  = log_mel.std(dim=[1,2],  keepdim=True)
        log_mel = (log_mel - mean) / (std + 1e-5)
        pixel_values = log_mel.permute(0,2,1).unsqueeze(1)
        hidden = self.backbone(pixel_values=pixel_values).last_hidden_state
        pooled = hidden[:,0] if self.pooling == "cls" else hidden[:,1:].mean(dim=1)
        
        return self.proj(pooled)
        
 

class ClapEncoder(BaseEncoder):
    """
    LAION-CLAP audio encoder (frozen) used as a feature extractor.

    Loads a published `laion_clap.CLAP_Module` checkpoint and returns the
    512-dim audio embedding produced by the audio branch only (no text branch
    is used). The backbone is frozen by default; only a trainable classifier
    head is meant to sit on top of the embedding.

    Input  : raw waveform tensor [B, T] at `sample_rate` Hz (the project's
             pipeline sample rate; resampled internally to 48 kHz, which is
             what CLAP was trained on).
    Output : embedding tensor    [B, output_dim]. When `output_dim == 512`
             the native CLAP embedding is returned unchanged; otherwise a
             single Linear projection maps 512 -> output_dim.

    Notes
    -----
    - Requires `pip install laion-clap huggingface_hub`.
    - Checkpoint resolution order:
        1. `pretrained_path` (explicit local file) wins if set.
        2. Otherwise `hf_repo` + `hf_filename` are used to fetch (and cache)
           the checkpoint from HuggingFace via `huggingface_hub.hf_hub_download`.
        3. If neither is set, falls back to `CLAP_Module.load_ckpt()` with no
           arguments, which fetches LAION's default 630k-audioset checkpoint.
    - For instrument classification, the music-pretrained HTSAT-base
      checkpoint (`music_audioset_epoch_15_esc_90.14.pt` in `lukewys/laion_clap`)
      is the recommended default and is auto-downloaded on first use.
    """

    _CLAP_SAMPLE_RATE = 48000
    _CLAP_EMBED_DIM = 512

    def __init__(
        self,
        output_dim: int,
        sample_rate: int = 22050,
        amodel: str = "HTSAT-base",
        enable_fusion: bool = False,
        pretrained_path: Optional[str] = None,
        hf_repo: Optional[str] = "lukewys/laion_clap",
        hf_filename: Optional[str] = "music_audioset_epoch_15_esc_90.14.pt",
        freeze: bool = True,
    ) -> None:
        super().__init__(output_dim=output_dim, input_kind="waveform")
        import laion_clap

        self.input_sample_rate = sample_rate
        self.backbone = laion_clap.CLAP_Module(
            enable_fusion=enable_fusion,
            amodel=amodel,
        )
        ckpt_path = self._resolve_ckpt(pretrained_path, hf_repo, hf_filename)
        if ckpt_path is not None:
            self.backbone.load_ckpt(ckpt=ckpt_path)
        else:
            self.backbone.load_ckpt()
        _freeze_module(self.backbone, freeze)

        if sample_rate != self._CLAP_SAMPLE_RATE:
            self.resample = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self._CLAP_SAMPLE_RATE,
            )
        else:
            self.resample = nn.Identity()

        if output_dim == self._CLAP_EMBED_DIM:
            self.proj: nn.Module = nn.Identity()
        else:
            self.proj = nn.Linear(self._CLAP_EMBED_DIM, output_dim)

    def forward(self, inputs: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.resample(inputs.float())
        embedding = self.backbone.get_audio_embedding_from_data(x=x, use_tensor=True)
        return self.proj(embedding)

    @staticmethod
    def _resolve_ckpt(
        pretrained_path: Optional[str],
        hf_repo: Optional[str],
        hf_filename: Optional[str],
    ) -> Optional[str]:
        if pretrained_path:
            return pretrained_path
        if hf_repo and hf_filename:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo_id=hf_repo, filename=hf_filename)
        return None


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
        return IdentityEncoder(output_dim=cfg.output_dim)
    if cfg.type == "conv_waveform":
        return ConvWaveformEncoder(output_dim=cfg.output_dim)
    if cfg.type == "mel_cnn": 
        return MelSpectrogramEncoder(
            output_dim=cfg.output_dim,
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            n_mels=cfg.n_mels
        )
    if cfg.type == "audiomae":
        return AudioMAEEncoder(
            output_dim=cfg.output_dim,
            model_name=getattr(cfg, "model_name", "facebook/audiomae-base"),
            sample_rate=getattr(cfg, "sample_rate", 16000),
            freeze=getattr(cfg, "freeze", True),
            pooling=getattr(cfg, "pooling", "mean"),
        )
    if cfg.type == "clap":
        return ClapEncoder(
            output_dim=cfg.output_dim,
            sample_rate=getattr(cfg, "sample_rate", 22050),
            amodel=getattr(cfg, "amodel", "HTSAT-base"),
            enable_fusion=getattr(cfg, "enable_fusion", False),
            pretrained_path=getattr(cfg, "pretrained_path", None),
            hf_repo=getattr(cfg, "hf_repo", "lukewys/laion_clap"),
            hf_filename=getattr(cfg, "hf_filename", "music_audioset_epoch_15_esc_90.14.pt"),
            freeze=getattr(cfg, "freeze", True),
        )
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
