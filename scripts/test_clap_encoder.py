"""
Smoke test for the CLAP audio encoder integration.

Builds a `ClapEncoder` (downloading the music-pretrained checkpoint via
HuggingFace on first run, then caching it), feeds a dummy waveform batch,
and verifies the output shape. Also runs the full encoder + classifier head
that the training pipeline would assemble, to catch wiring issues early.

Run from the repo root:
    python -m scripts.test_clap_encoder

Requires: torch, torchaudio, laion-clap, huggingface_hub.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from training_base.models.classifiers import build_classifier
from training_base.models.encoders import build_encoder


SAMPLE_RATE = 22050
CLIP_SECONDS = 10
BATCH_SIZE = 2
NUM_CLASSES = 6


def main() -> int:
    torch.manual_seed(0)

    encoder_cfg = SimpleNamespace(
        type="clap",
        input_kind="waveform",
        output_dim=512,
        freeze=True,
        sample_rate=SAMPLE_RATE,
        amodel="HTSAT-base",
        enable_fusion=False,
        pretrained_path=None,
        hf_repo="lukewys/laion_clap",
        hf_filename="music_audioset_epoch_15_esc_90.14.pt",
    )
    classifier_cfg = SimpleNamespace(type="mlp", hidden_dim=256, dropout=0.1)

    print("Building CLAP encoder (first run will download ~1 GB checkpoint)...")
    encoder = build_encoder(encoder_cfg).eval()
    print(f"  encoder.output_dim = {encoder.output_dim}")
    print(f"  encoder.input_kind = {encoder.input_kind}")

    classifier = build_classifier(classifier_cfg, encoder.output_dim, NUM_CLASSES)
    print(f"  classifier         = {type(classifier).__name__}")

    waveform = torch.randn(BATCH_SIZE, SAMPLE_RATE * CLIP_SECONDS)
    print(f"\nDummy input shape  : {tuple(waveform.shape)}  (B, T at {SAMPLE_RATE} Hz)")

    with torch.no_grad():
        embedding = encoder(waveform)
        logits = classifier(embedding)

    print(f"Embedding shape    : {tuple(embedding.shape)}  (expect (B, 512))")
    print(f"Logits shape       : {tuple(logits.shape)}     (expect (B, {NUM_CLASSES}))")

    assert embedding.shape == (BATCH_SIZE, 512), embedding.shape
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES), logits.shape

    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    print(f"\nEncoder params     : {total:,} total / {trainable:,} trainable")
    print(f"Classifier params  : {sum(p.numel() for p in classifier.parameters()):,} trainable")

    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
