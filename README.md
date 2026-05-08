# MIR Class Spring 2026, Final Project, Instrument Classification

This repo contains the training pipeline we use for instrument classification experiments on Slakh and OpenMIC, including:

- Slakh and OpenMIC dataset loaders
- multilabel training
- A vanilla Mel-CNN encoder trained from scratch, CLAP, and AudioMAE encoder options
- synthetic-to-real evaluation
- checkpoint evaluation scripts

## Installation

In order to use the repo you should create the following environment

```bash
conda create -n mir-inst python=3.10 pip
conda activate mir-inst
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Options

The pipeline currently supports:

- `type: "slakh"`
  Slakh waveform training/evaluation, including the cached MIDI index and on-the-fly remix path and augmentations for multilabel training.
- `type: "openmic"`
  OpenMIC waveform evaluation/training with configurable target label subsets and a configurable `relevance_threshold`.

## Typical Setup

### 1. Build the Slakh MIDI index once

we build this index so that during training we won't have to open the midi files one by one and check whether the instrument is active. It allows us for faster loading of dataset and training. 

```bash
python scripts/build_slakh_midi_index.py \
  --root /path/to/slakh2100_flac_redux \
  --output /path/to/slakh_midi_index.pt
```

### 2. Build OpenMIC splits 

Our Open Mic Dataset expects these. 
```bash
python scripts/build_openmic_splits.py \
  --root /path/to/openmic-2018 \
  --labels-json configs/slakh_to_openmic_conv_6_classes.json \
  --output-dir /path/to/openmic_partitions \
  --seed 42
```

### 3. Train

```bash
python -m training_base.cli --config configs/slakh_multilabel_conv_6_classes.json
```

### 4. Evaluate a saved checkpoint

```bash
python scripts/evaluate_checkpoint.py \
  --config configs/slakh_to_openmic_conv_6_classes.json \
  --ckpt-path /path/to/checkpoint.ckpt
```