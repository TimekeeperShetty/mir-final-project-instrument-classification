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
  OpenMIC annotations are sparse: labels that were not queried for a clip are masked out of loss and metrics instead of being treated as negatives.

## Typical Setup

### Audition rendered Slakh instruments

On the HPC, with `ffmpeg` (including `libmp3lame`) on PATH, run from the repository root:

```bash
python scripts/export_rendered_stem_pairs.py --count 20 --seed 42
```

This uses `/scratch/jnl9728/synth_banjo`, `synth_mandolin`, and `synth_ukulele`.
It selects 20 distinct Slakh tracks and one random available render per track,
matching the source MIDI in each provenance file. The Git-ignored
`rendered_stem_pairs/` directory receives full-length original and rendered MP3s,
a simultaneous stereo comparison (original left, rendered right), and a manifest
with the source paths and note-remapping statistics. Leading silence is preserved;
no onset alignment or silence trimming is performed. Different ending lengths are
padded in the stereo comparison. Use headphones or solo the channels to compare.
The command refuses to overwrite an existing output directory.

To audition only banjo, add `--instrument banjo`. To change source locations, use
`--slakh-root PATH` and repeat `--render-root banjo=PATH` (and the other desired
instruments). For another listening set, use `--output-dir rendered_stem_pairs/set2`.

### Train with rendered stem replacements

After checking timing, the example config
`configs/slakh_synthsod_rendered_to_openmic_clap_20_classes.json` enables replacements
with a 0.3 probability per eligible source stem per crop. To configure another
multilabel experiment, add this under `data.train.params` with `stem_remix: true`:

```json
"rendered_stem_replacements": {
  "probability": 0.3,
  "roots": {
    "banjo": "/scratch/jnl9728/synth_banjo",
    "mandolin": "/scratch/jnl9728/synth_mandolin",
    "ukulele": "/scratch/jnl9728/synth_ukulele"
  }
}
```

The loader chooses uniformly among available replacement instruments after the
probability check. Missing variants leave the original stem unchanged. Matching
uses the provenance split, track, and MIDI stem, so only renders belonging to the
selected training tracks are eligible. Replacement occurs before class selection
and changes both the audio path and class; other retained guitar stems still
produce a guitar label. Validation/test and SynthSOD stems are unaffected.
Replacement targets must be in `data.label_names`; no MIDI index rebuild is needed.
Omit the setting or use probability 0 to disable it.

The replacements inherit source MIDI activity intervals and t=0. Pitch folding
does not provide new timings; dropped or voice-capped notes can make those
intervals overestimate rendered activity. Listening checks timing, not exact
note-for-note activity. The loader does not infer corrected intervals from audio.

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
