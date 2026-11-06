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

### Build the aligned-only SynthSOD index

The aligned scores cover a subset of SynthSOD. Use their existing official
partitions: the supplied release has 351 train, 42 evaluation, and 91 test pieces.
No random re-split is needed. `evaluation` is stored as `validation` for the loader.
The builder cross-checks membership against the original SynthSOD split metadata
and excludes pieces without aligned scores; it never falls back to RMS.

On the HPC, with the v2 aligned-score release extracted:

```bash
python scripts/build_synthsod_aligned_index.py \
  --root /scratch/axs10538/datasets/synthsod/SynthSOD \
  --aligned-root /scratch/mg8253/mir_instrument_classification/SynthSOD_aligned_scores \
  --output /scratch/mg8253/mir_instrument_classification/synthsod_aligned_index.pt \
  --mix-root /scratch/mg8253/mir_instrument_classification/synthsod_aligned_mixes \
  --workers 4
```

The output index, its `.report.json` companion and mix directory must be new paths.
The report records split counts, excluded unannotated pieces, and target coverage.
Missing scored audio or scores cause an error instead of silently shrinking the
dataset. `--limit N` builds a trial with up to N pieces per split; use different
output/mix paths for trials. Full mixes are generated once, which requires audio
decoding and disk space even though subsequent remix training reads the source
stem excerpts directly.

Intervals are unions of aligned note start/end times, with silence rows ignored
and no RMS thresholds or added release tails. Since program 40 merges violin
sections, both section files are read at the same offset and summed as one logical
violin stem before augmentation. Sections cannot be independently dropped. Other
instruments retain their individual activity and normal remix behavior. Ambiguous
untuned percussion is omitted from the stems and the generated full mixes.

Use `configs/slakh_synthsod_aligned_rendered_to_openmic_clap_20_classes.json` to
combine the Slakh index with the new aligned-only SynthSOD index and rendered
Slakh replacements. Its training split remains `train`; the 42/91 SynthSOD
holdout pieces remain reserved, and the experiment still validates/tests on
OpenMIC. Existing configs and the RMS index remain available for comparison.
The replacement probability remains the provisional 0.3 setting; measure emitted
training-label frequencies before tuning it or the remix balancing parameters.

### Compare SynthSOD RMS activity with aligned scores

Use the aligned-score v2 release, extracted with `SynthSOD_score_data/` and the
`SynthSOD_metadata_aligned_*.json` files directly under the aligned root. On the HPC:

```bash
python scripts/compare_synthsod_activity.py \
  --rms-index /scratch/axs10538/datasets/synthsod/synthsod_index.pt \
  --aligned-root /scratch/mg8253/mir_instrument_classification/SynthSOD_aligned_scores \
  --synthsod-root /scratch/axs10538/datasets/synthsod/SynthSOD \
  --config configs/slakh_synthsod_rendered_to_openmic_clap_20_classes.json \
  --split train --crops-per-piece 200
```

The command reads cached RMS intervals and TXT note timings without decoding audio
or changing the index. It preserves official splits, ignores silence rows
(`MIDI_Instrument=-1`), and unions overlapping notes. Reports go to the Git-ignored
`outputs/synthsod_activity_comparison/`: `summary.json` contains per-instrument and
per-class time/crop metrics and coverage gaps, `per_piece.csv` identifies outliers,
and `crop_disagreements.jsonl` lists disagreeing crop locations. Output directories
must be new; pass `--output-dir outputs/another_comparison` for another run.
Use `--limit 20` for a quick subset or `--split all` for separate split reports.

`score_only` means the score marks activity that RMS misses; `rms_only` means RMS
marks activity outside the score. Precision/recall use scores as a reference, not
as exact acoustic ground truth. Positive-crop disagreement excludes agreement on
silence, so long rests cannot hide errors. Crops use the configured sample rate,
duration and activity ratio, before stochastic remix dropping or replacements.
`class_union` compares pooled class activity; `remix_eligibility` checks whether
an individual RMS stem passes the loader's activity rule. Violin sections share
program 40 in the TXT scores, so their score activity is combined: violin remix
eligibility is explicitly marked as ambiguous at section level. Do not assign
the pooled violin timings independently to each section stem in a future index.

`Close Mic` and `Tree` are microphone perspectives of the same performances:
Close Mic is the driest available perspective, while Tree includes the Decca Tree
stereo room/reverb perspective. The current index builder defaults to Close Mic
for RMS activity and Tree for training audio. Note ends and acoustic decay can
therefore differ even with correct temporal alignment.

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
