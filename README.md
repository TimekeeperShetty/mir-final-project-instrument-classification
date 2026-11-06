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

### Audit the combined training epoch and inspect 50 examples

```bash
python scripts/audit_combined_training_epoch.py --preview-count 50
```

The default config is
`configs/slakh_synthsod_aligned_rendered_to_openmic_clap_20_classes.json`; despite
its historical filename it has 19 target classes, excluding voice. Training uses
Slakh train, aligned SynthSOD train, and the rendered banjo/mandolin/ukulele stems
from training MIDIs. The audit iterates one full training-loader epoch without
training a model, respecting batch size and `drop_last`, and counts the actual
labels after crop activity checks, replacements, and remix class selection.

While it runs, it saves examples under the Git-ignored
`rendered_stem_pairs/combined_epoch_audit/`. Each `sample_*` folder contains the
final `mix.mp3`, `targets.mp3`, `context.mp3` when applicable, individual selected
stem MP3s, and `labels.txt`/`labels.json` with crop times, source paths, render
provenance and target/context roles. Stem audio is captured from the counted
loader call after augmentation, rather than decoded/augmented a second time.
Stems are scaled by the mix normalization gain; all exported files share an
additional documented attenuation to avoid clipping. With the configured identity
mix augmentation, targets plus context reconstruct the mix before MP3 encoding.

The saved examples are sampled across Slakh, render-capable Slakh and SynthSOD
tracks; this stratifies the previews without changing the epoch distribution.
`distribution.json` updates every ten batches and records per-label clip counts
and percentages, selected logical stem counts, original/rendered/SynthSOD source
contributions, context counts, empty targets and fallback usage. The final
`distribution.csv` provides a compact table. Actual retained-render coverage is
reported, with a warning if none survived in the epoch or saved examples.
Use a new `--output-dir rendered_stem_pairs/combined_epoch_seed43 --seed 43`
for another run. `--num-workers 0` is available for debugging. This exports up to
50 examples from one audit epoch, not 50 new files per subsequent model epoch.

The same config validates on the official validation subsets of Slakh and aligned
SynthSOD, with matching rendered validation MIDIs. Validation uses fixed seeded
crops/replacements, identity augmentation, all active target classes and all
active context stems; grouped violin handling and the per-class stem cap remain.
The CLI selects the checkpoint with the lowest synthetic validation loss and
then tests it on the configured OpenMIC test subset:

```bash
python -m training_base.cli --config configs/slakh_synthsod_aligned_rendered_to_openmic_clap_20_classes.json
```

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
  --output /scratch/mg8253/mir_instrument_classification/synthsod_aligned_stems_index.pt \
  --workers 4
```

The output index and its `.report.json` companion must be new paths.
The report records split counts, excluded unannotated pieces, and target coverage.
Missing score files cause an error. Pieces with missing audio, score overruns, or
instrument-set disagreements between score and aligned metadata are excluded with
an explicit console warning and a reason under `invalid_pieces_excluded` in the
report. Use `--strict` to stop instead. Differences affecting only dropped untuned
percussion do not exclude a piece. In the supplied local annotations,
`salve_regina_orch` lists strings in its metadata but trumpet/trombone in its TXT
score, so it is excluded rather than guessing a remapping. The expected training
count is therefore 350 if no other audio problems are found; holdouts remain 42/91.
`--limit N` builds a trial with up to N pieces per split; use different
output paths for trials. The builder reads score files and audio headers only;
it does not decode full stems, generate mixes, or require a `--mix-root`.
The index references the original stem files and requires `stem_remix: true`.
At each crop, the dataloader selects active classes, drops instruments according
to the configured remix policy, reads the selected stem excerpts, and sums them
on the fly. Labels are built from the retained target instruments. For synthetic
validation/test with this index, also set `deterministic_remix: true`.

Intervals are unions of aligned note start/end times, with silence rows ignored
and no RMS thresholds or added release tails. Since program 40 merges violin
sections, both section files are read at the same offset and summed as one logical
violin stem before augmentation. Sections cannot be independently dropped. Other
instruments retain their individual activity and normal remix behavior. Ambiguous
untuned percussion is omitted from the stems.

Use `configs/slakh_synthsod_aligned_rendered_to_openmic_clap_20_classes.json` to
combine the Slakh index with the new aligned-only SynthSOD index and rendered
Slakh replacements. Its training split remains `train`; synthetic validation uses
the `validation` splits, including the 42 aligned SynthSOD validation candidates.
Final testing uses OpenMIC. Existing alternative configs and the RMS index remain
available for comparison.
The replacement probability remains the provisional 0.3 setting; measure emitted
training-label frequencies before tuning it or the remix balancing parameters.

### Compare SynthSOD RMS activity with aligned scores

To listen to the old RMS decisions, export 20 random 10-second training-piece
crops into the same Git-ignored parent folder as the rendered Slakh comparisons:

```bash
python scripts/export_synthsod_rms_previews.py --count 20 --seed 42
```

Defaults use the original `/scratch/axs10538/datasets/synthsod/synthsod_index.pt`
and write to `rendered_stem_pairs/synthsod_rms/`. Each example contains every
individual stem MP3 (including RMS-inactive stems and stems absent from the RMS
index), `full_mix.mp3`, `rms_eligible_target_mix.mp3`, and `labels.txt`/`labels.json`.
Labels include both RMS class-union decisions and individual-stem eligibility,
with active seconds and exact crop times. The target mix is before class dropping,
context sampling and augmentation; it is an activity audit rather than a sampled
balanced training submix. All audio exports share one attenuation per example,
preserving relative stem levels. Every stem starts at the same crop offset.
The full mix includes all physical stems, including unlabelled context and
ambiguous percussion, so its label list describes only the configured targets.

Use `--config PATH` to match another experiment's sample rate, duration, activity
ratio and labels, or `--duration-sec 20` to override duration. Output directories
must be new; use `--output-dir rendered_stem_pairs/synthsod_rms_seed43 --seed 43`
for another batch. The script rejects aligned-score indexes.

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

Slakh and SynthSOD both treat viola, oboe, English horn, bassoon, tuba, French
horn and piccolo as background context, rather than nearby OpenMIC targets. Their
context identities are `viola`, `oboe`, `cor_anglais`, `bassoon`, `tuba`, `horn`
and `piccolo`. Existing raw MIDI indexes are remapped when loaded, so this policy
change does not require rebuilding them. Context inclusion still follows the
configured activity threshold, keep probability and maximum context-stem count.

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
produce a guitar label. SynthSOD stems are never replaced. Synthetic validation
can opt in with `deterministic_remix: true`, using only its own split's source
MIDIs and fixed replacement choices; OpenMIC testing does not use replacements.
When a valid render and its source MIDI exist but the original stem audio is
missing, the loader adds a render-only source using that MIDI's activity. It
chooses uniformly among its available renders without the replacement coin flip,
then applies the normal activity and class-dropping rules. Such a source does not
create a positive label for the missing original instrument. Source split/track
identities are preserved; missing training tracks can be assembled from their
available original stems plus renders. This happens at load time with no index
rebuild. Render-only sources are enabled by the same nonzero replacement setting.
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
