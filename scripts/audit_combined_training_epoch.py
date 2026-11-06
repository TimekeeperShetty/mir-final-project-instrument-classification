#!/usr/bin/env python3
"""Count one emitted training epoch and export a small sample of those exact crops."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import shutil
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.epoch_instrument_presence_stats import seed_everything, seed_worker
from scripts.export_synthsod_rms_previews import write_mp3
from training_base.config import load_json, parse_project_config
from training_base.data.datamodule import InstrumentDataModule, collate_slakh_batch


def dataset_source(item):
    return "synthsod" if str(item.get("track_id", "")).startswith("synthsod/") else "slakh"


class AuditDataset(Dataset):
    def __init__(self, dataset, previews):
        self.dataset = dataset
        self.previews = set(previews)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset.samples[index]
        save = index in self.previews
        inputs, target, debug = self.dataset._build_multilabel_excerpt_and_target(
            item, dataset_index=index, export_debug=save
        )
        capture = debug.pop("rendered_audio", None)
        audit = dict(dataset_index=index, track_id=item.get("track_id"), sample_id=item.get("sample_id"),
                     source=dataset_source(item), source_split=item["split"], **debug)
        return dict(inputs=inputs, target=target, domain=self.dataset.domain,
                    audit=audit, preview_audio=capture, save_preview=save)


def collate_audit(batch):
    result = collate_slakh_batch(batch)
    result["audit"] = [item["audit"] for item in batch]
    result["previews"] = [dict(batch_index=index, rendered_audio=item["preview_audio"])
                          for index, item in enumerate(batch) if item["save_preview"]]
    return result


def preview_indices(dataset, emitted_order, count, seed):
    # Only the saved examples are stratified. Counts still cover the entire epoch.
    buckets = defaultdict(list)
    for index in emitted_order:
        item = dataset.samples[index]
        source = dataset_source(item)
        if source == "slakh" and any(stem.get("rendered_variants")
                                     for stem in item.get("stems", []) + item.get("context_stems", [])):
            source = "slakh_with_render_candidates"
        buckets[source].append(index)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    while len(selected) < min(count, len(emitted_order)):
        for source in sorted(buckets):
            if buckets[source] and len(selected) < count:
                selected.append(buckets[source].pop())
    return selected


def save_preview(ffmpeg, output, position, audit, waveform, target, capture, labels, sample_rate):
    name = Path(str(audit["track_id"])).name
    directory = output / f"sample_{position:05d}_{audit['source']}_{name}"
    directory.mkdir()
    gain = capture["normalization_gain"]
    stems = []
    target_mix, context_mix = torch.zeros_like(waveform), torch.zeros_like(waveform)
    for number, record in enumerate(capture["stems"], 1):
        stem_audio = record["audio"] * gain
        origin = "synthsod" if audit["source"] == "synthsod" else "slakh_rendered" if record["replacement_provenance"] else "slakh_original"
        filename = f"{number:02d}_{record['role']}_{record['class_name']}_{record['stem_id']}_{origin}.mp3"
        metadata = {key: value for key, value in record.items() if key != "audio"}
        metadata.update(file=filename, origin=origin)
        stems.append((stem_audio, metadata))
        if record["role"] == "context":
            context_mix += stem_audio
        else:
            target_mix += stem_audio
    peak = max([1.0, float(waveform.abs().max()), float(target_mix.abs().max()), float(context_mix.abs().max())]
               + [float(stem_audio.abs().max()) for stem_audio, _ in stems])
    export_gain = 0.99 / peak
    write_mp3(ffmpeg, waveform * export_gain, sample_rate, directory / "mix.mp3")
    write_mp3(ffmpeg, target_mix * export_gain, sample_rate, directory / "targets.mp3")
    has_context = any(record["role"] == "context" for _, record in stems)
    if has_context:
        write_mp3(ffmpeg, context_mix * export_gain, sample_rate, directory / "context.mp3")
    for stem_audio, metadata in stems:
        write_mp3(ffmpeg, stem_audio * export_gain, sample_rate, directory / metadata["file"])
    active = [label for label, value in zip(labels, target.tolist()) if value >= 0.5]
    metadata = dict(audit)
    metadata.update(
        epoch_position=position, active_labels=active,
        labels={label: int(value >= 0.5) for label, value in zip(labels, target.tolist())},
        selected_stems=[record for _, record in stems], sample_rate=sample_rate,
        model_mix_normalization_gain=gain, shared_export_gain=export_gain,
        files=dict(mix="mix.mp3", targets="targets.mp3", context="context.mp3" if has_context else None),
        audio_note="Captured after stem augmentation in this counted sample; no second rendering. All files share export gain. MP3 is lossy.",
    )
    (directory / "labels.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"Track: {audit['track_id']} ({audit['source']}, {audit['source_split']})",
        f"Epoch position: {position}; crop: {audit['crop_start_sec']:.6f}-{audit['crop_end_sec']:.6f}s",
        f"Mode: {audit['mode']}", f"Active target labels: {', '.join(active) or '(none)'}",
        "Context: " + (", ".join(record["class_name"] for _, record in stems if record["role"] == "context") or "(none)"),
        "", "Stem | class | role | origin | active seconds | file",
    ]
    lines.extend(f"{record['stem_id']} | {record['class_name']} | {record['role']} | {record['origin']} | {record['active_sec']:.3f} | {record['file']}"
                 for _, record in stems)
    (directory / "labels.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dict(directory=directory.name, epoch_position=position, source=audit["source"],
                active_labels=active, has_rendered_targets=any(record["origin"] == "slakh_rendered" and record["role"] == "target" for _, record in stems))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/slakh_synthsod_aligned_rendered_to_openmic_clap_20_classes.json"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "rendered_stem_pairs" / "combined_epoch_audit")
    parser.add_argument("--preview-count", type=int, default=50)
    parser.add_argument("--seed", type=int, help="Override trainer.seed")
    parser.add_argument("--num-workers", type=int, help="Override loader worker count")
    args = parser.parse_args()
    if args.preview_count < 0 or (args.num_workers is not None and args.num_workers < 0):
        parser.error("Preview and worker counts must be nonnegative")
    ffmpeg = shutil.which("ffmpeg")
    if args.preview_count and ffmpeg is None:
        parser.error("ffmpeg with libmp3lame must be on PATH to export listening examples")
    raw_config = load_json(args.config)
    cfg = parse_project_config(raw_config)
    if cfg.data.task_type != "multilabel" or cfg.data.train.type != "slakh" or cfg.data.val is None or cfg.data.val.type != "slakh":
        parser.error("Use a combined synthetic multilabel training/validation config")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error("--output-dir must be a new folder")
    seed = cfg.trainer.seed if args.seed is None else args.seed
    seed_everything(seed)
    dm = InstrumentDataModule(cfg)
    dm.setup("fit")
    dataset = dm.train_dataset
    if not dataset.stem_remix_enabled:
        raise ValueError("The audit requires stem_remix=true")
    pool = Counter(dataset_source(item) for item in dataset.samples)
    val_pool = Counter(dataset_source(item) for item in dm.val_dataset.samples)
    render_capable = sum(any(stem.get("rendered_variants") for stem in item.get("stems", []) + item.get("context_stems", []))
                         for item in dataset.samples)
    if not pool["slakh"] or not pool["synthsod"] or not render_capable:
        raise ValueError(f"Expected Slakh, SynthSOD and rendered Slakh sources; pool={dict(pool)}, render-capable tracks={render_capable}")
    train_ids = {(dataset_source(item), item["track_id"]) for item in dataset.samples}
    val_ids = {(dataset_source(item), item["track_id"]) for item in dm.val_dataset.samples}
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"Training and validation track IDs overlap: {sorted(overlap)[:10]}")
    loader_cfg = cfg.data.train.loader
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    emitted_count = (len(order) // loader_cfg.batch_size) * loader_cfg.batch_size if loader_cfg.drop_last else len(order)
    if emitted_count == 0:
        raise ValueError("No batches would be emitted with this batch size/drop_last setting")
    selected = preview_indices(dataset, order[:emitted_count], args.preview_count, seed + 1)
    workers = loader_cfg.num_workers if args.num_workers is None else args.num_workers
    loader = DataLoader(
        AuditDataset(dataset, selected), sampler=order, batch_size=loader_cfg.batch_size,
        num_workers=workers, pin_memory=loader_cfg.pin_memory,
        persistent_workers=loader_cfg.persistent_workers and workers > 0,
        drop_last=loader_cfg.drop_last, collate_fn=collate_audit,
        worker_init_fn=seed_worker if workers else None,
        generator=torch.Generator().manual_seed(seed),
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(raw_config, indent=2) + "\n", encoding="utf-8")
    counts, stem_counts, context_counts = Counter(), Counter(), Counter()
    by_source, by_origin = defaultdict(Counter), defaultdict(Counter)
    source_examples, modes = Counter(), Counter()
    previews, empty, total, rendered_examples = [], 0, 0, 0

    def report(complete):
        warnings = []
        if complete and not rendered_examples:
            warnings.append("No rendered target survived selection this epoch; inspect availability and remix settings.")
        if complete and args.preview_count and not any(item["has_rendered_targets"] for item in previews):
            warnings.append("Saved examples contain no retained rendered target; use another seed or increase preview-count.")
        return dict(
            complete=complete, config=str(Path(args.config).resolve()), seed=seed,
            dataset_examples=len(dataset), epoch_examples=total, dropped_examples=len(dataset) - emitted_count,
            source_pool=dict(pool), validation_source_pool=dict(val_pool), render_capable_training_tracks=render_capable,
            emitted_source_examples=dict(source_examples), examples_with_rendered_targets=rendered_examples,
            empty_target_examples=empty, crop_modes=dict(modes),
            instruments={label: dict(present_examples=counts[label], absent_examples=total-counts[label],
                                     presence_percent=100 * counts[label] / total if total else 0,
                                     selected_logical_target_stems=stem_counts[label],
                                     slakh_clip_count=by_source["slakh"][label], synthsod_clip_count=by_source["synthsod"][label],
                                     slakh_original_contribution=by_origin["slakh_original"][label],
                                     slakh_rendered_contribution=by_origin["slakh_rendered"][label],
                                     synthsod_contribution=by_origin["synthsod"][label]) for label in dataset.label_names},
            context_clip_counts=dict(context_counts), previews=previews, warnings=warnings,
            notes=["Counts cover the entire emitted training epoch after replacement, activity filtering and class dropping.",
                   "Preview selection is stratified by source availability; the epoch itself is not reweighted.",
                   "Origin contributions count clips per origin and can overlap for a class within one clip.",
                   "A grouped SynthSOD violin is one logical stem backed by both section files.",
                   "This is one seeded audit epoch without model training, not a forecast of every future epoch."],
        )

    print(f"Training pool: {dict(pool)}; rendered-capable tracks: {render_capable}; validation: {dict(val_pool)}", flush=True)
    progress = tqdm(loader, desc="Combined training epoch", unit="batch")
    for batch_number, batch in enumerate(progress, 1):
        binary = batch["targets"] >= 0.5
        counts.update({label: int(binary[:, index].sum()) for index, label in enumerate(dataset.label_names)})
        empty += int((binary.sum(dim=1) == 0).sum())
        for local_index, audit in enumerate(batch["audit"]):
            labels = {label for index, label in enumerate(dataset.label_names) if binary[local_index, index]}
            by_source[audit["source"]].update(labels)
            source_examples[audit["source"]] += 1
            modes[audit["mode"]] += 1
            origins, contexts = defaultdict(set), set()
            for stem in audit["selected_stems"]:
                if stem["role"] == "context":
                    contexts.add(stem["class_name"])
                    continue
                origin = "synthsod" if audit["source"] == "synthsod" else "slakh_rendered" if stem.get("replacement_provenance") else "slakh_original"
                origins[origin].add(stem["class_name"])
                stem_counts[stem["class_name"]] += 1
            for origin, names in origins.items():
                by_origin[origin].update(names)
            rendered_examples += bool(origins.get("slakh_rendered"))
            context_counts.update(contexts)
        for preview in batch["previews"]:
            index = preview["batch_index"]
            previews.append(save_preview(ffmpeg, output, total + index + 1, batch["audit"][index],
                                         batch["inputs"][index], batch["targets"][index], preview["rendered_audio"],
                                         dataset.label_names, dataset.sample_rate))
        total += len(batch["audit"])
        progress.set_postfix(examples=total, rendered=rendered_examples, saved=len(previews))
        if batch_number % 10 == 0:
            (output / "distribution.json").write_text(json.dumps(report(False), indent=2) + "\n", encoding="utf-8")
    summary = report(True)
    (output / "distribution.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["instrument"] + list(next(iter(summary["instruments"].values()))))
        writer.writeheader()
        writer.writerows(dict(instrument=label, **values) for label, values in summary["instruments"].items())
    print(f"\nEmitted {total} examples; {rendered_examples} contain rendered targets; saved {len(previews)} previews.")
    print(f"{'instrument':<20} {'present':>8} {'present%':>10} {'Slakh':>8} {'SynthSOD':>9} {'rendered':>9}")
    for label, values in summary["instruments"].items():
        print(f"{label:<20} {values['present_examples']:>8} {values['presence_percent']:>9.2f}% "
              f"{values['slakh_clip_count']:>8} {values['synthsod_clip_count']:>9} {values['slakh_rendered_contribution']:>9}")
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Reports and listening examples: {output}")


if __name__ == "__main__":
    main()
