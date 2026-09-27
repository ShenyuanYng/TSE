"""Rank other class conditions on selected targets' GT-positive validation frames."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from creators.dataset import datapath_creator, dataset_creator
from datasets.paths.audioset_paths import resolve_path
from datasets.train_split import validate_split_config
from source_separator import SourceSeparator
from sweep_validation_thresholds import best_checkpoint
from utils.audioset_ontology import AudioSetOntology
from utils.ontology_confusion import (DEFAULT_TARGET_MIDS, TOP_FIELDS,
                                      OntologyConfusionAccumulator, probe_batches)


DEFAULT_ONTOLOGY = Path(__file__).resolve().parent / "resources/audioset_ontology.json"


def validate_reference(reference, checkpoint_path, epoch, split_sha256, dataset, clip_count):
    if (reference["checkpoint_epoch"] != epoch or
            Path(reference["checkpoint"]).parts[-2:] != Path(checkpoint_path).parts[-2:]):
        raise ValueError("Reference checkpoint does not match")
    if reference["partition"] != "cv" or reference["split_sha256"] != split_sha256:
        raise ValueError("Reference validation split does not match")
    if reference["validation_samples"] != len(dataset) or reference["validation_clips"] != clip_count:
        raise ValueError("Reference validation inventory does not match")
    if len(reference["results"]) != 1 or "per_class" not in reference["results"][0]:
        raise ValueError("Use the original per-class threshold_sweep.json as the reference")
    records = reference["results"][0]["per_class"]
    by_mid = {row["mid"]: row for row in records}
    if len(by_mid) != len(records) or set(by_mid) != set(dataset.mid_to_index):
        raise ValueError("Reference class map does not match")
    for mid, index in dataset.mid_to_index.items():
        if by_mid[mid]["class_index"] != index or by_mid[mid]["class_name"] != dataset.class_names[index]:
            raise ValueError(f"Reference class map differs for {mid}")
    return by_mid


def write_reports(output_dir, top_rows, all_rows, summaries, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows, fields in (
            ("ontology_confusion_top10.csv", top_rows, TOP_FIELDS),
            ("ontology_confusion_all_classes.csv", all_rows, TOP_FIELDS),
            ("ontology_target_summary.csv", summaries, tuple(summaries[0]))):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "ontology_confusion_summary.json").write_text(
        json.dumps({**metadata, "targets": summaries}, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    print("\nOther classes ranked by mean probability on target GT-positive frames:")
    for row in top_rows:
        print(f"{row['target_class_name']} | {row['rank']:2d} | {row['other_MID']} "
              f"{row['other_class_name']} | mean={row['other_gt_mean_probability']:.6f} "
              f"max={row['other_gt_max_probability']:.6f} | {row['ontology_relation']}")
    print(f"Reports: {output_dir.resolve()}", flush=True)


def run(checkpoint_path, reference_report, output_dir, ontology_path=DEFAULT_ONTOLOGY,
        target_mids=DEFAULT_TARGET_MIDS, device="auto", num_workers=None,
        class_chunk_size=32, frame_chunk_size=256):
    if class_chunk_size < 1 or frame_chunk_size < 1:
        raise ValueError("Class/frame chunk sizes must be positive")
    checkpoint_path, reference_report = Path(checkpoint_path).resolve(), Path(reference_report).resolve()
    reference = json.loads(reference_report.read_text(encoding="utf-8"))
    ontology_path = Path(ontology_path).resolve()
    ontology = AudioSetOntology.load(ontology_path)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = argparse.Namespace(**dict(checkpoint["hyper_parameters"]))
    validate_split_config(hp)
    if hp.pad_back_or_front != 0:
        raise ValueError("AudioSet packed sequences require padding at the back")
    split_path = Path(resolve_path(hp.train_split_path, hp.data_path)).resolve()
    if not split_path.is_file():
        raise FileNotFoundError(f"Saved validation split is required: {split_path}")
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = bool(hp.cudnn_deterministic)
    torch.backends.cudnn.benchmark = False
    module = SourceSeparator(hp)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    epoch = checkpoint.get("epoch", 0)
    del checkpoint
    module.to(device).eval()
    entries = datapath_creator(hp, "cv")
    dataset = dataset_creator(hp, "cv", path_list=entries, is_training=False)
    expected_counts = validate_reference(reference, checkpoint_path, epoch, split_hash, dataset, len(entries))
    for mid in target_mids:
        if mid not in expected_counts or mid not in ontology.nodes:
            raise ValueError(f"Target MID is missing from the class map or ontology: {mid}")
        if expected_counts[mid]["tp"] + expected_counts[mid]["fn"] == 0:
            raise ValueError(f"Target has no GT-positive frames in the reference: {mid}")
    entries_by_id = {entry["id"]: entry for entry in entries}
    if len(entries_by_id) != len(entries):
        raise ValueError("Validation inventory contains duplicate clip IDs")
    # GroupNorm includes padded time positions. Retain the reference batch size,
    # sample ordering and full batches rather than filtering the loader to 7 MIDs.
    batch_size = reference["batch_size"]
    num_workers = hp.num_workers if num_workers is None else num_workers
    if batch_size < 1 or num_workers < 0:
        raise ValueError("Invalid inference batch size/worker count")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=module.collate_fn)
    collector = OntologyConfusionAccumulator(dataset.mid_to_index, dataset.class_names, target_mids, ontology)
    print(f"Checkpoint: {checkpoint_path} (epoch={epoch})", flush=True)
    print(f"Validation: {len(entries)} clips, {len(dataset)} clip-MID pairs, {len(loader)} batches", flush=True)
    print(f"Targets: {len(target_mids)}; score all {dataset.num_classes} model classes; "
          f"ontology nodes: {len(ontology.nodes)}", flush=True)
    try:
        batch_count = probe_batches(module, loader, device, collector, entries_by_id,
                                    dataset.hop_length, dataset.sr, epoch,
                                    class_chunk_size, frame_chunk_size)
        top_rows, all_rows, summaries = collector.compute(expected_counts=expected_counts)
    finally:
        collector.reset()
    metadata = {
        "checkpoint": str(checkpoint_path), "checkpoint_epoch": epoch,
        "partition": "cv", "split_path": str(split_path), "split_sha256": split_hash,
        "reference_report": str(reference_report),
        "reference_sha256": hashlib.sha256(reference_report.read_bytes()).hexdigest(),
        "ontology_path": str(ontology_path),
        "ontology_sha256": hashlib.sha256(ontology_path.read_bytes()).hexdigest(),
        "ontology_source": "https://github.com/audioset/ontology",
        "ontology_license": "CC BY-SA 4.0, Google Inc.",
        "ontology_node_count": len(ontology.nodes), "model_class_count": dataset.num_classes,
        "model_mids_missing_from_ontology": sorted(set(dataset.mid_to_index) - set(ontology.nodes)),
        "validation_clips": len(entries), "validation_samples": len(dataset),
        "prediction_batches": batch_count, "batch_size": batch_size, "device": str(device),
        "class_chunk_size": class_chunk_size, "frame_chunk_size": frame_chunk_size,
        "strong_annotation_complete": hp.strong_annotation_complete,
        "target_mids": list(target_mids),
        "frame_scope": "Target strong_labels >= 0.5 AND target strong_mask AND frame_mask",
        "ranking": "Other-class sigmoid mean over all target GT-positive frames; MID breaks ties",
        "deduplication": "Merge (clip, target class_index) summaries across ranks before aggregation",
        "relation_direction": "other_is_child/descendant/parent/ancestor is relative to target MID",
        "ontology_path_direction": "Parent to child, shortest directed path in the full ontology DAG",
        "annotation_evidence": "Weak presence is clip-level. Strong overlap uses exact alternate MID, "
                               "original time grid and PRESENT intervals with NOT_PRESENT overriding.",
        "limitations": [
            "Conditions are scored independently and are not mutually exclusive probabilities.",
            "High alternate scores, hierarchy relations and annotation overlap do not prove a labeling error.",
            "This analysis only covers targets with exact-MID GT-positive frames; it cannot explain all "
            "weak targets lacking exact-MID PRESENT annotations.",
            "No ontology propagation, model/loss changes, threshold changes or training occur.",
        ],
    }
    is_primary = not (torch.distributed.is_available() and torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
    if is_primary:
        write_reports(output_dir, top_rows, all_rows, summaries, metadata)
    return {**metadata, "targets": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--run-dir", type=Path, help="Select val_loss best via the saved mylast.ckpt callback")
    parser.add_argument("--reference-report", type=Path, required=True,
                        help="Original per-class threshold_sweep.json, e.g. probability_reference_37074.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--target-mids", nargs="+", default=list(DEFAULT_TARGET_MIDS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--class-chunk-size", type=int, default=32)
    parser.add_argument("--frame-chunk-size", type=int, default=256)
    args = parser.parse_args()
    checkpoint = args.checkpoint if args.checkpoint else best_checkpoint(args.run_dir)
    run(checkpoint, args.reference_report, args.output_dir, args.ontology, args.target_mids,
        args.device, args.num_workers, args.class_chunk_size, args.frame_chunk_size)


if __name__ == "__main__":
    main()
