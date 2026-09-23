"""Evaluate a saved checkpoint on validation once, sweeping frame thresholds."""

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
from inspect_prediction import _move_to_device
from source_separator import SourceSeparator
from utils.audioset_metrics import ConditionalFrameAccuracy


DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
METRICS = ("precision", "recall", "f1", "accuracy")


def best_checkpoint(run_dir):
    """Use ModelCheckpoint's saved choice, not rounded log losses or filenames."""
    run_dir = Path(run_dir).resolve()
    last = torch.load(run_dir / "mylast.ckpt", map_location="cpu", weights_only=False)
    paths = {
        state["best_model_path"]
        for key, state in last.get("callbacks", {}).items()
        if "'monitor': 'val_loss'" in key and isinstance(state, dict)
        and state.get("best_model_path")
    }
    if len(paths) != 1:
        raise ValueError("Cannot identify a unique val_loss best checkpoint; use --checkpoint")
    saved = Path(paths.pop())
    # The run may have moved; its version directory and checkpoint name are stable.
    local = run_dir / saved.parent.name / saved.name
    if local.is_file():
        return local.resolve()
    if saved.is_file():
        return saved.resolve()
    raise FileNotFoundError(f"Best checkpoint is missing: {local} (saved path: {saved})")


@torch.inference_mode()
def sweep_batches(module, batches, device, thresholds=DEFAULT_THRESHOLDS,
                  epoch=0, log_every=50):
    """One forward per batch; only per-pair TP/FP/FN/TN survive the batch."""
    thresholds = tuple(thresholds)
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("Thresholds must be nonempty and unique")
    metrics = [ConditionalFrameAccuracy(value) for value in thresholds]
    module.eval()
    batch_count = 0
    for batch_count, batch in enumerate(batches, start=1):
        model_input, targets = _move_to_device(batch, device)
        predictions = module.task.forward(module.model, model_input, targets, epoch)
        # Reuse this exact frame_probs tensor for every threshold before discarding it.
        for metric in metrics:
            metric.update(predictions, targets)
        if log_every and batch_count % log_every == 0:
            print(f"Validation batches processed: {batch_count}", flush=True)
    results = []
    for metric in metrics:
        counts = metric.compute()  # Existing epoch aggregation and DDP deduplication.
        results.append({
            "threshold": metric.threshold,
            **{name: counts[f"conditional_frame_{name}"] for name in METRICS},
            **{name: counts[name] for name in ("tp", "fp", "fn", "tn", "valid_frames")},
        })
        metric.reset()
    return results, batch_count


def run(checkpoint_path, output_dir=None, device="auto", batch_size=None,
        num_workers=None, thresholds=DEFAULT_THRESHOLDS):
    checkpoint_path = Path(checkpoint_path).resolve()
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
    batch_size = hp.batch_size if batch_size is None else batch_size
    num_workers = hp.num_workers if num_workers is None else num_workers
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")

    # Match training's inference numerics, without creating a Trainer or optimizer.
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = bool(hp.cudnn_deterministic)
    torch.backends.cudnn.benchmark = False  # Same as validation_step.
    module = SourceSeparator(hp)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    epoch = checkpoint.get("epoch", 0)
    del checkpoint  # Optimizer state is never restored and need not remain in memory.
    module.to(device).eval()
    # datapath_creator verifies split settings and the saved training inventory.
    entries = datapath_creator(hp, "cv")
    dataset = dataset_creator(hp, "cv", path_list=entries, is_training=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=module.collate_fn)
    print(f"Checkpoint: {checkpoint_path} (epoch={epoch}, zero-based)", flush=True)
    print(f"Validation: {len(entries)} clips, {len(dataset)} clip-condition pairs, "
          f"{len(loader)} batches; device={device}", flush=True)
    results, batch_count = sweep_batches(module, loader, device, thresholds, epoch)
    report = {
        "checkpoint": str(checkpoint_path), "checkpoint_epoch": epoch,
        "partition": "cv", "split_path": str(split_path),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "validation_clips": len(entries), "validation_samples": len(dataset),
        "prediction_batches": batch_count, "batch_size": batch_size,
        "device": str(device), "results": results,
    }
    output_dir = (Path(output_dir) if output_dir is not None else
                  checkpoint_path.parent / f"{checkpoint_path.stem}_threshold_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "threshold_sweep.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print("\nThreshold  Precision     Recall         F1   Accuracy")
    for row in results:
        values = [f"{row[name]:.6f}" if row[name] is not None else "N/A" for name in METRICS]
        print(f"{row['threshold']:9.2f}  " + "  ".join(f"{value:>9}" for value in values))
    print(f"Valid frames: {results[0]['valid_frames']}")
    print(f"Reports: {output_dir.resolve()}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Explicit Lightning .ckpt file")
    source.add_argument("--run-dir", type=Path,
                        help="Experiment directory; select val_loss best via mylast.ckpt")
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, help="Default: checkpoint batch size")
    parser.add_argument("--num-workers", type=int, help="Default: checkpoint worker count")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint if args.checkpoint else best_checkpoint(args.run_dir)
    run(checkpoint, args.output_dir, args.device, args.batch_size, args.num_workers,
        args.thresholds)


if __name__ == "__main__":
    main()
