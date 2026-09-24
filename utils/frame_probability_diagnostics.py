"""Exact probability diagnostics for previously unpredicted Strong-positive classes."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from utils.audioset_metrics import ConditionalFrameAccuracy, frame_metrics_from_counts, tensor


PROBABILITY_FIELDS = (
    "MID", "class_name", "positive_frames", "valid_frames",
    "max_probability", "mean_probability", "p95_probability",
    "gt_positive_mean_probability", "gt_positive_max_probability",
    "gt_negative_mean_probability", "positive_negative_mean_gap",
    "selection_threshold", "predicted_positive_frames_at_selection_threshold",
    "diagnostic_threshold", "predicted_positive_frames", "TP", "FP", "FN",
    "Precision", "Recall", "F1", "threshold_observation",
)


class FrameProbabilityDiagnostics:
    """Keep masked probabilities only for selected classes, keyed like epoch metrics.

    Exact percentiles require frame values; this optional standalone diagnostic
    does not change the counts-only accumulator used by training/validation hooks.
    """

    def __init__(self, reference_rows, selection_threshold, diagnostic_threshold=0.5):
        self.selection_threshold = ConditionalFrameAccuracy(selection_threshold).threshold
        self.diagnostic_threshold = ConditionalFrameAccuracy(diagnostic_threshold).threshold
        self.classes = {}
        mids = set()
        for row in reference_rows:
            category, mid = row["class_index"], row["mid"]
            if category in self.classes or mid in mids:
                raise ValueError(f"Duplicate diagnostic class: {mid}")
            counts = [row[name] for name in ("tp", "fp", "fn", "tn")]
            if any(type(value) is not int or value < 0 for value in counts):
                raise ValueError(f"Invalid reference counts: {mid}")
            if counts[0] or counts[1] or counts[2] <= 0 or sum(counts) != row["valid_frames"]:
                raise ValueError(f"Expected Strong-positive class with no predictions: {mid}")
            self.classes[category] = dict(row)
            mids.add(mid)
        if not self.classes:
            raise ValueError("Reference has no Strong-positive classes without predictions")
        self.rows = {}

    @classmethod
    def from_report(cls, path, checkpoint_path, epoch, split_sha256, dataset,
                    validation_clips, diagnostic_threshold=0.5):
        path = Path(path).resolve()
        reference = json.loads(path.read_text(encoding="utf-8"))
        # Support moving a whole experiment, retaining its version/checkpoint names.
        if (reference["checkpoint_epoch"] != epoch or
                Path(reference["checkpoint"]).parts[-2:] != Path(checkpoint_path).parts[-2:]):
            raise ValueError("Probability reference uses a different checkpoint")
        if reference["partition"] != "cv" or reference["split_sha256"] != split_sha256:
            raise ValueError("Probability reference uses a different validation split")
        if (reference["validation_samples"] != len(dataset) or
                reference["validation_clips"] != validation_clips):
            raise ValueError("Probability reference validation inventory does not match")
        results = reference["results"]
        if len(results) != 1 or "per_class" not in results[0]:
            raise ValueError("Use threshold_sweep.json from the original --per-class evaluation")
        all_rows = results[0]["per_class"]
        if len(all_rows) != len(dataset.mid_to_index) or len({r["mid"] for r in all_rows}) != len(all_rows):
            raise ValueError("Probability reference class map does not match")
        for row in all_rows:
            index = dataset.mid_to_index.get(row["mid"])
            if (index is None or index != row["class_index"] or
                    dataset.class_names[index] != row["class_name"]):
                raise ValueError(f"Probability reference class map mismatch: {row['mid']}")
        selected = [row for row in all_rows
                    if row["tp"] == 0 and row["fp"] == 0 and row["fn"] > 0]
        collector = cls(selected, results[0]["threshold"], diagnostic_threshold)
        collector.provenance = {
            "reference_report": str(path),
            "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "checkpoint": str(checkpoint_path), "checkpoint_epoch": epoch,
            "split_sha256": split_sha256,
        }
        return collector

    def update(self, predictions, targets):
        selected = [(i, clip, int(category)) for i, (clip, category) in
                    enumerate(zip(targets["utt_id"], targets["class_index"]))
                    if int(category) in self.classes]
        if not selected:
            return
        probabilities = predictions["frame_probs"].detach()
        labels = tensor(targets["strong_labels"]).to(probabilities.device)
        mask = tensor(targets["strong_mask"]).to(device=probabilities.device, dtype=torch.bool)
        mask = mask & predictions["frame_mask"].bool()
        for index, clip, category in selected:
            scores = probabilities[index][mask[index]]
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Nonfinite probabilities in diagnostic valid frames")
            if ((scores < 0) | (scores > 1)).any():
                raise ValueError("Diagnostic frame_probs must be probabilities in [0, 1]")
            # Copy so retained values never keep the full batch tensor alive.
            self.rows[(clip, category)] = (
                scores.cpu().numpy().copy(),
                (labels[index][mask[index]] >= 0.5).cpu().numpy().copy(),
            )

    def compute(self):
        rows = self.rows
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, rows)
            rows = {key: value for part in gathered for key, value in part.items()}
        by_class = {category: [] for category in self.classes}
        for (_, category), values in rows.items():
            by_class[category].append(values)
        results = []
        for category, reference in self.classes.items():
            pairs = by_class[category]
            if not pairs:
                raise ValueError(f"No validation predictions for reference class {reference['mid']}")
            scores = np.concatenate([pair[0] for pair in pairs])
            labels = np.concatenate([pair[1] for pair in pairs])
            positive_count = int(labels.sum())
            if positive_count != reference["fn"] or scores.size != reference["valid_frames"]:
                raise ValueError(f"Reference frame counts changed for {reference['mid']}")
            # Cast the threshold to the tensor's original dtype, just as PyTorch does.
            at_selection = scores >= np.asarray(self.selection_threshold, dtype=scores.dtype)
            if at_selection.any():
                raise ValueError(f"Reference no-prediction result not reproduced for {reference['mid']}")
            predicted = scores >= np.asarray(self.diagnostic_threshold, dtype=scores.dtype)
            tp = int((predicted & labels).sum())
            fp = int((predicted & ~labels).sum())
            fn = positive_count - tp
            tn = int(scores.size) - positive_count - fp
            metrics = frame_metrics_from_counts(tp, fp, fn, tn)
            positive_scores, negative_scores = scores[labels], scores[~labels]
            positive_mean = float(positive_scores.mean(dtype=np.float64))
            negative_mean = (float(negative_scores.mean(dtype=np.float64))
                             if negative_scores.size else None)
            observation = ("gt_positives_recovered" if tp else
                           "only_false_positives" if fp else "all_scores_below_threshold")
            results.append({
                "MID": reference["mid"], "class_name": reference["class_name"],
                "positive_frames": positive_count, "valid_frames": int(scores.size),
                "max_probability": float(scores.max()),
                "mean_probability": float(scores.mean(dtype=np.float64)),
                "p95_probability": float(np.percentile(scores, 95, method="linear")),
                "gt_positive_mean_probability": positive_mean,
                "gt_positive_max_probability": float(positive_scores.max()),
                "gt_negative_mean_probability": negative_mean,
                "positive_negative_mean_gap": (positive_mean - negative_mean
                                               if negative_mean is not None else None),
                "selection_threshold": self.selection_threshold,
                "predicted_positive_frames_at_selection_threshold": 0,
                "diagnostic_threshold": self.diagnostic_threshold,
                "predicted_positive_frames": tp + fp, "TP": tp, "FP": fp, "FN": fn,
                "Precision": metrics["precision"], "Recall": metrics["recall"],
                "F1": metrics["f1"], "threshold_observation": observation,
            })
        return sorted(results, key=lambda row: (-row["positive_frames"], row["MID"]))

    def reset(self):
        self.rows.clear()


def write_probability_reports(rows, output_dir, provenance):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, subset in (("no_prediction_probability_stats.csv", rows),
                             ("no_prediction_probability_top20.csv", rows[:20])):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PROBABILITY_FIELDS)
            writer.writeheader()
            writer.writerows(subset)
    summary = {
        **provenance, "class_count": len(rows),
        "selection_threshold": rows[0]["selection_threshold"],
        "diagnostic_threshold": rows[0]["diagnostic_threshold"],
        "threshold_observation_counts": dict(Counter(row["threshold_observation"] for row in rows)),
        "scope": "Only strong_mask & frame_mask; global deduplication by (clip, class_index)",
        "percentile": "Exact 95th percentile across all class valid frames, linear interpolation",
        "mean": "Frame-weighted mean accumulated in float64, not mean of batch/clip means",
        "interpretation": (
            "gt_positives_recovered: lowering the selection threshold recovers some TP; "
            "only_false_positives: predictions at the diagnostic threshold still have TP=0; "
            "all_scores_below_threshold: every valid score is below the diagnostic threshold. "
            "Low scores alone cannot establish that a class was not learned. "
            "Compare GT-positive and GT-negative scores and the diagnostic precision/recall/F1."),
    }
    (output_dir / "no_prediction_probability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nProbability diagnostic classes: {len(rows)}; "
          f"selection threshold={summary['selection_threshold']}, "
          f"diagnostic threshold={summary['diagnostic_threshold']}")
    print("Top 20 by positive_frames (all-valid max/mean/p95; GT-positive mean/max):")
    for row in rows[:20]:
        values = "  ".join(f"{row[key]:.6f}" for key in PROBABILITY_FIELDS[4:9])
        print(f"{row['MID']}  {row['class_name']}  {row['positive_frames']}  {values}")
    print(f"Threshold observations: {summary['threshold_observation_counts']}")
    return summary
