"""Export Strong-positive classes and diagnose zero F1 from evaluation counts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


FIELDS = ("MID", "class_name", "positive_frames", "predicted_positive_frames",
          "TP", "FP", "FN", "Precision", "Recall", "F1")
NO_PREDICTIONS = "no_predictions"
PREDICTED_BUT_TP_ZERO = "predicted_but_tp_zero"


def positive_class_statistics(per_class_rows):
    """Filter TP+FN>0 and classify exact zero F1 using integer counts."""
    positive, zero_f1, seen_mids = [], [], set()
    source_class_count = 0
    for row in per_class_rows:
        source_class_count += 1
        mid = row["mid"]
        if not mid or mid in seen_mids:
            raise ValueError(f"Empty or duplicate MID in per-class results: {mid!r}")
        seen_mids.add(mid)
        tp, fp, fn = (int(str(row[key])) for key in ("tp", "fp", "fn"))
        if min(tp, fp, fn) < 0:
            raise ValueError(f"Negative confusion counts for {mid}")
        positive_frames = tp + fn
        if positive_frames == 0:
            continue
        predicted_frames = tp + fp
        result = {
            "MID": mid, "class_name": row["class_name"],
            "positive_frames": positive_frames,
            "predicted_positive_frames": predicted_frames,
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": tp / predicted_frames if predicted_frames else 0.0,
            "Recall": tp / positive_frames,
            "F1": 2 * tp / (2 * tp + fp + fn),
        }
        positive.append(result)
        # With positive support, F1 == 0 iff TP == 0. Never classify rounded scores.
        if tp == 0:
            zero_f1.append({**result, "zero_f1_reason": (
                NO_PREDICTIONS if predicted_frames == 0 else PREDICTED_BUT_TP_ZERO)})
    no_predictions = [row for row in zero_f1 if row["zero_f1_reason"] == NO_PREDICTIONS]
    predicted_but_tp_zero = [row for row in zero_f1 if row["zero_f1_reason"] == PREDICTED_BUT_TP_ZERO]
    summary = {
        "source_class_count": source_class_count,
        "positive_class_count": len(positive),
        "excluded_no_strong_positive_class_count": source_class_count - len(positive),
        "zero_f1_class_count": len(zero_f1),
        "no_predictions_class_count": len(no_predictions),
        "predicted_but_tp_zero_class_count": len(predicted_but_tp_zero),
        "nonzero_f1_class_count": len(positive) - len(zero_f1),
        "strong_positive_macro_f1": (sum(row["F1"] for row in positive) / len(positive)
                                     if positive else None),
        "positive_definition": "TP + FN > 0 on the original supervised valid frames",
        "predicted_positive_definition": "TP + FP on the same supervised valid frames",
        "zero_f1_definitions": {
            NO_PREDICTIONS: "TP + FP == 0 and TP + FN > 0",
            PREDICTED_BUT_TP_ZERO: "TP == 0 and FP > 0 and TP + FN > 0",
        },
    }
    return positive, zero_f1, no_predictions, predicted_but_tp_zero, summary


def write_positive_class_reports(per_class_rows, output_dir, provenance=None):
    positive, zero_f1, no_predictions, predicted_but_tp_zero, summary = (
        positive_class_statistics(per_class_rows))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exports = (
        ("strong_positive_class_metrics.csv", positive, FIELDS),
        ("zero_f1_classes.csv", zero_f1, (*FIELDS, "zero_f1_reason")),
        ("zero_f1_no_predictions.csv", no_predictions, FIELDS),
        ("zero_f1_predicted_but_tp_zero.csv", predicted_but_tp_zero, FIELDS),
    )
    for filename, rows, fields in exports:
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    summary["provenance"] = provenance or {}
    (output_dir / "positive_class_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    print(f"Strong-positive classes={len(positive)}, F1=0 classes={len(zero_f1)}, "
          f"no predictions={len(no_predictions)}, predicted but TP=0={len(predicted_but_tp_zero)}",
          flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True,
                        help="Existing per_class_metrics.csv from checkpoint validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.input_csv.resolve()
    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    write_positive_class_reports(rows, args.output_dir, provenance={
        "source_csv": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "evaluation": "Counts retain the source checkpoint, threshold, mask and deduplication",
    })
    print(f"Reports: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
