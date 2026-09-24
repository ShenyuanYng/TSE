"""Masked, deduplicated, frame-weighted diagnostics of unpredicted classes."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from utils.frame_probability_diagnostics import FrameProbabilityDiagnostics, write_probability_reports


def reference(category=0, mid="A", positive=2, valid=4):
    return {"class_index": category, "mid": mid, "class_name": f"Class, {mid}",
            "tp": 0, "fp": 0, "fn": positive, "tn": valid - positive, "valid_frames": valid}


def update(collector, clip, category, scores, labels, strong=None, frames=None):
    collector.update(
        {"frame_probs": torch.tensor([scores]),
         "frame_mask": torch.tensor([frames or [1] * len(scores)])},
        {"strong_labels": (torch.tensor([labels]), [len(scores)]),
         "strong_mask": (torch.tensor([strong or [1] * len(scores)]), [len(scores)]),
         "utt_id": [clip], "class_index": [category]},
    )


class ProbabilityDiagnosticsTest(unittest.TestCase):
    def test_global_frame_distribution_masks_duplicates_boundary_and_reset(self):
        collector = FrameProbabilityDiagnostics([reference()], .7)
        # Both kinds of masking exclude high/nonfinite scores. Frame means and
        # quantiles differ from averages of the two unequal-sized clip summaries.
        for _ in range(2):
            update(collector, "a", 0, [.1, .5, .6, .99, float("nan")], [0, 1, 0, 1, 1],
                   strong=[1, 1, 1, 0, 1], frames=[1, 1, 1, 1, 0])
        update(collector, "b", 0, [.2], [1])
        update(collector, "ignored", 9, [float("nan")], [1])
        row, = collector.compute()
        self.assertEqual(len(collector.rows), 2)
        self.assertEqual((row["positive_frames"], row["valid_frames"]), (2, 4))
        scores = np.array([.1, .5, .6, .2], dtype=np.float32)
        self.assertAlmostEqual(row["mean_probability"], .35)
        self.assertAlmostEqual(row["p95_probability"], .585)
        self.assertEqual(row["p95_probability"], float(np.percentile(scores, 95)))
        self.assertAlmostEqual(row["max_probability"], .6)
        self.assertAlmostEqual(row["gt_positive_mean_probability"], .35)
        self.assertEqual(row["gt_positive_max_probability"], .5)
        self.assertAlmostEqual(row["gt_negative_mean_probability"], .35)
        self.assertAlmostEqual(row["positive_negative_mean_gap"], 0.)
        self.assertEqual([row[key] for key in ("TP", "FP", "FN", "predicted_positive_frames")], [1, 1, 1, 2])
        self.assertEqual((row["Precision"], row["Recall"], row["F1"]), (.5, .5, .5))
        self.assertEqual(row["threshold_observation"], "gt_positives_recovered")
        collector.reset()
        self.assertFalse(collector.rows)
        with self.assertRaisesRegex(ValueError, "No validation predictions"):
            collector.compute()

    def test_distributed_dedup_precedes_percentiles_and_counts(self):
        collector = FrameProbabilityDiagnostics([reference(positive=2, valid=3)], .7)
        update(collector, "a", 0, [.1, .2], [0, 1])
        # The last rank wins on a duplicate, matching existing metric semantics.
        remote = {("a", 0): (np.array([.2, .3]), np.array([False, True])),
                  ("b", 0): (np.array([.4]), np.array([True]))}

        def gather(parts, rows):
            parts[:] = [rows, remote]

        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=2), \
             patch("torch.distributed.all_gather_object", side_effect=gather):
            row, = collector.compute()
        self.assertEqual(row["valid_frames"], 3)
        self.assertAlmostEqual(row["mean_probability"], .3)
        self.assertAlmostEqual(row["p95_probability"], .39)
        self.assertAlmostEqual(row["gt_positive_mean_probability"], .35)
        self.assertEqual(row["threshold_observation"], "all_scores_below_threshold")

    def test_nonfinite_and_out_of_range_valid_scores_rejected(self):
        for score in (float("nan"), float("inf"), -.1, 1.1):
            collector = FrameProbabilityDiagnostics([reference()], .7)
            with self.assertRaises((FloatingPointError, ValueError)):
                update(collector, "a", 0, [score], [1])

    def test_reference_selection_and_support_must_reproduce(self):
        for scores, labels, message in (([.7, .1], [1, 0], "not reproduced"),
                                        ([.2, .1], [1, 1], "counts changed"),
                                        ([.2], [1], "counts changed")):
            collector = FrameProbabilityDiagnostics([reference(positive=1, valid=2)], .7)
            update(collector, "a", 0, scores, labels)
            with self.assertRaisesRegex(ValueError, message):
                collector.compute()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            FrameProbabilityDiagnostics([reference(), reference()], .7)
        with self.assertRaisesRegex(ValueError, "Strong-positive"):
            FrameProbabilityDiagnostics([reference(positive=0)], .7)

    def test_false_positive_only_no_negatives_sorting_and_csv(self):
        collector = FrameProbabilityDiagnostics(
            [reference(positive=1, valid=2), reference(1, "B", positive=3, valid=3)], .7)
        update(collector, "a", 0, [.1, .6], [1, 0])
        update(collector, "b", 1, [.1, .2, .3], [1, 1, 1])
        rows = collector.compute()
        self.assertEqual([row["MID"] for row in rows], ["B", "A"])
        self.assertIsNone(rows[0]["gt_negative_mean_probability"])
        self.assertIsNone(rows[0]["positive_negative_mean_gap"])
        self.assertEqual(rows[1]["threshold_observation"], "only_false_positives")
        self.assertEqual(rows[1]["TP"], 0)
        self.assertEqual(rows[1]["FP"], 1)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            summary = write_probability_reports(rows, directory, {"checkpoint_epoch": 30})
            self.assertEqual(summary["class_count"], 2)
            self.assertEqual(summary["threshold_observation_counts"],
                             {"all_scores_below_threshold": 1, "only_false_positives": 1})
            for name in ("no_prediction_probability_stats.csv", "no_prediction_probability_top20.csv"):
                with (Path(directory) / name).open(newline="") as handle:
                    saved = list(csv.DictReader(handle))
                self.assertEqual(saved[0]["gt_negative_mean_probability"], "")
                self.assertEqual(saved[1]["class_name"], "Class, A")
                self.assertEqual(float(saved[1]["max_probability"]), rows[1]["max_probability"])
            saved_summary = json.loads((Path(directory) / "no_prediction_probability_summary.json").read_text())
            self.assertEqual(saved_summary, summary)


if __name__ == "__main__":
    unittest.main()
