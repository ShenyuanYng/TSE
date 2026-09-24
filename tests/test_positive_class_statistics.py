"""Strong-positive filtering and zero-F1 diagnosis need no model inference."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from summarize_positive_classes import FIELDS, positive_class_statistics, write_positive_class_reports


class PositiveClassStatisticsTest(unittest.TestCase):
    def examples(self):
        return [
            {"mid": "good", "class_name": "Good, 正例", "tp": 2, "fp": 1, "fn": 2},
            {"mid": "silent", "class_name": "No prediction", "tp": 0, "fp": 0, "fn": 8},
            {"mid": "wrong", "class_name": "Wrong predictions", "tp": 0, "fp": 4, "fn": 3},
            # Neither negative-only case belongs in the Strong-positive zero-F1 groups.
            {"mid": "negative", "class_name": "Negative only", "tp": 0, "fp": 5, "fn": 0},
            {"mid": "empty", "class_name": "No positives", "tp": 0, "fp": 0, "fn": 0},
            # A rounded upstream F1 must not turn a TP-positive class into zero F1.
            {"mid": "tiny", "class_name": "Tiny F1", "tp": 1, "fp": 0, "fn": 100000000, "f1": 0.0},
        ]

    def test_filters_and_partitions_using_exact_counts(self):
        positive, zero, silent, wrong, summary = positive_class_statistics(self.examples())
        self.assertEqual([r["MID"] for r in positive], ["good", "silent", "wrong", "tiny"])
        self.assertEqual([r["MID"] for r in zero], ["silent", "wrong"])
        self.assertEqual([r["MID"] for r in silent], ["silent"])
        self.assertEqual([r["MID"] for r in wrong], ["wrong"])
        self.assertEqual(positive[0]["positive_frames"], 4)
        self.assertEqual(positive[0]["predicted_positive_frames"], 3)
        self.assertAlmostEqual(positive[0]["Precision"], 2 / 3)
        self.assertEqual(positive[0]["Recall"], .5)
        self.assertAlmostEqual(positive[0]["F1"], 4 / 7)
        self.assertEqual(silent[0]["Precision"], 0)
        self.assertEqual(wrong[0]["predicted_positive_frames"], 4)
        self.assertEqual(summary["positive_class_count"], 4)
        self.assertEqual(summary["zero_f1_class_count"], 2)
        self.assertEqual(summary["excluded_no_strong_positive_class_count"], 2)

    def test_csv_schema_quoting_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = write_positive_class_reports(self.examples(), directory, {"threshold": .7})
            root = Path(directory)
            for filename, count in (("strong_positive_class_metrics.csv", 4),
                                    ("zero_f1_classes.csv", 2),
                                    ("zero_f1_no_predictions.csv", 1),
                                    ("zero_f1_predicted_but_tp_zero.csv", 1)):
                with (root / filename).open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    expected = [*FIELDS, "zero_f1_reason"] if filename == "zero_f1_classes.csv" else list(FIELDS)
                    self.assertEqual(reader.fieldnames, expected)
                    rows = list(reader)
                    self.assertEqual(len(rows), count)
                    if filename == "strong_positive_class_metrics.csv":
                        self.assertEqual(rows[0]["class_name"], "Good, 正例")
            self.assertEqual(json.loads((root / "positive_class_summary.json").read_text()), summary)
            self.assertEqual(summary["provenance"], {"threshold": .7})

    def test_empty_positive_set_and_invalid_inputs(self):
        self.assertIsNone(positive_class_statistics([])[-1]["strong_positive_macro_f1"])
        with tempfile.TemporaryDirectory() as directory:
            summary = write_positive_class_reports(self.examples()[3:5], directory)
            self.assertEqual(summary["positive_class_count"], 0)
            for path in Path(directory).glob("*.csv"):
                with path.open(newline="") as handle:
                    self.assertEqual(list(csv.DictReader(handle)), [])
        with self.assertRaisesRegex(ValueError, "duplicate MID"):
            positive_class_statistics(self.examples() * 2)
        for invalid in (-1, "1.5"):
            with self.assertRaises(ValueError):
                positive_class_statistics([{**self.examples()[0], "tp": invalid}])

    def test_offline_cli_preserves_source_and_does_not_import_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "per_class_metrics.csv"
            source.write_text('mid,class_name,tp,fp,fn\nA,Target,0,2,4\nB,Negative,0,0,0\n')
            original = source.read_bytes()
            script = Path(__file__).resolve().parents[1] / "summarize_positive_classes.py"
            result = subprocess.run([sys.executable, "-S", str(script), "--input-csv", str(source),
                                     "--output-dir", str(root / "results")],
                                    capture_output=True, text=True, check=True)
            self.assertIn("Strong-positive classes=1", result.stdout)
            self.assertEqual(source.read_bytes(), original)
            summary = json.loads((root / "results/positive_class_summary.json").read_text())
            self.assertEqual(summary["predicted_but_tp_zero_class_count"], 1)
            self.assertEqual(summary["provenance"]["source_csv"], str(source.resolve()))


if __name__ == "__main__":
    unittest.main()
