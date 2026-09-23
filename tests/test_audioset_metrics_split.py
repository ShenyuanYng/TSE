import unittest
from unittest.mock import patch
import numpy as np
import torch
from utils.audioset_metrics import ConditionalFrameAccuracy
from loss_functions.audioset_loss import audioset_strong_loss


class MetricsAndSplitTest(unittest.TestCase):
    def test_accuracy_counts_valid_frames_and_deduplicates(self):
        metric = ConditionalFrameAccuracy()
        pred = {"frame_probs": torch.tensor([[.5, .8, .99], [.1, .9, .1]]),
                "frame_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)}
        target = {"strong_labels": (torch.tensor([[1., 0, 0], [1., 0, 0]]), [3, 3]),
                  "strong_mask": (torch.tensor([[1., 1, 1], [1., 1, 0]]), [3, 3]),
                  "utt_id": ["a", "b"], "class_index": [0, 1]}
        metric.update(pred, target)
        metric.update(pred, target)
        # Include an all-negative class with only one valid frame. It must count
        # in accuracy; taking a class/batch mean would produce the wrong result.
        metric.update({"frame_probs": torch.tensor([[.2]]),
                       "frame_mask": torch.tensor([[True]])},
                      {"strong_labels": torch.tensor([[0.]]), "strong_mask": torch.ones(1, 1),
                       "utt_id": ["c"], "class_index": [2]})
        result = metric.compute()
        self.assertEqual(len(metric.rows), 3)
        self.assertEqual(result["correct_frames"], 2)
        self.assertEqual(result["valid_frames"], 5)
        self.assertEqual(result["conditional_frame_accuracy"], .4)
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]),
                         (1, 2, 1, 1))
        self.assertAlmostEqual(result["conditional_frame_precision"], 1 / 3)
        self.assertEqual(result["conditional_frame_recall"], .5)
        # The batch F1 values are .4 and 0: averaging them would give .2.
        self.assertEqual(result["conditional_frame_f1"], .4)
        metric.reset()
        self.assertFalse(metric.rows)
        for name in ("accuracy", "precision", "recall", "f1"):
            self.assertIsNone(metric.compute()[f"conditional_frame_{name}"])
        # The next epoch must not include any previous counts.
        metric.update(pred, target)
        self.assertEqual(metric.compute()["valid_frames"], 4)
        self.assertEqual(metric.compute()["tn"], 0)

    def test_accuracy_threshold_empty_mask_and_nonfinite_predictions(self):
        pred = {"frame_probs": torch.tensor([[.5, .6]]),
                "frame_mask": torch.tensor([[True, True]])}
        target = {"strong_labels": torch.tensor([[0., 1.]]), "strong_mask": torch.ones(1, 2),
                  "utt_id": ["a"], "class_index": [0]}
        metric = ConditionalFrameAccuracy(.6)
        metric.update(pred, target)
        self.assertEqual(metric.compute()["conditional_frame_accuracy"], 1.)
        for name in ("precision", "recall", "f1"):
            self.assertEqual(metric.compute()[f"conditional_frame_{name}"], 1.)
        pred["frame_probs"][0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            metric.update(pred, target)
        target["strong_mask"].zero_()
        metric.update(pred, target)
        self.assertEqual(metric.compute()["valid_frames"], 0)
        self.assertIsNone(metric.compute()["conditional_frame_accuracy"])
        for threshold in (-.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ConditionalFrameAccuracy(threshold)

    def test_distributed_accuracy_deduplicates_before_counting(self):
        metric = ConditionalFrameAccuracy()
        metric.rows = {("a", 0): (1, 1, 0, 1)}
        remote = {("a", 0): (1, 1, 0, 1), ("b", 1): (0, 0, 1, 1)}

        def gather(parts, rows):
            parts[:] = [rows, remote]

        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=2), \
             patch("torch.distributed.all_gather_object", side_effect=gather):
            result = metric.compute()
        self.assertEqual(result["correct_frames"], 3)
        self.assertEqual(result["valid_frames"], 5)
        self.assertEqual(result["conditional_frame_accuracy"], .6)
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]),
                         (1, 1, 1, 2))
        for name in ("precision", "recall", "f1"):
            self.assertEqual(result[f"conditional_frame_{name}"], .5)

    def test_zero_denominators(self):
        for probabilities, labels in (([.1, .2], [0., 0.]),
                                      ([.1, .2], [1., 1.]),
                                      ([.8, .9], [0., 0.])):
            with self.subTest(probabilities=probabilities, labels=labels):
                metric = ConditionalFrameAccuracy()
                metric.update(
                    {"frame_probs": torch.tensor([probabilities]),
                     "frame_mask": torch.ones(1, 2, dtype=torch.bool)},
                    {"strong_labels": torch.tensor([labels]),
                     "strong_mask": torch.ones(1, 2), "utt_id": ["a"], "class_index": [0]},
                )
                result = metric.compute()
                for name in ("precision", "recall", "f1"):
                    self.assertEqual(result[f"conditional_frame_{name}"], 0.)

    def test_per_class_aggregation_macro_validity_and_reset(self):
        metric = ConditionalFrameAccuracy(.7)
        examples = [
            # One class spans batches; the threshold boundary .7 is positive.
            ("a", 0, [.7, .1, .9, .99, .99], [1., 1, 0, 0, 0], [1] * 5, [1, 1, 1, 0, 0]),
            ("b", 0, [.8, .8, .8, .1, .1], [1., 1, 1, 0, 0], [1] * 5, [1] * 5),
            # The same clip with another class is a distinct condition.
            ("a", 1, [.1, .1, .1, .1, .99], [1., 1, 1, 1, 0], [1, 1, 1, 1, 0], [1] * 5),
            ("c", 2, [.1] * 5, [0.] * 5, [1] * 5, [1] * 5),
            ("d", 3, [float("nan")] * 5, [1.] * 5, [0] * 5, [1] * 5),
        ]
        for clip, category, probs, labels, strong, frames in examples * 2:
            metric.update(
                {"frame_probs": torch.tensor([probs]), "frame_mask": torch.tensor([frames])},
                {"strong_labels": torch.tensor([labels]), "strong_mask": torch.tensor([strong]),
                 "utt_id": [clip], "class_index": [category]},
            )
        result = metric.compute(include_per_class=True)
        by_class = {row["class_index"]: row for row in result["per_class"]}
        self.assertEqual([by_class[0][key] for key in ("tp", "fp", "fn", "tn")], [4, 1, 1, 2])
        self.assertEqual(by_class[0]["precision"], .8)
        self.assertEqual(by_class[0]["recall"], .8)
        self.assertEqual(by_class[0]["f1"], .8)
        self.assertEqual(by_class[1]["fn"], 4)
        self.assertEqual(by_class[2]["tn"], 5)
        self.assertEqual(by_class[2]["f1"], 0.)  # Negative-only is still a valid class.
        self.assertIsNone(by_class[3]["f1"])
        self.assertEqual(result["valid_class_count"], 3)
        self.assertAlmostEqual(result["macro_f1"], .8 / 3)
        self.assertAlmostEqual(result["conditional_frame_f1"], 4 / 7)
        self.assertEqual(result["valid_frames"], 17)
        for key in ("tp", "fp", "fn", "tn", "valid_frames"):
            self.assertEqual(result[key], sum(row[key] for row in result["per_class"]))
        metric.reset()
        empty = metric.compute(include_per_class=True)
        self.assertEqual(empty["per_class"], [])
        self.assertEqual(empty["valid_class_count"], 0)
        self.assertIsNone(empty["macro_f1"])

    def test_distributed_macro_uses_global_deduplicated_classes(self):
        metric = ConditionalFrameAccuracy(.7)
        metric.rows = {("a", 0): (1, 1, 1, 0), ("c", 2): (0, 0, 0, 5),
                       ("d", 3): (0, 0, 0, 0)}
        remote = {("a", 0): (1, 1, 1, 0), ("b", 0): (3, 0, 0, 2),
                  ("a", 1): (0, 0, 4, 0)}

        def gather(parts, rows):
            parts[:] = [rows, remote]

        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=2), \
             patch("torch.distributed.all_gather_object", side_effect=gather) as mocked_gather:
            result = metric.compute(include_per_class=True)
        mocked_gather.assert_called_once()
        self.assertEqual(result["valid_frames"], 17)
        self.assertEqual(result["valid_class_count"], 3)
        self.assertAlmostEqual(result["macro_f1"], .8 / 3)
        self.assertAlmostEqual(result["conditional_frame_f1"], 4 / 7)
        self.assertEqual(result["per_class"][0]["tp"], 4)

    def test_loss_ignores_masked_positions(self):
        logits = torch.tensor([[0., 100., -100.]], requires_grad=True)
        pred = {"frame_logits": logits, "frame_mask": torch.tensor([[1, 1, 0]])}
        target = {"strong_labels": torch.tensor([[1., 0, 1]]),
                  "strong_mask": torch.tensor([[1., 0, 1]])}
        loss = audioset_strong_loss(pred, target)["loss"]
        self.assertAlmostEqual(loss.item(), np.log(2), places=6)
        loss.backward()
        self.assertEqual(logits.grad[0, 1:].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
