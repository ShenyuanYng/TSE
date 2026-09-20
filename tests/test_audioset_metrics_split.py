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
        metric.reset()
        self.assertIsNone(metric.compute()["conditional_frame_accuracy"])

    def test_accuracy_threshold_empty_mask_and_nonfinite_predictions(self):
        pred = {"frame_probs": torch.tensor([[.5, .6]]),
                "frame_mask": torch.tensor([[True, True]])}
        target = {"strong_labels": torch.tensor([[0., 1.]]), "strong_mask": torch.ones(1, 2),
                  "utt_id": ["a"], "class_index": [0]}
        metric = ConditionalFrameAccuracy(.6)
        metric.update(pred, target)
        self.assertEqual(metric.compute()["conditional_frame_accuracy"], 1.)
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
        metric.rows = {("a", 0): (2, 3)}
        remote = {("a", 0): (2, 3), ("b", 1): (1, 2)}

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
