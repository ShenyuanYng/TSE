"""Threshold sweeps reuse predictions and preserve saved validation membership."""
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch

from creators.task import AudioSetStrongTask
from datasets.train_split import split_train
from source_separator import SourceSeparator
from sweep_validation_thresholds import DEFAULT_THRESHOLDS, best_checkpoint, run, sweep_batches
from utils.audioset_metrics import ConditionalFrameAccuracy
from utils.config import yaml_to_parser


class FixedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, model_input, target_dict, current_epoch):
        assert not self.training and torch.is_inference_mode_enabled()
        self.calls += 1
        return model_input


class ThresholdSweepTest(unittest.TestCase):
    def test_single_forward_per_batch_shared_probabilities_masks_and_dedup(self):
        model = FixedModel()
        module = SimpleNamespace(model=model, task=AudioSetStrongTask(), eval=model.eval)
        first = (
            {"frame_probs": torch.tensor([[.5, .8, .99], [.1, .9, .1]]),
             "frame_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)},
            {"strong_labels": (torch.tensor([[1., 0, 0], [1., 0, 0]]), [3, 3]),
             "strong_mask": (torch.tensor([[1., 1, 1], [1., 1, 0]]), [3, 3]),
             "utt_id": ["a", "b"], "class_index": [0, 1]},
        )
        second = (
            {"frame_probs": torch.tensor([[.2]]), "frame_mask": torch.tensor([[True]])},
            {"strong_labels": torch.tensor([[0.]]), "strong_mask": torch.ones(1, 1),
             "utt_id": ["c"], "class_index": [2]},
        )
        seen = []
        update = ConditionalFrameAccuracy.update

        def record(metric, predictions, targets):
            seen.append(predictions["frame_probs"])
            update(metric, predictions, targets)

        with patch.object(ConditionalFrameAccuracy, "update", record):
            rows, batches = sweep_batches(module, [first, first, second], "cpu", log_every=0)
        self.assertEqual((model.calls, batches, len(rows)), (3, 3, 7))
        for i, batch in enumerate((first, first, second)):
            self.assertTrue(all(probs is batch[0]["frame_probs"] for probs in seen[i * 7:(i + 1) * 7]))
        self.assertEqual([row["threshold"] for row in rows], list(DEFAULT_THRESHOLDS))
        self.assertTrue(all(row["valid_frames"] == 5 for row in rows))
        at_half = rows[2]
        self.assertEqual([at_half[key] for key in ("tp", "fp", "fn", "tn")], [1, 2, 1, 1])
        self.assertAlmostEqual(at_half["precision"], 1 / 3)
        self.assertEqual(at_half["recall"], .5)
        self.assertEqual(at_half["f1"], .4)  # Batch-F1 mean would be different.
        self.assertEqual(at_half["accuracy"], .4)
        self.assertEqual([row["accuracy"] for row in rows], [.4, .4, .4, .2, .2, .2, .4])
        self.assertTrue(all(row["f1"] == 0 for row in rows[3:]))
        # Reusing the sweep function cannot carry counts into the next invocation.
        fresh, _ = sweep_batches(module, [second], "cpu", log_every=0)
        self.assertTrue(all(row["valid_frames"] == 1 and row["tn"] == 1 for row in fresh))

    def test_best_selection_uses_saved_callback_and_supports_moved_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            best = root / "version_37074/_ckpt_epoch_12.ckpt"
            best.parent.mkdir()
            best.touch()
            torch.save({"callbacks": {
                "ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min'}": {
                    "best_model_path": "/original/run/version_37074/_ckpt_epoch_12.ckpt"},
                "ModelCheckpoint{'monitor': 'other_metric'}": {"best_model_path": "wrong.ckpt"},
            }}, root / "mylast.ckpt")
            self.assertEqual(best_checkpoint(root), best.resolve())
            best.unlink()
            with self.assertRaises(FileNotFoundError):
                best_checkpoint(root)

    def test_real_audio_checkpoint_reports_and_rejects_missing_or_changed_split(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "train"
            audio.mkdir()
            entries, weak, strong = [], [], []
            for index in range(4):
                clip = f"clip{index}"
                sf.write(str(audio / f"{clip}_0.wav"), np.full(1600 + index * 128, .1), 16000)
                entries.append({"id": f"{clip}_0", "ytid": clip})
                weak.append(f'{clip},0,0.2,"A"\n')
                strong.append(f"{clip}_0\t0.032\t0.064\tA\n")
            (root / "weak.csv").write_text("".join(weak))
            (root / "strong.tsv").write_text("".join(strong))
            (root / "classes.csv").write_text("index,mid,display_name\n0,A,Target\n")
            hp = yaml_to_parser("exp_defs/audioset_params.yaml").parse_args([])
            hp.data_path, hp.class_map_path = str(root), "classes.csv"
            hp.train_audio_dir, hp.train_weak_manifest = [str(audio)], ["weak.csv"]
            hp.train_strong_manifest, hp.train_val_fraction = "strong.tsv", .5
            hp.num_classes, hp.audioset_conv_channels = 1, [4]
            hp.audioset_temporal_hidden, hp.audioset_rnn_layers = 4, 1
            hp.audioset_embedding_dim, hp.batch_size, hp.num_workers = 3, 1, 0
            split_train(entries, root, hp.train_split_path, hp.train_val_fraction, hp.train_split_seed)
            split = root / hp.train_split_path
            original_split = split.read_bytes()
            model = SourceSeparator(hp)
            checkpoint = root / "synthetic.ckpt"
            torch.save({"epoch": 12, "hyper_parameters": vars(hp),
                        "state_dict": model.state_dict()}, checkpoint)
            original_checkpoint = checkpoint.read_bytes()
            loss = Mock(side_effect=AssertionError("Sweep must not call loss"))
            output = io.StringIO()
            with patch("source_separator.loss_creator", return_value=loss), \
                 patch.object(SourceSeparator, "configure_optimizers",
                              side_effect=AssertionError("Sweep must not train")), \
                 redirect_stdout(output):
                report = run(checkpoint, root / "reports", device="cpu")
            loss.assert_not_called()
            self.assertEqual(report["checkpoint_epoch"], 12)
            self.assertEqual(report["validation_clips"], 2)
            self.assertEqual(report["validation_samples"], 2)
            self.assertEqual(report["prediction_batches"], 2)
            self.assertEqual(json.loads((root / "reports/threshold_sweep.json").read_text()), report)
            with (root / "reports/threshold_sweep.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 7)
            self.assertIn("Precision", output.getvalue())
            self.assertIn("0.90", output.getvalue())
            self.assertEqual(split.read_bytes(), original_split)
            self.assertEqual(checkpoint.read_bytes(), original_checkpoint)
            self.assertEqual(report["partition"], "cv")
            results = report["results"]
            self.assertGreater(results[0]["valid_frames"], 0)
            for row in results:
                self.assertEqual(row["valid_frames"], sum(row[key] for key in ("tp", "fp", "fn", "tn")))
                self.assertEqual(row["valid_frames"], results[0]["valid_frames"])
                self.assertEqual(row["tp"] + row["fn"], results[0]["tp"] + results[0]["fn"])
            self.assertEqual([row["recall"] for row in results],
                             sorted((row["recall"] for row in results), reverse=True))
            # Existing inventory validation must refuse a different validation set.
            saved = json.loads(split.read_text())
            saved["cv"].pop()
            split.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                run(checkpoint, root / "changed", device="cpu")
            split.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Saved validation split"):
                run(checkpoint, root / "missing", device="cpu")
            self.assertFalse(split.exists())


if __name__ == "__main__":
    unittest.main()
