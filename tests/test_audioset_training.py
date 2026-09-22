"""Tiny real-audio fit/validation/checkpoint/test integration."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from pytorch_lightning import Trainer
from train import _checkpoint_callback
from resume_training import continue_run, completed_epochs
from source_separator import SourceSeparator
from utils.config import yaml_to_parser


class TrainingIntegrationTest(unittest.TestCase):
    def test_checkpoint_cannot_change_holdout_or_resume_legacy_split(self):
        hp = yaml_to_parser("exp_defs/audioset_params.yaml").parse_args([])
        model = SourceSeparator(hp)
        checkpoint = {"hyper_parameters": vars(hp).copy()}
        model.on_load_checkpoint(checkpoint)
        checkpoint["hyper_parameters"]["train_split_seed"] = 8
        with self.assertRaisesRegex(ValueError, "split setting changed: train_split_seed"):
            model.on_load_checkpoint(checkpoint)
        del checkpoint["hyper_parameters"]["train_split_path"]
        with self.assertRaisesRegex(ValueError, "Start a new experiment from scratch"):
            model.on_load_checkpoint(checkpoint)

    def test_all_partition_video_overlaps_are_rejected(self):
        hp = yaml_to_parser("exp_defs/audioset_params.yaml").parse_args([])
        for left, right in (("tr", "cv"), ("tr", "test"), ("cv", "test")):
            with self.subTest(left=left, right=right):
                model = SourceSeparator(hp)
                entries = {part: [{"id": f"{part}_0", "ytid": part}]
                           for part in ("tr", "cv", "test")}
                entries[right][0]["ytid"] = left
                with patch("source_separator.datapath_creator",
                           side_effect=lambda hp, part: entries[part]):
                    with self.assertRaisesRegex(ValueError, "share YouTube video IDs"):
                        model.setup()

    def test_fit_then_test_best_checkpoint(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for part, count in (("train", 2), ("eval", 6)):
                audio = root / "audio" / part
                audio.mkdir(parents=True)
                weak, strong = [], []
                for i in range(count):
                    clip = f"{part}{i}"
                    samples = 1600 + i * 128
                    wave = .1 * np.sin(2 * np.pi * 440 * np.arange(samples) / 16000)
                    sf.write(str(audio / f"{clip}_0.wav"), wave, 16000)
                    weak.append(f'{clip},0,{samples / 16000},"/m/a,/m/b"\n')
                    strong.extend([f"{clip}_0\t0.016\t0.048\t/m/a\n",
                                   f"{clip}_0\t0.064\t0.096\t/m/b\n"])
                (root / f"{part}.csv").write_text(''.join(weak))
                (root / f"{part}.tsv").write_text(''.join(strong))
            (root / "classes.csv").write_text("index,mid,display_name\n0,/m/a,A\n1,/m/b,B\n")
            hp = yaml_to_parser("exp_defs/audioset_params.yaml").parse_args([])
            hp.data_path = str(root)
            hp.class_map_path = "classes.csv"
            hp.train_weak_manifest, hp.test_weak_manifest = ["train.csv"], ["eval.csv"]
            hp.train_strong_manifest, hp.test_strong_manifest = "train.tsv", "eval.tsv"
            hp.num_classes, hp.num_workers, hp.batch_size = 2, 0, 2
            hp.audioset_conv_channels = [4]
            hp.audioset_temporal_hidden, hp.audioset_rnn_layers = 4, 1
            hp.audioset_embedding_dim = 3
            hp.frame_accuracy_threshold = 0.7
            hp.epochs = 2
            hp.save_path = str(root / "results/version_first")
            model = SourceSeparator(hp)
            checkpoint = _checkpoint_callback(hp.save_path)
            trainer = Trainer(accelerator="cpu", devices=1, max_epochs=1,
                              logger=False, callbacks=[checkpoint], enable_progress_bar=False,
                              enable_model_summary=False, num_sanity_val_steps=1)
            trainer.fit(model)
            validation_metrics = json.loads(
                (root / "results/version_first/val_frame_accuracy.json").read_text())
            for name in ("accuracy", "precision", "recall", "f1"):
                key = f"val_frame_{name}"
                self.assertIn(key, trainer.callback_metrics)
                self.assertAlmostEqual(trainer.callback_metrics[key].item(),
                                       validation_metrics[key], places=6)
            self.assertEqual(validation_metrics["val_frame_accuracy"],
                             validation_metrics["conditional_frame_accuracy"])
            self.assertFalse(model.val_accuracy.rows)
            self.assertTrue(checkpoint.best_model_path)
            self.assertTrue((root / "metadata/train_val_split.json").exists())
            self.assertEqual(len(model.split_paths["tr"]), 1)
            self.assertEqual(len(model.split_paths["cv"]), 1)
            self.assertEqual(len(model.split_paths["test"]), 6)
            self.assertTrue(all(e["ytid"].startswith("train") for e in model.split_paths["cv"]))
            self.assertEqual(len(model.val_dataloader().dataset), 2)
            self.assertEqual(len(model.test_dataloader().dataset), 12)
            self.assertFalse((root / "results/version_first/test_frame_accuracy.json").exists())
            trainer.test(model, ckpt_path="best")
            metrics = json.loads((root / "results/version_first/test_frame_accuracy.json").read_text())
            self.assertGreater(metrics["valid_frames"], 0)
            self.assertTrue(0 <= metrics["conditional_frame_accuracy"] <= 1)
            self.assertEqual(metrics["conditional_frame_accuracy"],
                             metrics["correct_frames"] / metrics["valid_frames"])
            self.assertEqual(metrics["threshold"], 0.7)
            self.assertIn("test_conditional_frame_accuracy", trainer.callback_metrics)
            self.assertFalse(any("mAP" in key for key in trainer.callback_metrics))
            self.assertFalse(list(root.rglob("*_frame_ap.json")))
            saved = torch.load(root / "results/mylast.ckpt", weights_only=False)
            self.assertEqual(completed_epochs(saved), 1)
            self.assertEqual(saved["hyper_parameters"]["audioset_embedding_dim"], 3)
            first_step = saved["global_step"]
            self.assertTrue(saved["optimizer_states"][0]["state"])
            # The sidecar is deliberately absent: the ckpt is authoritative.
            (root / "results/mylast_best_info.pt").unlink()
            continue_run(root / "results")
            restored = torch.load(root / "results/mylast.ckpt", weights_only=False)
            self.assertEqual(completed_epochs(restored), 2)
            self.assertEqual(restored["global_step"], first_step * 2)
            before = next(iter(saved["optimizer_states"][0]["state"].values()))["step"]
            after = next(iter(restored["optimizer_states"][0]["state"].values()))["step"]
            self.assertEqual(after.item(), before.item() * 2)
            self.assertTrue((root / "results/training_complete.json").is_file())
            # A third queued job must not perform any more optimization.
            continue_run(root / "results")
            final = torch.load(root / "results/mylast.ckpt", weights_only=False)
            self.assertEqual(final["global_step"], restored["global_step"])



if __name__ == "__main__":
    unittest.main()
