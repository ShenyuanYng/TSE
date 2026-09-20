"""Strong Train video holdout, complete Strong Eval, and persisted boundaries."""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from creators.dataset import datapath_creator
from datasets.train_split import split_train, video_id, validate_split_config


class TrainSplitTest(unittest.TestCase):
    def setUp(self):
        # Multiple, uneven numbers of clips per video must never cross splits.
        self.entries = [{"id": f"video{i}_{j}", "ytid": f"video{i}",
                         "weak_mids": ["A", "B"]}
                        for i in range(20) for j in range(1 + i % 3)]

    def test_ratio_grouping_persistence_and_order_independence(self):
        with tempfile.TemporaryDirectory() as root:
            first = split_train(self.entries, root, "split.json")
            reversed_entries = list(reversed(self.entries))
            reused = split_train(reversed_entries, root, "split.json")
            fresh = split_train(reversed_entries, root, "fresh.json")
            train = {video_id(e) for e in first["tr"]}
            val = {video_id(e) for e in first["cv"]}
            self.assertEqual((len(train), len(val)), (17, 3))
            self.assertFalse(train & val)
            self.assertEqual(sum(map(len, first.values())), len(self.entries))
            for part in ("tr", "cv"):
                expected = {e["id"] for e in first[part]}
                self.assertEqual(expected, {e["id"] for e in reused[part]})
                self.assertEqual(expected, {e["id"] for e in fresh[part]})

    def test_inventory_settings_and_legacy_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            split_train(self.entries, root, "split.json")
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                split_train(self.entries[:-1], root, "split.json")
            for options in ({"seed": 8}, {"fraction": .2}):
                with self.assertRaisesRegex(ValueError, "settings changed"):
                    split_train(self.entries, root, "split.json", **options)
            Path(root, "legacy.json").write_text(json.dumps(
                {"seed": 7, "test_fraction": .5, "cv": [], "test": []}))
            with self.assertRaisesRegex(ValueError, "settings changed"):
                split_train(self.entries, root, "legacy.json")

    def test_invalid_input_and_tiny_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            for fraction in (0, 1, -.1, float("nan")):
                with self.assertRaisesRegex(ValueError, "fraction"):
                    split_train(self.entries, root, "split.json", fraction=fraction)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                split_train(self.entries * 2, root, "split.json")
            with self.assertRaisesRegex(ValueError, "two independent videos"):
                split_train(self.entries[:1], root, "split.json")
            tiny = split_train([self.entries[0], self.entries[1]], root, "split.json")
            self.assertEqual({k: len(v) for k, v in tiny.items()}, {"tr": 1, "cv": 1})

    def test_saved_video_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            split_train(self.entries, root, "split.json")
            path = Path(root, "split.json")
            saved = json.loads(path.read_text())
            # Keep the full inventory but place clips from video1 on both sides.
            saved["cv"] = ["video1_0"]
            saved["tr"] = [e["id"] for e in self.entries if e["id"] != "video1_0"]
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "both training and validation"):
                split_train(self.entries, root, "split.json")

    def test_creators_use_train_for_validation_and_all_eval_for_test(self):
        with tempfile.TemporaryDirectory() as root:
            hp = Namespace(dataset="audioset_strong", data_path=root,
                           train_audio_dir=["train_audio"], test_audio_dir=["eval_audio"],
                           train_weak_manifest=["train.csv"], test_weak_manifest=["eval.csv"],
                           train_strong_manifest="train.tsv", test_strong_manifest="eval.tsv",
                           audioset_strict_paths=True, train_val_fraction=.15,
                           train_split_seed=7, train_split_path="train_val_split.json")
            # Even a single eval video is retained whole; no eval split is required.
            evaluation = [{"id": "eval_0", "ytid": "eval", "weak_mids": ["A"]}]

            def paths(**kwargs):
                is_train = kwargs["strong_label_path"] == "train.tsv"
                self.assertEqual(kwargs["audio_dir"], ["train_audio" if is_train else "eval_audio"])
                self.assertEqual(kwargs["weak_label_path"], ["train.csv" if is_train else "eval.csv"])
                return self.entries if is_train else evaluation

            with patch("datasets.paths.audioset_paths.get_audioset_paths", side_effect=paths):
                # Test retrieval must not generate or depend on a holdout file.
                self.assertEqual(datapath_creator(hp, "test"), evaluation)
                self.assertFalse(Path(root, hp.train_split_path).exists())
                train = datapath_creator(hp, "tr")
                val = datapath_creator(hp, "cv")
            self.assertEqual({e["id"] for e in train + val}, {e["id"] for e in self.entries})
            self.assertEqual(len({video_id(e) for e in val}), 3)
            self.assertFalse(Path(root, "metadata/eval_split.json").exists())

    def test_old_configuration_requires_fresh_experiment(self):
        with self.assertRaisesRegex(ValueError, "Start a new experiment from scratch"):
            validate_split_config(Namespace(eval_split_path="eval_split.json"))


if __name__ == "__main__":
    unittest.main()
