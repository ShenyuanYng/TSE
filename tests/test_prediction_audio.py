"""Verify exported audio keeps original timing, samples, and channels."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from datasets.train_split import split_train
from inspect_prediction import _export_audio_comparison, _intervals
from inspect_matched_predictions import run
from source_separator import SourceSeparator
from utils.config import yaml_to_parser


class PredictionAudioTest(unittest.TestCase):
    def test_stereo_alignment_and_raw_annotation_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Nonzero stereo signal: an accidental shift, normalization, or mixdown is visible.
            wave = np.column_stack((np.linspace(.1, .9, 1003), np.linspace(-.8, -.2, 1003))).astype("float32")
            source = root / "source.wav"
            sf.write(source, wave, 1000, subtype="FLOAT")
            events = [{"mid": "A", "onset": .1003, "offset": .5002},
                      {"mid": "A", "onset": .3, "offset": .4, "present": False},
                      {"mid": "B", "onset": 0., "offset": 1.003}]
            urls = _export_audio_comparison(root / "cv_004_0.78.html", source,
                                            [(0., float(np.float32(.032))),
                                             (float(np.float32(.992)), 1.003)], events, "A")
            prediction_mask = np.zeros(1003, dtype=bool)
            prediction_mask[:32] = True
            prediction_mask[992:] = True
            truth_mask = np.zeros(1003, dtype=bool)
            truth_mask[101:501] = True
            truth_mask[300:400] = False
            for role, expected in (("original", wave),
                                   ("predicted", wave * prediction_mask[:, None]),
                                   ("ground_truth", wave * truth_mask[:, None])):
                path = root / urls[role]
                restored, sr = sf.read(path, dtype="float32", always_2d=True)
                self.assertEqual(sr, 1000)
                self.assertEqual(restored.shape, wave.shape)
                np.testing.assert_array_equal(restored, expected)
                self.assertEqual(path.name, f"cv_004_0.78_{role}.wav")

    def test_empty_predictions_keep_length_and_final_interval_is_clipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            sf.write(source, np.full(1003, .25), 1000, subtype="FLOAT")
            urls = _export_audio_comparison(root / "report.html", source, [], [], "A")
            for role in ("predicted", "ground_truth"):
                audio, sr = sf.read(root / urls[role])
                self.assertEqual((len(audio), sr), (1003, 1000))
                self.assertFalse(audio.any())
            intervals = _intervals(np.array([False, True]), np.array([.976, .992]), .016, 1.003)
            self.assertEqual(intervals, [(.992, 1.003)])

    def test_batch_export_and_html_links_with_synthetic_checkpoint(self):
        # A tiny random model tests file generation only, not prediction quality.
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir = root / "train"
            audio_dir.mkdir()
            entries = []
            for clip in ("first", "second"):
                sf.write(audio_dir / f"{clip}_0.wav", np.full(1600, .1), 16000)
                entries.append({"id": f"{clip}_0", "ytid": clip})
            (root / "weak.csv").write_text('first,0,0.1,"A"\nsecond,0,0.1,"A"\n')
            (root / "strong.tsv").write_text("first_0\t0.032\t0.064\tA\nsecond_0\t0.032\t0.064\tA\n")
            (root / "classes.csv").write_text("index,mid,display_name\n0,A,Target\n")
            hp = yaml_to_parser("exp_defs/audioset_params.yaml").parse_args([])
            hp.data_path, hp.class_map_path = str(root), "classes.csv"
            hp.train_audio_dir, hp.train_weak_manifest = [str(audio_dir)], ["weak.csv"]
            hp.train_strong_manifest = "strong.tsv"
            hp.num_classes, hp.audioset_conv_channels = 1, [4]
            hp.audioset_temporal_hidden, hp.audioset_rnn_layers = 4, 1
            hp.audioset_embedding_dim = 3
            split_train(entries, root, hp.train_split_path)
            model = SourceSeparator(hp)
            checkpoint = root / "synthetic.ckpt"
            torch.save({"hyper_parameters": vars(hp), "state_dict": model.state_dict()}, checkpoint)
            output = root / "reports"
            summary = run(checkpoint, output, num_samples=1)
            self.assertEqual(len(summary["samples"]), 2)
            self.assertEqual(len(list((output / "audio").glob("*.wav"))), 6)
            self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
            for row in summary["samples"]:
                self.assertTrue(row["report"].endswith(f"_{row['Accuracy']:.2f}.html"))
                document = (output / row["report"]).read_text()
                for url in row["audio_files"].values():
                    self.assertIn(url, document)
                    self.assertTrue((output / url).is_file())
                    info = sf.info(output / url)
                    self.assertEqual((info.frames, info.samplerate, info.channels), (1600, 16000, 1))


if __name__ == "__main__":
    unittest.main()
