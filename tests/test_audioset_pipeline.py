import os
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from datasets.audioset_logmel_dataset import AudioSetLogMelDataset
from datasets.paths.audioset_paths import get_audioset_paths
from loss_functions.audioset_loss import audioset_strong_loss
from nets.audioset_crnn import AudioSetCRNN
from utils.collate import collate_seq


class AudioSetModel1Test(unittest.TestCase):
    def test_embedding_controls_predictions_and_receives_gradients(self):
        torch.manual_seed(7)
        model = AudioSetCRNN(num_classes=3, conv_channels=[4], temporal_hidden=4,
                             rnn_layers=1, dropout=0.0, embedding_dim=3).eval()
        audio = torch.randn(1, 7, 16).expand(2, -1, -1).clone()
        inputs = {"logmel": (audio, torch.tensor([7, 7])),
                  "weak_labels": torch.tensor([[1., 0., 0.], [0., 1., 0.]])}
        predictions = model(inputs)
        self.assertEqual(predictions["frame_logits"].shape, (2, 7))
        self.assertFalse(torch.allclose(*predictions["frame_logits"]))
        targets = {"strong_labels": torch.tensor([[0., 1., 1., 0., 0., 1., 0.]] * 2),
                   "strong_mask": torch.ones(2, 7)}
        audioset_strong_loss(predictions, targets)["loss"].backward()
        grad = model.class_embedding.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue((grad[:2].abs().sum(dim=1) > 0).all())
        self.assertEqual(grad[2].abs().sum().item(), 0)
        # Identical embeddings must give identical predictions for the same audio.
        with torch.no_grad():
            model.class_embedding.weight[1].copy_(model.class_embedding.weight[0])
        logits = model(inputs)["frame_logits"]
        torch.testing.assert_close(logits[0], logits[1])
        # Concatenation preserves audio information even with a zero embedding.
        with torch.no_grad():
            model.class_embedding.weight[0].zero_()
        logits = model(inputs)["frame_logits"][0]
        self.assertFalse(torch.allclose(logits, logits[0].expand_as(logits)))

    def test_embedding_dimension_is_independent_of_temporal_width(self):
        for dim in (1, 3, 128):
            with self.subTest(dim=dim):
                model = AudioSetCRNN(num_classes=3, conv_channels=[4], temporal_hidden=4,
                                     rnn_layers=1, dropout=0.0, embedding_dim=dim).eval()
                captured = []
                hook = model.fusion[0].register_forward_pre_hook(
                    lambda module, args: captured.append(args[0].detach()))
                condition = torch.tensor([[0., 1., 0.]])
                result = model({"logmel": torch.randn(1, 5, 16), "weak_labels": condition})
                hook.remove()
                self.assertEqual(model.class_embedding.weight.shape, (3, dim))
                self.assertEqual(captured[0].shape, (1, 5, 8 + dim))
                expected = (condition @ model.class_embedding.weight).unsqueeze(1).expand(-1, 5, -1)
                torch.testing.assert_close(captured[0][..., 8:], expected)
                self.assertEqual(result["frame_logits"].shape, (1, 5))
        with self.assertRaisesRegex(ValueError, "embedding_dim"):
            AudioSetCRNN(embedding_dim=0)

    def test_default_half_overlap_keeps_strong_labels_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = os.path.join(directory, "ten_seconds.wav")
            sf.write(audio_path, np.zeros(160000, dtype=np.float32), 16000)
            class_map = os.path.join(directory, "classes.csv")
            with open(class_map, "w") as handle:
                handle.write("index,mid,display_name\n0,/m/a,A\n")
            entries = [{"audio": audio_path, "id": "clip", "weak_mids": ["/m/a"],
                        "events": [{"mid": "/m/a", "onset": 2., "offset": 4.}]}]
            dataset = AudioSetLogMelDataset(entries, class_map, num_classes=1, training=False)
            inputs, target = dataset[0]
            self.assertEqual(dataset.hop_length * 2, dataset.window_length)
            self.assertEqual(inputs["logmel"].shape, (626, 64))
            np.testing.assert_allclose(target["frame_times"][:3], [0, .016, .032])
            np.testing.assert_array_equal(np.flatnonzero(target["strong_labels"]),
                                          np.arange(125, 250))
            self.assertEqual(target["frame_times"][-1], 10.)
            self.assertEqual(target["strong_mask"][-1], 0)
            self.assertEqual(target["strong_mask"].sum(), 625)

    def test_paired_manifests_select_the_correct_audio_split(self):
        with tempfile.TemporaryDirectory() as root:
            balanced_audio = os.path.join(root, "balance_wav")
            unbalanced_audio = os.path.join(root, "unbalanced_wav")
            os.makedirs(balanced_audio)
            os.makedirs(unbalanced_audio)
            sf.write(os.path.join(balanced_audio, "shared_id.wav"), np.zeros(8000), 8000)
            sf.write(os.path.join(unbalanced_audio, "shared_id.wav"), np.ones(8000), 8000)

            balanced_manifest = os.path.join(root, "balanced.csv")
            with open(balanced_manifest, "w", encoding="utf-8") as handle:
                handle.write('shared_id, 0.000, 1.000, "/m/a"\n')
            unbalanced_manifest = os.path.join(root, "unbalanced.csv")
            with open(unbalanced_manifest, "w", encoding="utf-8") as handle:
                handle.write('shared_id, 10.000, 11.000, "/m/a"\n')
            strong_manifest = os.path.join(root, "strong.tsv")
            with open(strong_manifest, "w", encoding="utf-8") as handle:
                handle.write("shared_id_0\t0.100\t0.500\t/m/a\n")
                handle.write("shared_id_10000\t0.200\t0.600\t/m/a\n")

            paths = get_audioset_paths(
                data_root=root,
                audio_dir=[balanced_audio, unbalanced_audio],
                weak_label_path=[balanced_manifest, unbalanced_manifest],
                strong_label_path=strong_manifest,
            )

            selected = {entry["id"]: entry["audio"] for entry in paths}
            self.assertEqual(selected["shared_id_0"], os.path.join(balanced_audio, "shared_id.wav"))
            self.assertEqual(
                selected["shared_id_10000"],
                os.path.join(unbalanced_audio, "shared_id.wav"),
            )

    def test_audio_and_weak_label_input_strong_loss_backward(self):
        with tempfile.TemporaryDirectory() as root:
            audio_dir = os.path.join(root, "audio")
            os.makedirs(audio_dir)
            samplerate = 8000
            time = np.arange(samplerate, dtype=np.float32) / samplerate
            waveform = 0.1 * np.sin(2 * np.pi * 440 * time)
            sf.write(os.path.join(audio_dir, "abc_0.wav"), waveform, samplerate)

            class_map = os.path.join(root, "class_labels_indices.csv")
            with open(class_map, "w", encoding="utf-8") as handle:
                handle.write("index,mid,display_name\n")
                handle.write("0,/m/a,Event A\n1,/m/b,Event B\n2,/m/c,Event C\n")
            weak_manifest = os.path.join(root, "weak.csv")
            with open(weak_manifest, "w", encoding="utf-8") as handle:
                handle.write("# YTID, start_seconds, end_seconds, positive_labels\n")
                handle.write('abc, 0.000, 1.000, "/m/a,/m/b,/m/c"\n')
            strong_manifest = os.path.join(root, "strong.tsv")
            with open(strong_manifest, "w", encoding="utf-8") as handle:
                handle.write("abc_0\t0.200\t0.600\t/m/a\n")

            paths = get_audioset_paths(
                data_root=root,
                audio_dir=audio_dir,
                weak_label_path=weak_manifest,
                strong_label_path=strong_manifest,
            )
            dataset = AudioSetLogMelDataset(
                paths,
                class_map_path=class_map,
                num_classes=3,
                sr=samplerate,
                rms_norm=0,
                window_length=256,
                hop_length=128,
                n_mels=32,
                training=False,
            )
            self.assertEqual(len(dataset), 3)
            for index in range(3):
                sample, truth = dataset[index]
                self.assertEqual(sample["weak_labels"].sum(), 1)
                self.assertEqual(sample["weak_labels"].argmax(), index)
                np.testing.assert_array_equal(sample["logmel"], dataset[0][0]["logmel"])
                if index > 0:
                    self.assertEqual(truth["strong_labels"].sum(), 0)
            model_input, target = dataset[0]
            np.testing.assert_allclose(target["frame_times"][:3], [0, .016, .032])
            self.assertEqual(target["strong_mask"][-1], 1)
            self.assertTrue(np.isfinite(model_input["logmel"]).all())
            self.assertEqual(model_input["logmel"].shape[1:], (32,))
            self.assertEqual(model_input["weak_labels"].tolist(), [1.0, 0.0, 0.0])
            self.assertNotIn("weak_labels", target)
            self.assertGreater(target["strong_labels"].sum(), 0)

            model_inputs, targets = collate_seq([dataset[0], dataset[0]])
            model = AudioSetCRNN(
                num_classes=3,
                conv_channels=[4, 8],
                temporal_hidden=4,
                rnn_layers=1,
                dropout=0.0,
            )
            predictions = model(model_inputs)
            self.assertEqual(
                predictions["frame_logits"].shape[:2],
                model_inputs["logmel"][0].shape[:2],
            )
            self.assertEqual(predictions["frame_logits"].ndim, 2)
            self.assertNotIn("clip_logits", predictions)

            losses = audioset_strong_loss(predictions, targets)
            self.assertTrue(torch.isfinite(losses["loss"]))
            losses["loss"].backward()
            self.assertTrue(
                any(parameter.grad is not None for parameter in model.fusion.parameters())
            )


if __name__ == "__main__":
    unittest.main()
