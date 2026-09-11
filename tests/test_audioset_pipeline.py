import os
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from datasets.audioset_ri_dataset import AudioSetRIDataset
from datasets.paths.audioset_paths import get_audioset_paths
from loss_functions.audioset_loss import audioset_strong_loss
from nets.audioset_crnn import AudioSetCRNN
from utils.collate import collate_seq


class AudioSetModel1Test(unittest.TestCase):
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
                handle.write('abc, 0.000, 1.000, "/m/a"\n')
            strong_manifest = os.path.join(root, "strong.tsv")
            with open(strong_manifest, "w", encoding="utf-8") as handle:
                handle.write("abc_0\t0.200\t0.600\t/m/a\n")

            paths = get_audioset_paths(
                data_root=root,
                audio_dir=audio_dir,
                weak_label_path=weak_manifest,
                strong_label_path=strong_manifest,
            )
            dataset = AudioSetRIDataset(
                paths,
                class_map_path=class_map,
                num_classes=3,
                sr=samplerate,
                rms_norm=0,
                window_length=256,
                hop_length=128,
            )
            model_input, target = dataset[0]
            self.assertEqual(model_input["mix_ri"].shape[1:], (1, 1, 2, 129))
            self.assertEqual(model_input["weak_labels"].tolist(), [1.0, 0.0, 0.0])
            self.assertNotIn("weak_labels", target)
            self.assertGreater(target["strong_labels"][:, 0].sum(), 0)

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
                model_inputs["mix_ri"][0].shape[:2],
            )
            self.assertEqual(predictions["frame_logits"].shape[-1], 3)
            self.assertNotIn("clip_logits", predictions)

            losses = audioset_strong_loss(predictions, targets)
            self.assertTrue(torch.isfinite(losses["loss"]))
            losses["loss"].backward()
            self.assertTrue(
                any(parameter.grad is not None for parameter in model.weak_encoder.parameters())
            )


if __name__ == "__main__":
    unittest.main()
