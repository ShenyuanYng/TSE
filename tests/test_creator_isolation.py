"""Dependency-light tests for the AudioSet creator boundaries."""

import sys
import types
import unittest
from argparse import Namespace
from unittest.mock import patch

from creators.dataset import dataset_creator
from creators.loss import loss_creator
from creators.model import model_creator
from creators.task import AudioSetStrongTask, task_creator


class CreatorIsolationTest(unittest.TestCase):
    def test_model_creator_reads_only_model_options(self):
        module = types.ModuleType("nets.audioset_crnn")

        class FakeAudioSetCRNN:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        module.AudioSetCRNN = FakeAudioSetCRNN
        hparams = Namespace(
            model_name="AudioSetCRNN",
            num_classes=3,
            audioset_conv_channels=[4, 8],
            audioset_temporal_hidden=4,
            audioset_embedding_dim=3,
            audioset_rnn_layers=1,
        )
        with patch.dict(sys.modules, {"nets.audioset_crnn": module}):
            model = model_creator(hparams)
        self.assertEqual(model.kwargs["num_classes"], 3)
        self.assertEqual(model.kwargs["embedding_dim"], 3)

    def test_dataset_creator_reads_only_dataset_options(self):
        dataset_module = types.ModuleType("datasets.audioset_logmel_dataset")
        path_module = types.ModuleType("datasets.paths.audioset_paths")

        class FakeAudioSetDataset:
            def __init__(self, path_list, **kwargs):
                self.path_list = path_list
                self.kwargs = kwargs

        dataset_module.AudioSetLogMelDataset = FakeAudioSetDataset
        path_module.resolve_path = lambda path, root: f"{root}/{path}"
        hparams = Namespace(
            dataset="audioset_strong",
            feature_type="logmel",
            data_path="data",
            class_map_path="classes.csv",
            num_classes=3,
            samplerate=8000,
            train_on_chunk=1.0,
            rms_norm=0,
            inwintime=0.032,
            hoptime=0.016,
            fft_size=0,
            window_type="hann",
            strong_annotation_complete=True,
        )
        modules = {
            "datasets.audioset_logmel_dataset": dataset_module,
            "datasets.paths.audioset_paths": path_module,
        }
        with patch.dict(sys.modules, modules):
            dataset = dataset_creator(
                hparams,
                "tr",
                path_list=[{"id": "example"}],
                is_training=True,
            )
        self.assertEqual(dataset.path_list[0]["id"], "example")
        self.assertEqual(dataset.kwargs["class_map_path"], "data/classes.csv")

    def test_loss_creator_reads_only_loss_options(self):
        module = types.ModuleType("loss_functions.audioset_loss")

        def fake_loss(*args, **kwargs):
            return args, kwargs

        module.audioset_strong_loss = fake_loss
        hparams = Namespace(
            loss_type="audioset_strong",
            class_positive_weight=2.0,
            balance_positive_negative=False,
        )
        with patch.dict(sys.modules, {"loss_functions.audioset_loss": module}):
            loss = loss_creator(hparams)
            _, kwargs = loss("prediction", "target", training=True)
        self.assertEqual(kwargs["class_positive_weight"], 2.0)
        self.assertFalse(kwargs["balance_positive_negative"])

    def test_task_creator_passes_the_input_dictionary(self):
        calls = []

        class RecordingModel:
            def __call__(self, model_input, target_dict, current_epoch):
                calls.append((model_input, target_dict, current_epoch))
                return "prediction"

        model = RecordingModel()
        target = {"strong_labels": "target"}
        inputs = {"logmel": "audio", "weak_labels": "weak"}
        audioset = task_creator(
            Namespace(task_type="audioset_strong", loss_type="audioset_strong")
        )
        self.assertIsInstance(audioset, AudioSetStrongTask)
        audioset.forward(model, inputs, target, 4)
        self.assertEqual(calls.pop(), (inputs, target, 4))

    def test_removed_tasks_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Unsupported task_type"):
            task_creator(Namespace(task_type="ri_inference", loss_type="ri_inference"))


if __name__ == "__main__":
    unittest.main()
