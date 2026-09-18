import unittest

from utils.config import yaml_to_parser


class ConfigParserTest(unittest.TestCase):
    def test_default_stft_has_half_overlap(self):
        hp = yaml_to_parser("exp_defs/audioset_params.yaml").parse_args([])
        from creators.dataset import _stft_config
        window, hop, _ = _stft_config(hp)
        self.assertEqual((window, hop), (512, 256))

    def test_embedding_dimension_and_accuracy_threshold_overrides(self):
        parser = yaml_to_parser("exp_defs/audioset_params.yaml")
        hp = parser.parse_args(["--audioset-embedding-dim", "64",
                                "--frame-accuracy-threshold", "0.7"])
        self.assertEqual(hp.audioset_embedding_dim, 64)
        self.assertEqual(hp.frame_accuracy_threshold, 0.7)

    def test_quote_stripped_path_lists_are_valid_yaml(self):
        parser = yaml_to_parser("exp_defs/audioset_params.yaml")
        hparams = parser.parse_args(
            [
                "--train-audio-dir",
                "[/hdd2/audio/balanced, /hdd2/audio/unbalanced]",
                "--train-weak-manifest",
                "[metadata/balanced.csv, metadata/unbalanced.csv]",
                "--audioset-conv-channels",
                "[16, 32, 64]",
            ]
        )
        self.assertEqual(
            hparams.train_audio_dir,
            ["/hdd2/audio/balanced", "/hdd2/audio/unbalanced"],
        )
        self.assertEqual(
            hparams.train_weak_manifest,
            ["metadata/balanced.csv", "metadata/unbalanced.csv"],
        )
        self.assertEqual(hparams.audioset_conv_channels, [16, 32, 64])


if __name__ == "__main__":
    unittest.main()
