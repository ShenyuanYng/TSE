import unittest

from utils.config import yaml_to_parser


class ConfigParserTest(unittest.TestCase):
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
