"""Graph direction, all-condition scores, positive-frame ranking and provenance."""

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
from diagnose_ontology_confusions import run
from nets.audioset_crnn import AudioSetCRNN
from source_separator import SourceSeparator
from sweep_validation_thresholds import run as reference_run
from utils.audioset_ontology import AudioSetOntology
from utils.config import yaml_to_parser
from utils.ontology_confusion import (OntologyConfusionAccumulator, probe_batches,
                                      score_all_conditions, strong_overlap_counts)


def ontology_records():
    # X is outside the model class map. C has multiple parents, and D is unknown.
    return [{"id": mid, "name": mid, "child_ids": children}
            for mid, children in (("A", ["B", "X"]), ("B", []), ("X", ["C"]),
                                  ("Z", ["C", "Y"]), ("C", []), ("Y", []))]


def entry(mid="A", weak=("A",)):
    return {"weak_mids": list(weak), "events": [
        {"mid": mid, "onset": 0., "offset": 1., "present": True}]}


class OntologyTests(unittest.TestCase):
    def test_dag_relations_full_paths_multiple_parents_and_missing_nodes(self):
        graph = AudioSetOntology(ontology_records())
        expected = (("A", "B", "other_is_child", 1),
                    ("A", "C", "other_is_descendant", 2),
                    ("C", "A", "other_is_ancestor", 2),
                    ("C", "Z", "other_is_parent", 1),
                    ("C", "Y", "no_lineage_relation", None),
                    ("A", "D", "unknown_mid", None), ("A", "A", "same", 0))
        for target, other, relation, distance in expected:
            result = graph.relation(target, other)
            self.assertEqual(result["ontology_relation"], relation)
            self.assertEqual(result["ontology_distance"], distance)
        self.assertEqual(graph.relation("C", "A")["ontology_path"], "A > X > C")
        self.assertEqual(graph.relation("C", "Y")["shared_parent_mids"], "Z")
        self.assertFalse(graph.relation("C", "Y")["is_ancestor_or_descendant"])
        with self.assertRaisesRegex(ValueError, "cycle"):
            AudioSetOntology([{"id": "A", "child_ids": ["B"]}, {"id": "B", "child_ids": ["A"]}])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            AudioSetOntology([{"id": "A", "child_ids": ["B"]}])

    def test_official_snapshot_relations(self):
        graph = AudioSetOntology.load("resources/audioset_ontology.json")
        self.assertEqual(len(graph.nodes), 632)
        self.assertEqual(graph.relation("/m/09x0r", "/m/02qldy")["ontology_relation"], "other_is_child")
        self.assertEqual(graph.relation("/m/05kq4", "/m/034srq")["ontology_relation"], "other_is_child")
        self.assertEqual(graph.relation("/m/06d_3", "/m/07jdr")["ontology_relation"], "other_is_child")
        self.assertEqual(graph.relation("/m/03j1ly", "/m/012n7d")["ontology_relation"], "other_is_child")
        self.assertEqual(graph.relation("/m/07yv9", "/m/03j1ly")["ontology_relation"], "other_is_descendant")

    def test_all_condition_head_matches_full_forward_for_every_condition(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = AudioSetCRNN(num_classes=4, conv_channels=[4], temporal_hidden=4,
                             rnn_layers=1, embedding_dim=3).eval()
        features = []
        hook = model.temporal_norm.register_forward_hook(lambda _m, _i, output: features.append(output.detach()))
        inputs = {"logmel": (torch.randn(2, 7, 8), torch.tensor([7, 4])),
                  "weak_labels": torch.tensor([[1., 0, 0, 0], [0, 1., 0, 0]])}
        with torch.inference_mode():
            original = model(inputs)
            hook.remove()
            probed = score_all_conditions(model, features[0].reshape(-1, 8), 2, 3).reshape(2, 7, 4)
            for category in range(4):
                condition = torch.zeros(2, 4)
                condition[:, category] = 1
                actual = model({**inputs, "weak_labels": condition})
                np.testing.assert_allclose(probed[:, :, category], actual["frame_probs"].numpy(), atol=1e-6)
            np.testing.assert_allclose(probed[0, :, 0], original["frame_probs"][0].numpy(), atol=1e-6)
            np.testing.assert_allclose(score_all_conditions(model, features[0].reshape(-1, 8), 1, 1),
                                       probed.reshape(-1, 4), atol=1e-6)
        self.assertEqual(score_all_conditions(model, features[0][:0].reshape(0, 8)).shape, (0, 4))

    def test_global_rank_weighting_annotations_and_distributed_dedup(self):
        collector = OntologyConfusionAccumulator({"A": 0, "B": 1, "C": 2, "D": 3},
                                                 ["A", "B", "C", "D"], ["A"], AudioSetOntology(ontology_records()))
        first_scores = np.array([[.1, .9, .8, .95]])
        collector.update_pair("one", 0, 1, first_scores[:, 0], first_scores,
                              entry(weak=("A", "C")), np.array([.1]))
        collector.update_pair("one", 0, 1, first_scores[:, 0], first_scores,
                              entry(weak=("A", "C")), np.array([.1]))
        first_rows = collector.rows.copy()
        second_scores = np.array([[.2, .6, .8, .05], [.3, .6, .2, .05], [.4, .6, .2, .05]])
        second = entry(weak=("A", "B"))
        second["events"] += [{"mid": "B", "onset": .15, "offset": .35},
                              {"mid": "B", "onset": .2, "offset": .25, "present": False}]
        collector.update_pair("two", 0, 3, second_scores[:, 0], second_scores, second, np.array([.1, .2, .3]))
        collector.update_pair("zero", 0, 5, np.array([]), np.empty((0, 4)), entry(), np.array([]))
        remote = collector.rows.copy()
        collector.rows = first_rows
        def gather(parts, local):
            parts[:] = [local, remote]
        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=2), \
             patch("torch.distributed.all_gather_object", side_effect=gather):
            top, all_rows, summaries = collector.compute({"A": {"tp": 0, "fn": 4, "valid_frames": 9}}, top_k=2)
        self.assertEqual([r["other_MID"] for r in top], ["B", "C"])
        self.assertEqual([r["other_MID"] for r in all_rows], ["B", "C", "D"])
        self.assertEqual(top[0]["target_positive_frames"], 4)
        self.assertEqual(top[0]["target_positive_clips"], 2)
        self.assertAlmostEqual(top[0]["other_gt_mean_probability"], .675)
        self.assertAlmostEqual(top[0]["target_gt_mean_probability"], .25)
        self.assertEqual(top[0]["other_weak_present_gt_frames"], 3)
        self.assertEqual(top[0]["other_weak_present_clips"], 1)
        self.assertEqual(top[0]["other_strong_overlap_frames"], 1)
        self.assertEqual(top[0]["frames_other_gt_target"], 4)
        self.assertEqual(sum(r["frames_top1_among_others"] for r in all_rows), 4)
        self.assertEqual(top[1]["ontology_relation"], "other_is_descendant")
        self.assertEqual(all_rows[2]["ontology_relation"], "unknown_mid")
        self.assertEqual(summaries[0]["target_valid_frames"], 9)
        collector.reset()
        self.assertFalse(collector.rows)

    def test_probe_mask_and_audio_encoding_reuse(self):
        torch.manual_seed(12)
        model = AudioSetCRNN(num_classes=2, conv_channels=[4], temporal_hidden=4,
                             rnn_layers=1, embedding_dim=3).eval()
        module = SimpleNamespace(model=model, task=AudioSetStrongTask(), eval=model.eval)
        collector = OntologyConfusionAccumulator({"A": 0, "B": 1}, ["A", "B"], ["A"],
                                                 AudioSetOntology(ontology_records()))
        inputs = {"logmel": (torch.randn(1, 8, 8), torch.tensor([7])),
                  "weak_labels": torch.tensor([[1., 0.]])}
        targets = {"utt_id": ["clip"], "class_index": [0],
                   "strong_labels": torch.ones(1, 8), "strong_mask": torch.tensor([[1, 0, 1, 1, 1, 1, 1, 1]])}
        # .112 is a boundary at frame 7 that float32 frame_times can round down.
        annotations = entry()
        annotations["events"].append({"mid": "B", "onset": .096, "offset": .112})
        with patch.object(model.encoder, "forward", wraps=model.encoder.forward) as encoder:
            count = probe_batches(module, [(inputs, targets)] * 2, "cpu", collector,
                                  {"clip": annotations}, 256, 16000, log_every=0)
        self.assertEqual((count, encoder.call_count), (2, 2))
        self.assertEqual(len(collector.rows), 1)
        top, _, _ = collector.compute({"A": {"tp": 0, "fn": 6, "valid_frames": 6}})
        self.assertEqual(top[0]["target_positive_frames"], 6)
        self.assertEqual(top[0]["other_strong_overlap_frames"], 1)
        self.assertFalse(model.temporal_norm._forward_hooks)
        bad = np.array([[.9, .2]])
        with self.assertRaisesRegex(ValueError, "differ"):
            collector.update_pair("bad", 0, 1, np.array([.1]), bad, annotations, np.array([.1]))

    def test_real_checkpoint_export_and_reference_validation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            audio = root / "audio"
            audio.mkdir()
            entries, weak, strong = [], [], []
            for index in range(4):
                clip = f"clip{index}"
                sf.write(str(audio / f"{clip}_0.wav"), np.full(2400 + index * 128, .1), 16000)
                entries.append({"id": f"{clip}_0", "ytid": clip})
                weak.append(f'{clip},0,0.2,"A,B"\n')
                strong.extend((f"{clip}_0\t0.032\t0.096\tA\n", f"{clip}_0\t0.064\t0.112\tB\n"))
            (root / "weak.csv").write_text("".join(weak))
            (root / "strong.tsv").write_text("".join(strong))
            (root / "classes.csv").write_text('index,mid,display_name\n0,A,"目标,A"\n1,B,Other\n')
            graph_path = root / "ontology.json"
            graph_path.write_text(json.dumps(ontology_records()))
            hp = yaml_to_parser("exp_defs/audioset_params.yaml").parse_args([])
            hp.data_path, hp.class_map_path = str(root), "classes.csv"
            hp.train_audio_dir, hp.train_weak_manifest = [str(audio)], ["weak.csv"]
            hp.train_strong_manifest, hp.train_val_fraction = "strong.tsv", .5
            hp.num_classes, hp.audioset_conv_channels = 2, [4]
            hp.audioset_temporal_hidden, hp.audioset_rnn_layers = 4, 1
            hp.audioset_embedding_dim, hp.batch_size, hp.num_workers = 3, 2, 0
            split_train(entries, root, hp.train_split_path, hp.train_val_fraction, hp.train_split_seed)
            model = SourceSeparator(hp)
            checkpoint = root / "test.ckpt"
            torch.save({"epoch": 30, "hyper_parameters": vars(hp), "state_dict": model.state_dict()}, checkpoint)
            checkpoint_bytes = checkpoint.read_bytes()
            split_bytes = (root / hp.train_split_path).read_bytes()
            reference_run(checkpoint, root / "reference", device="cpu", per_class=True)
            reference_path = root / "reference/threshold_sweep.json"
            reference = json.loads(reference_path.read_text())
            loss = Mock(side_effect=AssertionError("No loss in ontology diagnostic"))
            with patch("source_separator.loss_creator", return_value=loss), \
                 patch.object(SourceSeparator, "configure_optimizers", side_effect=AssertionError("No training")):
                report = run(checkpoint, reference_path, root / "output", graph_path, ["A"], "cpu", 0, 1, 2)
            loss.assert_not_called()
            self.assertEqual(report["prediction_batches"], 2)
            self.assertEqual(report["targets"][0]["target_positive_frames"],
                             reference["results"][0]["per_class"][0]["tp"] + reference["results"][0]["per_class"][0]["fn"])
            self.assertEqual(report["targets"][0]["evaluated_other_classes"], 1)
            with (root / "output/ontology_confusion_top10.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["target_class_name"], "目标,A")
            self.assertEqual(rows[0]["other_MID"], "B")
            self.assertEqual(rows[0]["ontology_relation"], "other_is_child")
            self.assertGreater(int(rows[0]["other_strong_overlap_frames"]), 0)
            self.assertEqual(float(rows[0]["other_weak_present_gt_frame_ratio"]), 1.)
            self.assertEqual(json.loads((root / "output/ontology_confusion_summary.json").read_text()), report)
            self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
            self.assertEqual((root / hp.train_split_path).read_bytes(), split_bytes)
            reference["split_sha256"] = "wrong"
            reference_path.write_text(json.dumps(reference))
            with self.assertRaisesRegex(ValueError, "split"):
                run(checkpoint, reference_path, root / "invalid", graph_path, ["A"], "cpu")
            self.assertFalse((root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
