"""Clip inventory, graph distance, pooling and metric-join regression coverage."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from analyze_weak_strong_labels import (NearestOntology, analyze, load_metrics,
                                      main, pool_strong_labels)
from utils.audioset_ontology import AudioSetOntology


def ontology():
    # X is an intermediate node absent from the model vocabulary; Z is isolated.
    return AudioSetOntology([
        {"id": mid, "name": mid, "child_ids": children}
        for mid, children in (("A", ["B", "X"]), ("B", ["C"]),
                              ("X", ["Y"]), ("C", []), ("Y", []),
                              ("P", ["Y"]), ("Z", []))])


def event(mid, start=0., end=1., present=True):
    return dict(mid=mid, onset=start, offset=end, present=present)


def entry(clip, weak, strong):
    return dict(id=clip, clip_start=100., clip_end=110.,
                weak_mids=weak, events=[event(mid) for mid in strong])


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class WeakStrongTests(unittest.TestCase):
    def test_full_graph_both_directions_and_disconnected(self):
        graph = NearestOntology(ontology())
        cases = [("A", "A", 0, "exact_match"), ("A", "B", 1, "child"),
                 ("B", "A", 1, "parent"), ("A", "C", 2, "descendant"),
                 ("C", "A", 2, "ancestor"), ("B", "X", 2, "sibling"),
                 ("X", "P", 2, "co_parent"), ("C", "Y", 4, "other"),
                 ("A", "Z", None, "unrelated")]
        for source, target, distance, relation in cases:
            with self.subTest(source=source, target=target):
                result = graph.nearest(source, {target})
                self.assertEqual(result["distance"], distance)
                self.assertEqual(result["relation_type"], relation)
                self.assertEqual(result["ontology_close"], distance is not None and distance <= 2)
        self.assertEqual(graph.nearest("A", {"C", "B"})["nearest_mid"], "B")
        self.assertEqual(graph.nearest("A", {"A", "B"})["category"], "exact_match")
        self.assertEqual(graph.nearest("A", set())["reason"], "empty_target_set")
        self.assertEqual(graph.nearest("unknown", {"A"})["reason"], "unknown_source_mid")
        self.assertEqual(graph.nearest("A", {"unknown"})["reason"], "unknown_target_mids")
        self.assertEqual(graph.nearest("unknown", {"unknown"})["distance"], 0)

    def test_shortest_path_beats_long_lineage_and_ties_are_reproducible(self):
        graph = NearestOntology(ontology())
        result = graph.nearest("A", {"X", "B"})
        self.assertEqual(result["nearest_mid"], "B")
        self.assertEqual(result["nearest_tie_count"], 2)
        self.assertEqual([row["mid"] for row in json.loads(result["nearest_ties_json"])], ["B", "X"])
        # A long directed path can have a shorter undirected shortcut.
        graph = NearestOntology(AudioSetOntology([
            {"id": mid, "child_ids": children} for mid, children in
            (("A", ["B", "Q"]), ("B", ["C"]), ("C", ["D"]),
             ("D", ["Q"]), ("Q", []))]))
        self.assertEqual(graph.nearest("A", {"D"})["distance"], 2)
        self.assertEqual(graph.nearest("A", {"D"})["relation_type"], "descendant")

    def test_pooling_deduplicates_clips_and_absence_overrides(self):
        clip = entry("one", [], [])
        clip["events"] = [event("A"), event("A", 3, 4), event("B", present=False),
                          event("C", 0, 2), event("C", 0, 1, False), event("C", 1, 2, False),
                          event("D", 1, 2), event("D", 1, 1.5, False),
                          event("E", 12, 13), event("F", -2, 0), event("G", 2, 2)]
        self.assertEqual(pool_strong_labels(clip), {"A", "D"})
        clip["events"] += [event("bad", 3, 2)]
        with self.assertRaisesRegex(ValueError, "Invalid strong interval"):
            pool_strong_labels(clip)

    def test_directional_denominators_zero_support_and_outer_join(self):
        clips = [entry("1", ["A", "A", "B"], ["A", "A"]),
                 entry("2", ["A"], ["C"]), entry("3", ["A"], ["Z"]),
                 entry("4", ["A"], []), entry("5", [], ["B"])]
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "metrics.csv"
            metrics_path.write_text("mid,precision,recall,f1,tp,fn\nA,0,0,0,0,5\nB,1,1,1,3,0\nM,0.5,0.5,0.5,1,1\n")
            metrics = load_metrics(metrics_path)
            summary = analyze(clips, {"A": 0, "B": 1, "Z": 2, "unused": 3},
                              ["A", "B", "Z", "unused"], ontology(), metrics,
                              directory, write_pairs=True)
            rows = {row["MID"]: row for row in csv_rows(Path(directory) / "weak_strong_per_class.csv")}
            a = rows["A"]
            self.assertEqual(a["weak_positive_clips"], "4")
            self.assertEqual(a["strong_positive_clips"], "1")
            self.assertEqual(a["exact_match_rate"], "0.25")
            self.assertEqual(a["d2_rate"], "0.25")
            self.assertEqual(a["unrelated_rate"], "0.5")
            self.assertEqual(a["ontology_close_rate"], "0.5")
            self.assertEqual(a["weak_only_rate"], "0.75")
            self.assertEqual(a["strong_only_rate"], "0.0")
            self.assertEqual(a["strong_to_weak_exact_match_rate"], "1.0")
            self.assertEqual(rows["unused"]["exact_match_rate"], "")
            self.assertEqual(rows["C"]["metrics_available"], "False")
            self.assertEqual(rows["C"]["F1"], "")
            self.assertEqual(rows["M"]["support"], "0")
            self.assertEqual(summary["class_count"], 6)
            for stats in summary["overall"].values():
                self.assertEqual(sum(stats[b + "_count"] for b in
                                     ("exact_match", "d1", "d2", "distant", "unrelated")), stats["support"])
            self.assertEqual(summary["overall"]["weak_to_strong"]["support"], 5)
            self.assertEqual(summary["overall"]["strong_to_weak"]["support"], 4)
            self.assertEqual(len(csv_rows(Path(directory) / "nearest_label_pairs.csv")), 9)
            ordered = csv_rows(Path(directory) / "weak_strong_by_f1.csv")
            self.assertEqual([row["MID"] for row in ordered[:3]], ["A", "M", "B"])
            with self.assertRaisesRegex(ValueError, "Duplicate clip"):
                analyze([clips[0], clips[0]], {}, [], ontology(), {}, directory)

    def test_metrics_validation_and_uppercase_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_text("MID,Precision,Recall,F1\nA,1,1,1\n")
            self.assertEqual(load_metrics(path)["A"]["F1"], 1)
            path.write_text("MID,Precision,Recall,F1\nA,,,\n")
            self.assertIsNone(load_metrics(path)["A"]["F1"])
            path.write_text("MID,Precision,Recall,F1\nA,1,1,1\nA,1,1,1\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_metrics(path)
            path.write_text("MID,Precision,Recall,F1\nA,1,1,nan\n")
            with self.assertRaisesRegex(ValueError, "Invalid"):
                load_metrics(path)

    def test_cli_reuses_manifest_audio_join_and_reads_split_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "audio").mkdir()
            for clip in ("one_0", "two_0"):
                # File existence is enough: this analysis must never decode audio.
                (root / "audio" / (clip + ".wav")).touch()
            (root / "weak.csv").write_text('YTID,start_seconds,end_seconds,positive_labels\none,0,10,"A,A"\ntwo,0,10,B\n')
            (root / "weak2.csv").write_text('YTID,start_seconds,end_seconds,positive_labels\none,0,10,B\n')
            (root / "strong.tsv").write_text("clip_id\tonset\toffset\tMID\none_0\t0\t1\tA\none_0\t2\t3\tA\ntwo_0\t0\t1\tC\n")
            (root / "classes.csv").write_text("index,mid,display_name\n0,A,A\n1,B,B\n2,C,C\n")
            (root / "metrics.csv").write_text("mid,precision,recall,f1\nA,1,1,1\nB,0,0,0\nC,0,0,0\n")
            (root / "ontology.json").write_text(json.dumps(list(ontology().nodes.values())))
            split = root / "split.json"
            split.write_text('{"tr": ["one_0"], "cv": ["two_0"]}')
            before = split.read_bytes()
            args = ["--data-root", directory, "--audio-dirs", "audio", "--weak-manifests", "weak.csv", "weak2.csv",
                    "--strong-manifest", "strong.tsv", "--class-map", "classes.csv",
                    "--metrics-csv", str(root / "metrics.csv"), "--ontology", str(root / "ontology.json"),
                    "--output-dir", str(root / "out"), "--split-json", str(split)]
            main(args)
            summary_path = root / "out/weak_strong_summary.json"
            self.assertEqual(json.loads(summary_path.read_text())["clip_counts"]["clips"], 2)
            self.assertEqual(json.loads(summary_path.read_text())["overall"]["weak_to_strong"]["support"], 3)
            main(args + ["--partition", "tr"])
            self.assertEqual(json.loads(summary_path.read_text())["clip_counts"]["clips"], 1)
            self.assertEqual(split.read_bytes(), before)
            (root / "audio/two_0.wav").unlink()
            with self.assertRaises(FileNotFoundError):
                main(args)
            main(args[:-2] + ["--allow-missing-audio"])
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["provenance"]["excluded_strong_clip_ids"], ["two_0"])


if __name__ == "__main__":
    unittest.main()
