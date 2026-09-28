"""Audit clip-level Weak–Strong relations over all Strong Train classes, on CPU.

Reuse the training manifest/audio join and ontology loader without constructing a
dataset, decoding audio, generating a split, or importing the training pipeline.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

from datasets.paths.audioset_paths import get_audioset_paths, load_class_map, resolve_path
from utils.audioset_ontology import AudioSetOntology


DEFAULT_ONTOLOGY = Path(__file__).resolve().parent / "resources/audioset_ontology.json"
DIRECTIONS = ("weak_to_strong", "strong_to_weak")
BUCKETS = ("exact_match", "d1", "d2", "distant", "unrelated")
RELATIONS = ("parent", "child", "ancestor", "descendant", "sibling", "co_parent")
PAIR_FIELDS = ("clip_id", "direction", "source_mid", "nearest_mid", "distance",
               "category", "relation_type", "ontology_close", "reason",
               "nearest_tie_count", "nearest_ties_json")


class NearestOntology:
    """Shortest undirected distance, retaining the existing directed DAG semantics."""

    def __init__(self, ontology):
        self.ontology = ontology
        self.neighbors = {mid: ontology.parents[mid] | set(node["child_ids"])
                          for mid, node in ontology.nodes.items()}
        self.distances = {}

    def distances_from(self, source):
        if source not in self.distances:
            distances = {source: 0}
            queue = deque([source])
            while queue:
                current = queue.popleft()
                for neighbor in self.neighbors.get(current, ()):
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
            self.distances[source] = distances
        return self.distances[source]

    def relation(self, source, target, distance):
        # Relation describes TARGET relative to SOURCE in either audit direction.
        if distance == 0:
            return "exact_match"
        if distance is None:
            return "unrelated"
        directed = self.ontology.relation(source, target)["ontology_relation"]
        if directed.startswith("other_is_"):
            return directed.removeprefix("other_is_")
        if self.ontology.parents[source] & self.ontology.parents[target]:
            return "sibling"
        # A multi-parent DAG also permits A -> C <- B, a genuine d=2 path.
        if set(self.ontology.nodes[source]["child_ids"]) & set(
                self.ontology.nodes[target]["child_ids"]):
            return "co_parent"
        return "other"

    def nearest(self, source, targets):
        distances = self.distances_from(source)
        reachable = [(distances[mid], mid) for mid in targets if mid in distances]
        if not reachable:
            reason = ("empty_target_set" if not targets else
                      "unknown_source_mid" if source not in self.ontology.nodes else
                      "unknown_target_mids" if not any(mid in self.ontology.nodes for mid in targets)
                      else "disconnected")
            return dict(nearest_mid="", distance=None, category="unrelated",
                        relation_type="unrelated", ontology_close=False, reason=reason,
                        nearest_tie_count=0, nearest_ties_json="[]")
        distance, target = min(reachable)  # MID breaks ties reproducibly.
        ties = [{"mid": mid, "relation_type": self.relation(source, mid, distance)}
                for d, mid in sorted(reachable) if d == distance]
        category = ("exact_match" if distance == 0 else "d1" if distance == 1 else
                    "d2" if distance == 2 else "distant")
        return dict(nearest_mid=target, distance=distance, category=category,
                    relation_type=self.relation(source, target, distance),
                    ontology_close=distance <= 2, reason="",
                    nearest_tie_count=len(ties),
                    nearest_ties_json=json.dumps(ties, ensure_ascii=False, separators=(",", ":")))


def pool_strong_labels(entry):
    """Pool positive-duration PRESENT intervals; NOT_PRESENT overrides by MID.

    This is continuous-time clip support, not a count on the model's frame grid.
    Clip-relative intervals are intersected with the manifest clip duration.
    """
    duration = float(entry["clip_end"]) - float(entry["clip_start"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid clip duration for {entry['id']}")
    intervals = defaultdict(lambda: {True: [], False: []})
    for event in entry.get("events", []):
        onset, offset = float(event["onset"]), float(event["offset"])
        if not math.isfinite(onset) or not math.isfinite(offset) or offset < onset:
            raise ValueError(f"Invalid strong interval for {entry['id']}: {event}")
        start, end = max(0., onset), min(duration, offset)
        if start < end:
            intervals[event["mid"]][event.get("present", True)].append((start, end))
    pooled = set()
    for mid, by_status in intervals.items():
        absent = sorted(by_status[False])
        for start, end in by_status[True]:
            cursor = start
            for left, right in absent:
                if left > cursor or cursor >= end:
                    break
                cursor = max(cursor, right)
            if cursor < end:
                pooled.add(mid)
                break
    return pooled


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def directional_statistics(counts):
    support = counts["support"]
    result = {"support": support}
    for bucket in BUCKETS:
        result[f"{bucket}_count"] = counts[bucket]
        result[f"{bucket}_rate"] = rate(counts[bucket], support)
    close = sum(counts[key] for key in ("exact_match", "d1", "d2"))
    result.update(ontology_close_count=close, ontology_close_rate=rate(close, support))
    for distance in (1, 2):
        for relation in RELATIONS:
            key = f"d{distance}_{relation}"
            result[f"{key}_count"] = counts[key]
            result[f"{key}_rate"] = rate(counts[key], support)
    result["nearest_tie_count"] = counts["nearest_tie"]
    result["empty_target_set_count"] = counts["empty_target_set"]
    return result


def load_metrics(path):
    """Accept original per_class_metrics.csv or the positive-class export."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        keys = {key.lower(): key for key in columns}
        if not {"mid", "precision", "recall", "f1"}.issubset(keys):
            raise ValueError("Metrics CSV requires MID, Precision, Recall, F1 (case insensitive)")
        rows = {}
        for original in reader:
            mid = original[keys["mid"]].strip()
            if not mid or mid in rows:
                raise ValueError(f"Empty or duplicate metrics MID: {mid!r}")
            row = {"eval_" + key: value for key, value in original.items()}
            for field in ("Precision", "Recall", "F1"):
                raw = original[keys[field.lower()]].strip()
                if not raw:
                    row[field] = None  # Original evaluation has no valid frames.
                    continue
                value = float(raw)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid {field} for {mid}: {value}")
                row[field] = value
            row["metrics_class_name"] = original.get(keys.get("class_name"), "")
            if "tp" in keys and "fn" in keys:
                row["eval_positive_frames"] = int(original[keys["tp"]]) + int(original[keys["fn"]])
            rows[mid] = row
    if not rows:
        raise ValueError("Metrics CSV is empty")
    return rows


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(entries, mid_to_index, class_names, ontology, metrics, output_dir,
            provenance=None, write_pairs=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = NearestOntology(ontology)
    counts = {direction: defaultdict(Counter) for direction in DIRECTIONS}
    totals = {direction: Counter() for direction in DIRECTIONS}
    clip_counts, observed = Counter(), set()
    seen = set()
    pair_handle = None
    if write_pairs:
        pair_handle = (output_dir / "nearest_label_pairs.csv").open("w", newline="", encoding="utf-8")
        pair_writer = csv.DictWriter(pair_handle, fieldnames=PAIR_FIELDS)
        pair_writer.writeheader()
    try:
        for entry in entries:
            if entry["id"] in seen:
                raise ValueError(f"Duplicate clip ID: {entry['id']}")
            seen.add(entry["id"])
            weak, strong = set(entry.get("weak_mids", [])), pool_strong_labels(entry)
            observed.update(weak | strong)
            clip_counts["clips"] += 1
            clip_counts["clips_with_weak_labels"] += bool(weak)
            clip_counts["clips_with_strong_positive_labels"] += bool(strong)
            clip_counts["clips_with_identical_label_sets"] += weak == strong
            clip_counts["clips_with_any_exact_overlap"] += bool(weak & strong)
            for direction, source, target in ((DIRECTIONS[0], weak, strong),
                                              (DIRECTIONS[1], strong, weak)):
                for mid in sorted(source):
                    result = graph.nearest(mid, target)
                    keys = ["support", result["category"]]
                    if result["distance"] in (1, 2):
                        keys.append(f"d{result['distance']}_{result['relation_type']}")
                    if result["nearest_tie_count"] > 1:
                        keys.append("nearest_tie")
                    if result["reason"]:
                        keys.append(result["reason"])
                    counts[direction][mid].update(keys)
                    totals[direction].update(keys)
                    if pair_handle:
                        pair_writer.writerow(dict(clip_id=entry["id"], direction=direction,
                                                  source_mid=mid, **result))
    finally:
        if pair_handle:
            pair_handle.close()
    if not seen:
        raise ValueError("No clips to analyze")

    rows = []
    for mid in sorted(set(mid_to_index) | observed | set(metrics)):
        forward = directional_statistics(counts[DIRECTIONS[0]][mid])
        reverse = directional_statistics(counts[DIRECTIONS[1]][mid])
        weak, strong = forward["support"], reverse["support"]
        exact = forward["exact_match_count"]
        assert exact == reverse["exact_match_count"]
        index = mid_to_index.get(mid)
        name = (class_names[index] if index is not None else
                ontology.nodes.get(mid, {}).get("name", metrics.get(mid, {}).get("metrics_class_name", "")))
        row = dict(MID=mid, class_name=name, class_index=index,
                   in_class_map=mid in mid_to_index, in_ontology=mid in ontology.nodes,
                   support=weak, positive_clips=strong,
                   weak_positive_clips=weak, strong_positive_clips=strong,
                   union_positive_clips=weak + strong - exact, exact_match_clips=exact,
                   weak_only_clips=weak - exact, strong_only_clips=strong - exact,
                   weak_only_rate=rate(weak - exact, weak),
                   strong_only_rate=rate(strong - exact, strong))
        for key in (*BUCKETS, "ontology_close"):
            row[f"{key}_rate"] = forward[f"{key}_rate"]
        for direction, stats in zip(DIRECTIONS, (forward, reverse)):
            row.update({direction + "_" + key: value for key, value in stats.items()})
        row.update(metrics_available=mid in metrics, Precision=None, Recall=None, F1=None)
        row.update(metrics.get(mid, {}))
        rows.append(row)
    write_csv(output_dir / "weak_strong_per_class.csv", rows)
    ordered = sorted(rows, key=lambda row: (row["F1"] is None,
                                            row["F1"] if row["F1"] is not None else 0,
                                            row["MID"]))
    write_csv(output_dir / "weak_strong_by_f1.csv", ordered)

    summary = {
        "provenance": provenance or {}, "clip_counts": dict(clip_counts),
        "class_count": len(rows), "class_map_count": len(mid_to_index),
        "observed_class_count": len(observed), "ontology_node_count": len(ontology.nodes),
        "observed_mids_missing_from_ontology": sorted(observed - set(ontology.nodes)),
        "observed_mids_missing_from_class_map": sorted(observed - set(mid_to_index)),
        "class_map_mids_without_metrics": sorted(set(mid_to_index) - set(metrics)),
        "metrics_mids_outside_class_map": sorted(set(metrics) - set(mid_to_index)),
        "definitions": {
            "population": "One observation per unique audio clip; no weak-condition expansion or F1 filtering",
            "strong_pooling": "Any positive-duration PRESENT interval within clip, after NOT_PRESENT subtraction",
            "distance": "Shortest undirected parent-child graph path in the FULL ontology, including non-model nodes",
            "categories": {"exact_match": "d=0 (identical MID only)", "d1": "d=1", "d2": "d=2",
                           "distant": "finite d>2", "unrelated": "No reachable target, including empty target sets"},
            "unknown_mids": "Retained and flagged; identical MIDs are exact even if absent from ontology",
            "ontology_close": "d<=2; exact_match + d1 + d2, never used to redefine exact match",
            "directional_rates": "Count / source-side positive clips of this MID; zero support => null/empty CSV cell",
            "support": "weak_positive_clips (denominator of unprefixed distance rates)",
            "positive_clips": "strong_positive_clips; union_positive_clips counts either side once",
            "unprefixed_distance_rates": "Aliases of weak_to_strong rates; reverse rates have strong_to_weak prefix",
            "weak_only_rate": "(weak_positive_clips - exact_match_clips) / weak_positive_clips",
            "strong_only_rate": "(strong_positive_clips - exact_match_clips) / strong_positive_clips",
            "only_rates": "Exact-MID absence on other side, including close and distant matches; not an extra distance bucket",
            "relation_type": "Nearest TARGET relative to SOURCE; directed lineage first, then sibling (shared parent), then co_parent (shared child)",
            "co_parent": "Multi-parent DAG permits A -> C <- B at d=2; cannot truthfully call these siblings or lineage",
            "ties": "Choose lexicographically smallest nearest MID, count source once; optional pair CSV preserves all tied MIDs and relations",
            "overall_rates": "Micro weighted over all source (clip, MID) positives, separately in each direction",
            "metrics": "Original evaluation frame metrics joined by MID; no recomputation or zero-fill of absent metrics",
            "interpretation": "Clip label differences and co-occurrence do not by themselves prove annotation errors or cause low F1",
        },
        "overall": {direction: directional_statistics(totals[direction]) for direction in DIRECTIONS},
    }
    for direction, only in zip(DIRECTIONS, ("weak_only", "strong_only")):
        stats = summary["overall"][direction]
        stats[only + "_count"] = stats["support"] - stats["exact_match_count"]
        stats[only + "_rate"] = rate(stats[only + "_count"], stats["support"])
    # Compact groups support low-F1/support inspection without excluding classes.
    groups = {"missing_or_undefined_F1": [], "F1=0": [], "0<F1<=0.25": [],
              "0.25<F1<=0.5": [], "0.5<F1<=0.75": [], "0.75<F1<=1": []}
    for row in rows:
        f1 = row["F1"]
        group = ("missing_or_undefined_F1" if f1 is None else "F1=0" if f1 == 0 else
                 "0<F1<=0.25" if f1 <= .25 else "0.25<F1<=0.5" if f1 <= .5 else
                 "0.5<F1<=0.75" if f1 <= .75 else "0.75<F1<=1")
        groups[group].append(row)
    summary["f1_groups"] = []
    for group, members in groups.items():
        weak = sum(row["weak_positive_clips"] for row in members)
        strong = sum(row["strong_positive_clips"] for row in members)
        summary["f1_groups"].append(dict(
            group=group, class_count=len(members), weak_positive_clip_class_pairs=weak,
            strong_positive_clip_class_pairs=strong,
            mean_weak_positive_clips=rate(weak, len(members)),
            mean_strong_positive_clips=rate(strong, len(members)),
            weak_only_rate=rate(sum(row["weak_only_clips"] for row in members), weak),
            strong_only_rate=rate(sum(row["strong_only_clips"] for row in members), strong),
            weak_to_strong_ontology_close_rate=rate(sum(
                row["weak_to_strong_ontology_close_count"] for row in members), weak),
            classes_with_zero_eval_positive_frames=sum(row.get("eval_positive_frames") == 0 for row in members)))
    write_csv(output_dir / "weak_strong_f1_groups.csv", summary["f1_groups"])
    (output_dir / "weak_strong_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--audio-dirs", nargs="+", required=True,
                        help="Training audio directories, paired with weak manifests as in training")
    parser.add_argument("--weak-manifests", nargs="+", required=True)
    parser.add_argument("--strong-manifest", default="metadata/audioset_train_strong.tsv")
    parser.add_argument("--class-map", default="metadata/class_labels_indices.csv")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=("all", "tr", "cv"), default="all",
                        help="Default all: full joined Strong Train before holdout")
    parser.add_argument("--split-json", type=Path,
                        help="Existing saved train/validation split; required for tr/cv, never created or changed")
    parser.add_argument("--allow-missing-audio", action="store_true",
                        help="Use training strict=False inventory; report excluded strong-manifest IDs")
    parser.add_argument("--write-pairs", action="store_true", help="Also stream per-clip directional nearest matches")
    args = parser.parse_args(argv)
    if args.partition != "all" and args.split_json is None:
        parser.error("--partition tr/cv requires an existing --split-json")
    paths = {key: resolve_path(value, args.data_root) for key, value in
             (("strong_manifest", args.strong_manifest), ("class_map", args.class_map))}
    metrics = load_metrics(args.metrics_csv)
    ontology = AudioSetOntology.load(args.ontology)
    mid_to_index, names = load_class_map(paths["class_map"])
    entries = get_audioset_paths(args.data_root, args.audio_dirs, args.weak_manifests,
                                 args.strong_manifest, strict=not args.allow_missing_audio)
    joined_ids = {entry["id"] for entry in entries}
    # Audit skipped inventory using the same TSV reader, only if skipping is enabled.
    excluded = []
    if args.allow_missing_audio:
        from datasets.paths.audioset_paths import read_strong_labels
        excluded = sorted(set(read_strong_labels(paths["strong_manifest"])) - joined_ids)
    if args.split_json:
        split = json.loads(args.split_json.read_text(encoding="utf-8"))
        tr, cv = split["tr"], split["cv"]
        if (len(tr + cv) != len(set(tr + cv)) or set(tr + cv) != joined_ids or not tr or not cv):
            raise ValueError("Saved split must partition the full joined inventory without duplicates")
        if args.partition != "all":
            selected = set(split[args.partition])
            entries = [entry for entry in entries if entry["id"] in selected]
    provenance = {
        "partition": args.partition, "joined_clip_count_before_partition": len(joined_ids),
        "strict_paths": not args.allow_missing_audio,
        "excluded_strong_clip_count": len(excluded), "excluded_strong_clip_ids": excluded,
        "audio_dirs": [resolve_path(path, args.data_root) for path in args.audio_dirs],
        "weak_manifests": [fingerprint(resolve_path(path, args.data_root)) for path in args.weak_manifests],
        **{key: fingerprint(path) for key, path in paths.items()},
        "ontology": fingerprint(args.ontology), "metrics_csv": fingerprint(args.metrics_csv),
        "split_json": fingerprint(args.split_json) if args.split_json else None,
    }
    print(f"Analyzing {len(entries)} clips, all {len(mid_to_index)} mapped classes, both directions", flush=True)
    summary = analyze(entries, mid_to_index, names, ontology, metrics, args.output_dir,
                      provenance, write_pairs=args.write_pairs)
    for direction, stats in summary["overall"].items():
        print(direction, json.dumps({key: stats[key] for key in
              ("support", "exact_match_rate", "d1_rate", "d2_rate", "distant_rate", "unrelated_rate", "ontology_close_rate")}))
    print(f"Reports: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
