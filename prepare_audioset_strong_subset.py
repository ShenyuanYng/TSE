"""Build rsync file lists for a local AudioSet Strong training subset.

This script never copies or modifies the shared raw corpus.  It joins the
official weak manifests to the strong clip ids so YTID-only WAV names are read
from the correct balanced, unbalanced, or evaluation directory.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict

from datasets.paths.audioset_paths import read_strong_labels, read_weak_labels


SOURCES = {
    "balance_wav": "balanced_train_segments.csv",
    "unbalanced_wav": "unbalanced_train_segments.csv",
    "eval_wav": "eval_segments.csv",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _manifest_index(metadata_dir, keep_ids):
    by_clip_id = defaultdict(list)
    for source_dir, filename in SOURCES.items():
        path = os.path.join(metadata_dir, filename)
        for entry in read_weak_labels(path, keep_ids=keep_ids):
            by_clip_id[entry["id"]].append((source_dir, entry["ytid"]))
    return by_clip_id


def _resolve_partition(
    name,
    strong_ids,
    allowed_sources,
    raw_root,
    weak_index,
    fallback_source=None,
    fallback_partition=None,
):
    output_sources = list(allowed_sources)
    if fallback_source is not None:
        output_sources.append(fallback_source)
    files_by_source = {source: set() for source in output_sources}
    clips_by_source_file = defaultdict(list)
    rows = []
    missing_audio = []
    missing_manifest = []
    ambiguous = []

    for clip_id in strong_ids:
        candidates = [
            item for item in weak_index.get(clip_id, []) if item[0] in allowed_sources
        ]
        if not candidates:
            missing_manifest.append(clip_id)
            continue
        if len(candidates) != 1:
            ambiguous.append((clip_id, candidates))
            continue

        source_dir, ytid = candidates[0]
        filename = ytid + ".wav"
        source_path = os.path.join(raw_root, source_dir, filename)
        if not os.path.isfile(source_path):
            fallback_path = (
                os.path.join(raw_root, fallback_source, filename)
                if fallback_source is not None
                else None
            )
            if fallback_path is None or not os.path.isfile(fallback_path):
                missing_audio.append(clip_id)
                continue
            source_dir = fallback_source
            row_partition = fallback_partition
        else:
            row_partition = name

        files_by_source[source_dir].add(filename)
        clips_by_source_file[(source_dir, filename)].append(clip_id)
        rows.append((row_partition, clip_id, source_dir, filename))

    collisions = {
        key: clip_ids
        for key, clip_ids in clips_by_source_file.items()
        if len(clip_ids) > 1
    }

    resolved_by_partition = Counter(row[0] for row in rows)
    print(f"{name}: strong clips={len(strong_ids)}, resolved={len(rows)}")
    for partition, count in sorted(resolved_by_partition.items()):
        print(f"  {partition}: resolved clips={count}")
    for source_dir in output_sources:
        print(f"  {source_dir}: unique WAV files={len(files_by_source[source_dir])}")
    print(f"  missing audio={len(missing_audio)}")
    print(f"  missing weak-manifest entry={len(missing_manifest)}")
    print(f"  ambiguous manifest entries={len(ambiguous)}")
    print(f"  source filename collisions={len(collisions)}")
    if missing_audio:
        print(f"  first missing audio={missing_audio[:5]}")
    if missing_manifest:
        print(f"  first missing manifest={missing_manifest[:5]}")
    if ambiguous:
        print(f"  first ambiguous={ambiguous[:2]}")
    if collisions:
        print(f"  first source collisions={list(collisions.items())[:2]}")
        raise RuntimeError(
            "A source WAV maps to multiple strong clip ids; copying it by YTID would "
            "silently use the wrong 10-second excerpt"
        )
    return files_by_source, rows


def main():
    args = parse_args()
    raw_root = os.path.abspath(os.path.expandvars(os.path.expanduser(args.raw_root)))
    metadata_dir = os.path.abspath(
        os.path.expandvars(os.path.expanduser(args.metadata_dir))
    )
    output_dir = os.path.abspath(
        os.path.expandvars(os.path.expanduser(args.output_dir))
    )
    os.makedirs(output_dir, exist_ok=True)

    train_ids = sorted(
        read_strong_labels(os.path.join(metadata_dir, "audioset_train_strong.tsv"))
    )
    eval_ids = sorted(
        read_strong_labels(os.path.join(metadata_dir, "audioset_eval_strong.tsv"))
    )
    weak_index = _manifest_index(metadata_dir, set(train_ids) | set(eval_ids))
    train_files, train_rows = _resolve_partition(
        "train",
        train_ids,
        ("balance_wav", "unbalanced_wav"),
        raw_root,
        weak_index,
        fallback_source="valid_wav",
        fallback_partition="validation",
    )
    eval_files, eval_rows = _resolve_partition(
        "test",
        eval_ids,
        ("eval_wav",),
        raw_root,
        weak_index,
    )

    all_files = dict(train_files)
    all_files.update(eval_files)
    for source_dir, filenames in all_files.items():
        path = os.path.join(output_dir, source_dir + ".txt")
        with open(path, "w", encoding="utf-8") as handle:
            for filename in sorted(filenames):
                handle.write(filename + "\n")

    index_path = os.path.join(output_dir, "strong_subset_index.csv")
    with open(index_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["partition", "clip_id", "source_dir", "source_filename"])
        writer.writerows(train_rows)
        writer.writerows(eval_rows)
    print(f"Wrote rsync lists and index to {output_dir}")


if __name__ == "__main__":
    main()
