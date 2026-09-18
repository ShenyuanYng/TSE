"""Persist a video-grouped eval split before expanding class conditions."""
import hashlib
import json
import os
import tempfile
from pathlib import Path
"""
划分验证集和测试集
"""

def video_id(entry):
    return entry.get("ytid") or entry["id"].rsplit("_", 1)[0]


def split_eval(entries, data_root, split_path, fraction=0.5, seed=7):
    if not 0 < fraction < 1:
        raise ValueError("eval_test_fraction must be between 0 and 1")
    path = Path(split_path)
    if not path.is_absolute():
        path = Path(data_root) / path
    ids = sorted(e["id"] for e in entries)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate clip IDs in eval")
    groups = {}
    for entry in entries:
        groups.setdefault(video_id(entry), []).append(entry)
    if len(groups) < 2:
        raise ValueError("Eval requires at least two independent videos")
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["seed"] != seed or saved["test_fraction"] != fraction:
            raise ValueError("Split settings changed; use a new eval_split_path")
        if sorted(saved["cv"] + saved["test"]) != ids:
            raise ValueError("Eval inventory changed; use a new eval_split_path")
    else:
        # Deterministic grouped multilabel balancing of condition distributions.
        import numpy as np
        labels = sorted({m for e in entries for m in e.get("weak_mids", [])})
        lookup = {m: i for i, m in enumerate(labels)}
        vectors = {}
        for key, group in groups.items():
            vector = np.zeros(len(labels) + 1)
            for entry in group:
                for mid in set(entry.get("weak_mids", [])):
                    vector[lookup[mid]] += 1
                vector[-1] += 1
            vectors[key] = vector
        total = sum(vectors.values())
        target = total * fraction
        scale = np.maximum(total, 1)
        ordered = sorted(groups, key=lambda k: (
            -float((vectors[k][:-1] / scale[:-1]).sum()),
            hashlib.sha256(f"{seed}:{k}".encode()).hexdigest()))
        current = np.zeros_like(total)
        test_groups = set()
        for key in ordered:
            before = (((current - target) / scale) ** 2).sum()
            after = (((current + vectors[key] - target) / scale) ** 2).sum()
            if after < before:
                test_groups.add(key)
                current += vectors[key]
        if not test_groups:
            test_groups.add(ordered[0])
        if len(test_groups) == len(groups):
            test_groups.remove(ordered[-1])
        saved = {"seed": seed, "test_fraction": fraction, "cv": [], "test": []}
        for entry in entries:
            saved["test" if video_id(entry) in test_groups else "cv"].append(entry["id"])
        for name in ("cv", "test"):
            saved[name].sort()
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            json.dump(saved, handle, indent=2)
            temporary = handle.name
        os.replace(temporary, path)
    cv_ids, test_ids = set(saved["cv"]), set(saved["test"])
    if cv_ids & test_ids or not cv_ids or not test_ids:
        raise ValueError("Invalid overlapping or empty eval split")
    result = {"cv": [e for e in entries if e["id"] in cv_ids],
              "test": [e for e in entries if e["id"] in test_ids]}
    if {video_id(e) for e in result["cv"]} & {video_id(e) for e in result["test"]}:
        raise ValueError("A video occurs in both validation and test")
    return result
