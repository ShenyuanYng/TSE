"""Sample weak/strong MID-matched targets from a new-split checkpoint and report predictions."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from creators.dataset import datapath_creator, dataset_creator
from datasets.paths.audioset_paths import resolve_path
from datasets.train_split import validate_split_config
from inspect_prediction import (_load_checkpoint, _move_to_device, _intervals,
                                _safe_div, _write_report, _export_audio_comparison)
from utils.collate import collate_seq


def matched_indices(dataset):
    """A target is (clip, weak MID) with a same-MID, nonempty PRESENT event."""
    return [index for index, (entry_index, mid) in enumerate(dataset.samples)
            if any(event["mid"] == mid and event.get("present", True)
                   and event["offset"] > event["onset"]
                   for event in dataset.path_list[entry_index].get("events", []))]


def select_indices(dataset, count, seed):
    candidates = matched_indices(dataset)
    # Stable across manifest order; choose before inference, never by model score.
    candidates.sort(key=lambda i: (dataset.path_list[dataset.samples[i][0]]["id"],
                                   dataset.samples[i][1]))
    selected = random.Random(seed).sample(candidates, min(count, len(candidates)))
    return candidates, selected


def frame_metrics(labels, probabilities, valid, threshold):
    if not np.any(valid):
        raise ValueError("No supervised frames")
    if not np.isfinite(probabilities[valid]).all():
        raise FloatingPointError("Nonfinite probabilities in valid frames")
    truth, predicted = labels >= .5, probabilities >= threshold
    tp = int(np.sum(truth & predicted & valid))
    fp = int(np.sum(~truth & predicted & valid))
    fn = int(np.sum(truth & ~predicted & valid))
    tn = int(np.sum(~truth & ~predicted & valid))
    counts = dict(tp=tp, fp=fp, fn=fn, tn=tn)
    metrics = {"Accuracy": _safe_div(tp + tn, tp + fp + fn + tn),
               "Precision": _safe_div(tp, tp + fp), "Recall": _safe_div(tp, tp + fn),
               "F1": _safe_div(2 * tp, 2 * tp + fp + fn),
               "Frame IoU": _safe_div(tp, tp + fp + fn)}
    return metrics, counts


def _finite_metrics(metrics):
    return {key: value if math.isfinite(value) else None for key, value in metrics.items()}


def _write_index(output, summary, rows):
    def cell(value):
        if value is None:
            return "N/A"
        return html.escape(f"{value:.4f}" if isinstance(value, float) else str(value))

    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell(row[key])}</td>" for key in
                    ("partition", "clip_id", "class_name", "Accuracy", "Precision", "Recall", "F1", "Frame IoU"))
                    + f'<td><a href="{row["report"]}">打开对比</a></td></tr>')
    partition_text = "".join(
        f'<li>{part}：全部 target {stats["total_targets"]:,}；同 MID 匹配 '
        f'{stats["matched_targets"]:,}；抽取 {stats["selected_targets"]}；'
        f'成功 {stats["reported_targets"]}；跳过 {len(stats["skipped"])}。</li>'
        for part, stats in summary["partitions"].items())
    (output / "index.html").write_text(f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>强弱标签匹配样本：预测对比</title>
<style>body{{font:16px system-ui;color:#17202a;max-width:1280px;margin:32px auto;padding:0 20px}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border:1px solid #d9dee5;text-align:left}}
th{{background:#f5f7fa}}code{{overflow-wrap:anywhere}}li{{margin:8px 0}}</style>
<h1>强弱标签同 MID 匹配样本：预测与真实强标签</h1>
<p>模型：<code>{html.escape(summary["checkpoint"])}</code></p>
<p>seed={summary["seed"]}；阈值={summary["threshold"]}。
从弱标签 MID 有同 MID 的 PRESENT 强标签的 target 中均匀随机抽样，未按预测效果筛选。</p>
<ul>{partition_text}</ul>
<p>tr 为训练集诊断，cv 为验证集诊断。只包含匹配正例，不能代表完整数据集，
也不能衡量不存在目标类别时的拒绝能力；同 MID 匹配不等于人工核实。</p>
<p>每个页面包含概率曲线、真实区间、阈值生成的伪强标签区间及逐帧 CSV。
每个样本在 audio/ 中有 original、predicted、ground_truth 三条对齐 WAV，
将同一样本的三条音频一起导入 Audacity，可比较波形和频谱；区间外静音，不拼接。
人工原始标注见 <a href="ground_truth.tsv">ground_truth.tsv</a>；
预测区间见 <a href="pseudo_strong.tsv">pseudo_strong.tsv</a>；
统计和跳过原因见 <a href="summary.json">summary.json</a>。</p>
<table><thead><tr><th>集合</th><th>音频 ID</th><th>目标类别</th><th>Accuracy</th><th>Precision</th>
<th>Recall</th><th>F1</th><th>帧 IoU</th><th>报告</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></html>''', encoding="utf-8")


def run(checkpoint, output, partitions=("tr", "cv"), num_samples=20, seed=7,
        threshold=None, device="cpu", copy_audio=False):
    # Retain the old argument for existing commands; aligned WAVs are now always exported.
    del copy_audio
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if not partitions or any(part not in ("tr", "cv") for part in partitions):
        raise ValueError("Only tr/cv diagnostics are supported")
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    module, hp = _load_checkpoint(checkpoint, torch.device(device))
    validate_split_config(hp)
    split_path = Path(resolve_path(hp.train_split_path, hp.data_path))
    if not split_path.is_file():
        raise FileNotFoundError(f"Training split file is required; do not regenerate it: {split_path}")
    threshold = hp.frame_accuracy_threshold if threshold is None else threshold
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    # Read whole clips deterministically, including when inspecting train samples.
    hp = copy.deepcopy(hp)
    hp.train_on_chunk = None
    summary = dict(checkpoint=str(checkpoint), seed=seed, threshold=threshold,
                   selection="uniform random matched clip-MID targets, before prediction",
                   split_file=str(split_path),
                   split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                   audio_export="Original amplitude and channels; predictions thresholded on the frame grid; "
                                "ground truth gated by raw annotation times; silence outside intervals",
                   partitions={})
    rows, pseudo, truth_rows = [], [], []
    for part in dict.fromkeys(partitions):
        entries = datapath_creator(hp, part)
        dataset = dataset_creator(hp, part, path_list=entries, is_training=False)
        candidates, selected = select_indices(dataset, num_samples, seed)
        stats = dict(total_targets=len(dataset), matched_targets=len(candidates),
                     selected_targets=len(selected), reported_targets=0, skipped=[])
        summary["partitions"][part] = stats
        print(f"{part}: total={len(dataset)}, matched={len(candidates)}, selected={len(selected)}", flush=True)
        for rank, index in enumerate(selected, 1):
            entry_index, mid = dataset.samples[index]
            entry = entries[entry_index]
            inputs, target = example = dataset[index]
            duration = sf.info(target["audio_path"]).duration
            times, labels = np.asarray(target["frame_times"]), np.asarray(target["strong_labels"])
            if not np.any((labels >= .5) & np.asarray(target["strong_mask"], dtype=bool)):
                stats["skipped"].append(dict(clip_id=entry["id"], mid=mid,
                                             reason="No PRESENT supervised frame after time alignment"))
                continue
            model_input, _ = collate_seq([example])
            with torch.inference_mode():
                prediction = module.model(_move_to_device(model_input, torch.device(device)))
            probabilities = prediction["frame_probs"][0].cpu().numpy()
            valid = (np.asarray(target["strong_mask"], dtype=bool)
                     & prediction["frame_mask"][0].cpu().numpy().astype(bool))
            metrics, counts = frame_metrics(labels, probabilities, valid, threshold)
            true_intervals = _intervals((labels >= .5) & valid, times, hp.hoptime, duration)
            audio_valid = prediction["frame_mask"][0].cpu().numpy().astype(bool) & (times < duration)
            if not np.isfinite(probabilities[audio_valid]).all():
                raise FloatingPointError("Nonfinite probabilities in audio frames")
            pred_intervals = _intervals((probabilities >= threshold) & audio_valid,
                                        times, hp.hoptime, duration)
            name = f"{part}_{rank:03d}_{metrics['Accuracy']:.2f}"
            class_name = dataset.class_names[int(target["class_index"])]
            metadata = dict(checkpoint=str(checkpoint), partition=part, sample_index=index,
                            clip_id=entry["id"], audio=target["audio_path"], class_mid=mid,
                            class_name=class_name, threshold=threshold,
                            match_rule="weak MID == PRESENT strong MID（未人工核实）",
                            valid_frames=int(valid.sum()), positive_frames=int(((labels >= .5) & valid).sum()),
                            audio_duration_seconds=duration)
            audio_files = _export_audio_comparison(output / f"{name}.html", target["audio_path"],
                                                   pred_intervals, entry.get("events", []), mid)
            _write_report(output / f"{name}.html", metadata, metrics, times, labels,
                          probabilities, valid, true_intervals, pred_intervals, threshold,
                          end_time=duration, audio_files=audio_files)
            rows.append(dict(partition=part, sample_index=index, clip_id=entry["id"],
                             mid=mid, class_name=class_name, report=f"{name}.html",
                             audio_files=audio_files,
                             **counts, **_finite_metrics(metrics)))
            pseudo.extend((part, entry["id"], start, end, mid) for start, end in pred_intervals)
            truth_rows.extend((part, entry["id"], event["onset"], event["offset"], mid,
                               "PRESENT" if event.get("present", True) else "NOT_PRESENT")
                              for event in entry["events"] if event["mid"] == mid)
            stats["reported_targets"] += 1
            print(f"  {name}: {entry['id']} {mid} F1={metrics['F1']:.4f}", flush=True)
    for filename, values, columns in (
            ("pseudo_strong.tsv", pseudo, ["partition", "clip_id", "onset", "offset", "mid"]),
            ("ground_truth.tsv", truth_rows, ["partition", "clip_id", "onset", "offset", "mid", "status"])):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(columns)
            writer.writerows(values)
    summary["samples"] = rows
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                  allow_nan=False), encoding="utf-8")
    _write_index(output, summary, rows)
    print(f"Wrote {output / 'index.html'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", nargs="+", choices=("tr", "cv"), default=["tr", "cv"])
    parser.add_argument("--num-samples", type=int, default=20, help="Targets sampled per partition")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, help="Default: checkpoint frame_accuracy_threshold")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--copy-audio", action="store_true",
                        help="Accepted for compatibility; three aligned WAVs are always exported")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.checkpoint, args.output_dir, args.partition, args.num_samples, args.seed,
        args.threshold, args.device, args.copy_audio)


if __name__ == "__main__":
    main()
