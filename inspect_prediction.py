"""Compare one checkpoint prediction with its frame-level strong annotation."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path

import numpy as np
import torch

from creators.dataset import datapath_creator, dataset_creator
from source_separator import SourceSeparator
# from utils.audioset_metrics import average_precision  # AP 暂停使用
from utils.collate import collate_seq


def _load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "hyper_parameters" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(f"Not a Lightning training checkpoint: {path}")
    hparams = argparse.Namespace(**dict(checkpoint["hyper_parameters"]))
    module = SourceSeparator(hparams)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module.to(device).eval()
    return module, hparams


def _move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _select_sample(dataset, sample_index, clip_id, class_mid):
    if sample_index is not None:
        if not 0 <= sample_index < len(dataset):
            raise IndexError(f"sample-index must be in [0, {len(dataset) - 1}]")
        entry_index, mid = dataset.samples[sample_index]
        entry = dataset.path_list[entry_index]
        if clip_id is not None and entry["id"] != clip_id:
            raise ValueError("sample-index and clip-id select different samples")
        if class_mid is not None and mid != class_mid:
            raise ValueError("sample-index and class-mid select different samples")
        return sample_index

    matches = []
    positive_matches = []
    for index, (entry_index, mid) in enumerate(dataset.samples):
        entry = dataset.path_list[entry_index]
        if clip_id is not None and entry["id"] != clip_id:
            continue
        if class_mid is not None and mid != class_mid:
            continue
        matches.append(index)
        if any(event["mid"] == mid and event.get("present", True)
               for event in entry.get("events", [])):
            positive_matches.append(index)
    if positive_matches:
        return positive_matches[0]
    if matches:
        return matches[0]
    filters = f"clip_id={clip_id!r}, class_mid={class_mid!r}"
    raise ValueError(f"No dataset sample matches {filters}")


def _intervals(active: np.ndarray, times: np.ndarray, hop_seconds: float):
    indices = np.flatnonzero(active)
    if not len(indices):
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(indices) - 1]
    return [(float(times[indices[start]]), float(times[indices[end]] + hop_seconds))
            for start, end in zip(starts, ends)]


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def _format_number(value):
    return "N/A" if not math.isfinite(value) else f"{value:.4f}"


def _format_intervals(intervals, limit=30):
    shown = intervals[:limit]
    text = ", ".join(f"{start:.3f}–{end:.3f}s" for start, end in shown)
    if len(intervals) > limit:
        text += f", …（另有 {len(intervals) - limit} 段）"
    return text or "无"


def _svg(times, probabilities, valid, truth_intervals, predicted_intervals, threshold):
    width, height = 1100, 420
    left, right, top, bottom = 85, 30, 35, 115
    plot_width, plot_height = width - left - right, height - top - bottom
    valid_times = times[valid]
    start = float(valid_times[0])
    end = float(valid_times[-1]) if valid_times[-1] > start else start + 1.0

    def x(value):
        return left + (value - start) / (end - start) * plot_width

    def y(value):
        return top + (1.0 - value) * plot_height

    background = []
    for onset, offset in truth_intervals:
        background.append(
            f'<rect x="{x(onset):.2f}" y="{top}" width="{max(1, x(offset)-x(onset)):.2f}" '
            f'height="{plot_height}" fill="#7bc67b" opacity="0.24"/>'
        )

    indices = np.flatnonzero(valid)
    stride = max(1, math.ceil(len(indices) / 2500))
    selected = indices[::stride]
    if selected[-1] != indices[-1]:
        selected = np.r_[selected, indices[-1]]
    points = " ".join(f"{x(times[i]):.2f},{y(probabilities[i]):.2f}" for i in selected)

    ticks = []
    for value in np.linspace(start, end, 6):
        xpos = x(value)
        ticks.append(f'<line x1="{xpos:.2f}" y1="{top+plot_height}" x2="{xpos:.2f}" '
                     f'y2="{top+plot_height+6}" stroke="#555"/>')
        ticks.append(f'<text x="{xpos:.2f}" y="{top+plot_height+24}" text-anchor="middle" '
                     f'font-size="13">{value:.2f}s</text>')

    truth_blocks = "".join(
        f'<rect x="{x(onset):.2f}" y="{height-68}" width="{max(1, x(offset)-x(onset)):.2f}" '
        'height="15" fill="#268a43"/>' for onset, offset in truth_intervals
    )
    prediction_blocks = "".join(
        f'<rect x="{x(onset):.2f}" y="{height-39}" width="{max(1, x(offset)-x(onset)):.2f}" '
        'height="15" fill="#2764c4"/>' for onset, offset in predicted_intervals
    )
    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Prediction and ground truth">
  <rect width="{width}" height="{height}" fill="white"/>
  {''.join(background)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#555"/>
  <line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#555"/>
  <line x1="{left}" y1="{y(threshold):.2f}" x2="{width-right}" y2="{y(threshold):.2f}"
        stroke="#d14" stroke-dasharray="7 6"/>
  <polyline points="{points}" fill="none" stroke="#2764c4" stroke-width="2"/>
  <text x="{left-14}" y="{y(1)+5:.2f}" text-anchor="end" font-size="13">1.0</text>
  <text x="{left-14}" y="{y(threshold)+5:.2f}" text-anchor="end" font-size="13">{threshold:.2f}</text>
  <text x="{left-14}" y="{y(0)+5:.2f}" text-anchor="end" font-size="13">0.0</text>
  <text x="20" y="{top+plot_height/2:.2f}" transform="rotate(-90 20 {top+plot_height/2:.2f})"
        text-anchor="middle" font-size="14">预测概率</text>
  {''.join(ticks)}
  <text x="{left-12}" y="{height-56}" text-anchor="end" font-size="13">真实</text>
  <text x="{left-12}" y="{height-27}" text-anchor="end" font-size="13">预测</text>
  {truth_blocks}
  {prediction_blocks}
</svg>
"""


def _write_report(output_path, metadata, metrics, times, labels, probabilities, valid,
                  truth_intervals, predicted_intervals, threshold):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_seconds", "strong_label", "prediction_probability", "valid"])
        writer.writerows(zip(times, labels.astype(int), probabilities, valid.astype(int)))

    rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in metadata.items()
    )
    metric_rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{_format_number(value)}</td></tr>"
        for name, value in metrics.items()
    )
    graphic = _svg(times, probabilities, valid, truth_intervals, predicted_intervals, threshold)
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>AudioSet 单样本预测对比</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;color:#17202a}}
h1{{font-size:26px}} h2{{margin-top:30px}} table{{border-collapse:collapse;min-width:520px}}
th,td{{border:1px solid #d9dee5;padding:8px 12px;text-align:left}} th{{background:#f5f7fa}}
.plot{{border:1px solid #d9dee5;border-radius:8px;overflow:hidden}} code{{background:#f5f7fa;padding:2px 5px}}
</style></head><body>
<h1>AudioSet 单样本预测与真实强标签</h1>
<table>{rows}</table>
<h2>时间对比</h2><div class="plot">{graphic}</div>
<p>蓝线：模型逐帧概率；绿色背景与“真实”条：强标签；“预测”条：概率 ≥ {threshold:.2f}。</p>
<h2>单样本指标</h2><table>{metric_rows}</table>
<p>Accuracy 容易受大量负帧影响；判断定位质量时优先看 AP、Precision、Recall、F1 和时间曲线。</p>
<h2>时间区间</h2>
<p><strong>真实：</strong>{html.escape(_format_intervals(truth_intervals))}</p>
<p><strong>预测：</strong>{html.escape(_format_intervals(predicted_intervals))}</p>
<p>逐帧数据：<code>{html.escape(csv_path.name)}</code></p>
</body></html>"""
    output_path.write_text(document, encoding="utf-8")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Best or last Lightning .ckpt file")
    parser.add_argument("--partition", choices=("cv", "test"), default="cv")
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--clip-id", help="Exact AudioSet clip id")
    parser.add_argument("--class-mid", help="Exact AudioSet MID, for example /m/01b_21")
    parser.add_argument("--threshold", type=float,
                        help="Override the checkpoint's frame_accuracy_threshold")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("prediction_comparison.html"))
    args = parser.parse_args()
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    module, hparams = _load_checkpoint(args.checkpoint.resolve(), device)
    if args.threshold is None:
        args.threshold = hparams.frame_accuracy_threshold
    entries = datapath_creator(hparams, args.partition)
    dataset = dataset_creator(hparams, args.partition, path_list=entries, is_training=False)
    index = _select_sample(dataset, args.sample_index, args.clip_id, args.class_mid)
    example = dataset[index]
    model_input, _ = collate_seq([example])
    model_input = _move_to_device(model_input, device)
    with torch.inference_mode():
        prediction = module.model(model_input, target_dict=None, current_epoch=0)

    _, target = example
    probabilities = prediction["frame_probs"][0].detach().cpu().numpy()
    frame_mask = prediction["frame_mask"][0].detach().cpu().numpy().astype(bool)
    times = np.asarray(target["frame_times"])
    labels = np.asarray(target["strong_labels"])
    valid = np.asarray(target["strong_mask"]).astype(bool) & frame_mask
    binary = probabilities >= args.threshold
    positive = labels >= 0.5
    if not np.isfinite(probabilities[valid]).all():
        raise FloatingPointError("Nonfinite probabilities in supervised evaluation frames")
    correct = int(np.sum((binary == positive) & valid))
    metrics = {
        # "AP": average_precision(labels[valid], probabilities[valid]),
        "Accuracy": _safe_div(correct, int(valid.sum())),
    }
    hop_seconds = float(hparams.hoptime)
    truth_intervals = _intervals(positive & valid, times, hop_seconds)
    predicted_intervals = _intervals(binary & valid, times, hop_seconds)
    class_index = int(target["class_index"])
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "partition": args.partition,
        "sample_index": index,
        "clip_id": target["utt_id"],
        "audio": target["audio_path"],
        "class_mid": dataset.samples[index][1],
        "class_index": class_index,
        "class_name": dataset.class_names[class_index],
        "valid_frames": int(valid.sum()),
        "positive_frames": int(np.sum(positive & valid)),
        "threshold": args.threshold,
        "device": str(device),
    }
    csv_path = _write_report(args.output.resolve(), metadata, metrics, times, labels,
                             probabilities, valid, truth_intervals, predicted_intervals,
                             args.threshold)
    print(f"Wrote {args.output.resolve()}")
    print(f"Wrote {csv_path}")
    print(" | ".join(f"{name}={_format_number(value)}" for name, value in metrics.items()))


if __name__ == "__main__":
    main()
