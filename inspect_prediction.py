"""Compare one checkpoint prediction with its frame-level strong annotation."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path

import numpy as np
import soundfile as sf
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


def _intervals(active: np.ndarray, times: np.ndarray, hop_seconds: float, end_time=None):
    indices = np.flatnonzero(active)
    if not len(indices):
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(indices) - 1]
    return [(float(times[indices[start]]),
             min(float(times[indices[end]] + hop_seconds),
                 float(end_time) if end_time is not None else float("inf")))
            for start, end in zip(starts, ends)]


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def _audio_interval_mask(intervals, frames, sample_rate, sample_aligned=False):
    """Map half-open intervals to samples without shifting or shortening audio."""
    mask = np.zeros(frames, dtype=bool)
    for onset, offset in intervals:
        if not (math.isfinite(onset) and math.isfinite(offset)) or offset <= onset:
            raise ValueError(f"Invalid audio interval: {onset}, {offset}")
        # Predicted frame timestamps originate on the sample grid but are float32.
        # Raw annotations may lie between samples and use the half-open convention.
        convert = round if sample_aligned else lambda value: math.ceil(value - 1e-7)
        start = min(frames, max(0, convert(onset * sample_rate)))
        stop = min(frames, max(0, convert(offset * sample_rate)))
        mask[start:stop] = True
    return mask


def _export_audio_comparison(report_path, source_path, predicted_intervals, events, mid):
    """Export aligned raw-amplitude WAVs; this is time gating, not separation."""
    report_path = Path(report_path)
    waveform, sr = sf.read(source_path, dtype="float32", always_2d=True)
    predicted = _audio_interval_mask(predicted_intervals, len(waveform), sr, sample_aligned=True)
    positive, negative = [], []
    for event in events:
        if event["mid"] == mid:
            intervals = positive if event.get("present", True) else negative
            intervals.append((event["onset"], event["offset"]))
    truth = _audio_interval_mask(positive, len(waveform), sr)
    truth &= ~_audio_interval_mask(negative, len(waveform), sr)
    (report_path.parent / "audio").mkdir(parents=True, exist_ok=True)
    urls = {}
    for role, keep in (("original", None), ("predicted", predicted), ("ground_truth", truth)):
        relative = f"audio/{report_path.stem}_{role}.wav"
        # Float WAV avoids additional PCM quantization or clipping after decoding.
        audio = waveform if keep is None else np.where(keep[:, None], waveform, 0.0)
        sf.write(report_path.parent / relative, audio, sr, subtype="FLOAT")
        urls[role] = relative
    return urls


def _format_number(value):
    return "N/A" if not math.isfinite(value) else f"{value:.4f}"


def _format_intervals(intervals, limit=30):
    shown = intervals[:limit]
    text = ", ".join(f"{start:.3f}–{end:.3f}s" for start, end in shown)
    if len(intervals) > limit:
        text += f", …（另有 {len(intervals) - limit} 段）"
    return text or "无"


def _svg(times, probabilities, valid, truth_intervals, predicted_intervals, threshold,
         end_time=None):
    width, height = 1100, 420
    left, right, top, bottom = 85, 30, 35, 115
    plot_width, plot_height = width - left - right, height - top - bottom
    valid_times = times[valid]
    start = 0.0 if end_time is not None else float(valid_times[0])
    end = float(end_time) if end_time is not None else max(
        [float(valid_times[-1])] + [offset for _, offset in truth_intervals + predicted_intervals])
    if end <= start:
        end = start + 1.0

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
                  truth_intervals, predicted_intervals, threshold, audio_url=None,
                  end_time=None, audio_files=None):
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
    graphic = _svg(times, probabilities, valid, truth_intervals, predicted_intervals, threshold,
                   end_time=end_time)
    audio_player = (f'<audio controls preload="none" src="{html.escape(audio_url, quote=True)}"></audio>'
                    if audio_url else "")
    if audio_files:
        audio_player = '<h2>对齐音频（可一起导入 Audacity）</h2>'
        for role, title in (("original", "原始音频"), ("predicted", "预测区间保留原声"),
                            ("ground_truth", "真实标注区间保留原声")):
            url = html.escape(audio_files[role], quote=True)
            audio_player += (f'<p>{title}：<a href="{url}" download>下载 WAV</a></p>'
                             f'<audio controls preload="none" src="{url}"></audio>')
        audio_player += ('<p>三条音频长度、采样率、声道和时间起点一致，保留原始幅度。'
                         '预测音频按阈值区间保留，真实标注音频按原始标注秒数保留，其余位置静音。'
                         '没有拼接、归一化或淡入淡出；保留区间仍包含原音频中的其他声音。'
                         '下方真实标签曲线按特征帧对齐，与原始标注边界可能有不足一帧的差异。</p>')
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
{audio_player}
<h2>时间对比</h2><div class="plot">{graphic}</div>
<p>蓝线：模型逐帧概率；绿色背景与“真实”条：强标签；“预测”条：概率 ≥ {threshold:.2f}。</p>
<h2>单样本指标</h2><table>{metric_rows}</table>
<p>区间为有效监督帧的时间区间；预测区间使用当前阈值，未做平滑或短事件过滤。
Accuracy 容易受大量负帧影响，需结合目标正帧的 Precision、Recall、F1 和时间曲线判断。
同 MID 匹配不代表已经人工核实音频或标注正确。</p>
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
    duration = sf.info(target["audio_path"]).duration
    truth_intervals = _intervals(positive & valid, times, hop_seconds, duration)
    # Export predictions independently of annotation availability.
    audio_valid = frame_mask & (times < duration)
    if not np.isfinite(probabilities[audio_valid]).all():
        raise FloatingPointError("Nonfinite probabilities in audio frames")
    predicted_intervals = _intervals(binary & audio_valid,
                                     times, hop_seconds, duration)
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
    entry_index, mid = dataset.samples[index]
    audio_files = _export_audio_comparison(args.output.resolve(), target["audio_path"],
                                           predicted_intervals,
                                           dataset.path_list[entry_index].get("events", []), mid)
    csv_path = _write_report(args.output.resolve(), metadata, metrics, times, labels,
                             probabilities, valid, truth_intervals, predicted_intervals,
                             args.threshold, end_time=duration, audio_files=audio_files)
    print(f"Wrote {args.output.resolve()}")
    print(f"Wrote {csv_path}")
    print(" | ".join(f"{name}={_format_number(value)}" for name, value in metrics.items()))


if __name__ == "__main__":
    main()
