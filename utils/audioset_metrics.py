"""Epoch-level conditional frame metrics; padding and unknown frames excluded."""
import math

import torch


def tensor(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def frame_metrics_from_counts(tp, fp, fn, tn):
    """Shared micro/per-class ratios; empty supervision is distinct from zero F1."""
    total = tp + fp + fn + tn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "correct_frames": tp + tn, "valid_frames": total,
        "accuracy": (tp + tn) / total if total else None,
        "precision": (tp / (tp + fp) if tp + fp else 0.0) if total else None,
        "recall": (tp / (tp + fn) if tp + fn else 0.0) if total else None,
        "f1": (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0) if total else None,
    }


class ConditionalFrameAccuracy:
    """Accumulate TP/FP/FN/TN, deduplicating clip-condition pairs across DDP."""

    def __init__(self, threshold=0.5):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("frame_accuracy_threshold must be between 0 and 1")
        self.threshold = float(threshold)
        self.rows = {} #初始化，临时仓库

    # Store only confusion counts per sample so DDP sampler repeats can be removed.
    def update(self, predictions, targets):
        probabilities = predictions["frame_probs"].detach()#模型预测结果[B,T]
        labels = tensor(targets["strong_labels"]).to(probabilities.device)#strong label
        mask = tensor(targets["strong_mask"]).to(device=probabilities.device, dtype=torch.bool)
        mask = mask & predictions["frame_mask"].bool()
        if not torch.isfinite(probabilities[mask]).all():
            raise FloatingPointError("Nonfinite probabilities in supervised evaluation frames")

        positive = probabilities >= self.threshold
        actual_positive = labels >= 0.5
        counts = torch.stack([
            (positive & actual_positive & mask).sum(dim=1),
            (positive & ~actual_positive & mask).sum(dim=1),
            (~positive & actual_positive & mask).sum(dim=1),
            (~positive & ~actual_positive & mask).sum(dim=1),
        ], dim=1).cpu().tolist()
        for clip, category, pair in zip(targets["utt_id"], targets["class_index"], counts):
            self.rows[(clip, int(category))] = tuple(pair)

    # Compute ratios once from the entire deduplicated epoch, never per batch.
    def compute(self, include_per_class=False):
        rows = self.rows
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, rows)
            rows = {key: value for part in gathered for key, value in part.items()}
        tp, fp, fn, tn = (sum(row[i] for row in rows.values()) for i in range(4))
        micro = frame_metrics_from_counts(tp, fp, fn, tn)
        result = {
            **{f"conditional_frame_{name}": micro[name]
               for name in ("accuracy", "precision", "recall", "f1")},
            **{name: micro[name] for name in
               ("tp", "fp", "fn", "tn", "correct_frames", "valid_frames")},
            "threshold": self.threshold,
        }
        if include_per_class:
            # Group only AFTER the existing cross-rank clip-condition deduplication.
            by_class = {}
            for (_, category), counts in rows.items():
                totals = by_class.setdefault(category, [0, 0, 0, 0])
                for index, count in enumerate(counts):
                    totals[index] += count
            per_class = [
                {"class_index": category, **frame_metrics_from_counts(*counts)}
                for category, counts in sorted(by_class.items())
            ]
            valid = [row for row in per_class if row["valid_frames"] > 0]
            result.update(
                per_class=per_class,
                macro_f1=sum(row["f1"] for row in valid) / len(valid) if valid else None,
                valid_class_count=len(valid),
            )
        return result

    def reset(self):
        self.rows.clear()


# 原 mAP 实现暂时注释保留，不参与训练、验证或测试。
# """Exact epoch-level conditional frame AP; padding and unlabelled frames excluded."""
# import numpy as np
# import torch
#
# """
# 计算模型评估指标mAP
# """
#
# #根据真实标签label和模型预测概率，计算某一个类别的AP
# def average_precision(labels, scores):
#     labels = np.asarray(labels, dtype=np.float64)
#     scores = np.asarray(scores)
#     positives = labels.sum()#正样本总数 P=TP+FN
#     if positives == 0:
#         return float("nan")
#     #排序
#     order = np.argsort(-scores, kind="stable")
#     labels, scores = labels[order], scores[order]
#     ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
#     true_positives = np.cumsum(labels)[ends]#TP
#     precision = true_positives / (ends + 1)
#     return float((np.diff(np.r_[0, true_positives]) * precision).sum() / positives)
#
#
# class ConditionalFrameMAP:
#     def __init__(self):
#         self.rows = {}#初始化，临时仓库
#
#     #每跑一个batch，就把frame_probs 模型预测概率  strong_labels gd strong_mask 哪些帧有效  保存下来
#     def update(self, predictions, targets):
#         probabilities = predictions["frame_probs"].detach().cpu().numpy()
#         labels = tensor(targets["strong_labels"]).detach().cpu().numpy()
#         mask = tensor(targets["strong_mask"]).detach().cpu().bool().numpy()
#         mask &= predictions["frame_mask"].detach().cpu().numpy()
#         for i, (clip, category) in enumerate(zip(targets["utt_id"], targets["class_index"])):
#             # DDP's sampler can repeat examples. Key by clip and condition to deduplicate.
#             self.rows[(clip, int(category))] = (labels[i][mask[i]], probabilities[i][mask[i]])
#             """
#             rows = {("Audio_A", Dog):([0,0,1,1,0],[0.1,0.2,0.9,0.8,0.1)}
#             下一个batch继续往里面加
#             """
#
#     #一个epoch/整个验证集跑完之后，计算平均map
#     def compute(self):
#         rows = self.rows
#         if torch.distributed.is_available() and torch.distributed.is_initialized():
#             gathered = [None] * torch.distributed.get_world_size()
#             torch.distributed.all_gather_object(gathered, rows)
#             rows = {key: value for part in gathered for key, value in part.items()}
#         #把之前的(clip, category) 按 category 重新分组。
#         by_class = {}
#         for (_, category), pair in rows.items():
#             by_class.setdefault(category, []).append(pair)
#         """
#         by_class:
#         Dog:
#             Audio_A
#             Audio_C
#         Speech:
#             Audio_B
#             Audio_E
#         Car:
#             Audio_D
#         """
#         aps = {}
#         #把同一类别所有音频的frame全部拼起来
#         for category, pairs in by_class.items():
#             labels = np.concatenate([p[0] for p in pairs])#gd
#             scores = np.concatenate([p[1] for p in pairs])#模型预测值
#             ap = average_precision(labels, scores)
#             if np.isfinite(ap):
#                 aps[category] = ap
#         return float(np.mean(list(aps.values()))) if aps else float("nan"), aps
#
#     #一轮评估结束以后，把之前积累的预测结果清空，为下一轮 validation 做准备。
#     def reset(self):
#         self.rows.clear()
