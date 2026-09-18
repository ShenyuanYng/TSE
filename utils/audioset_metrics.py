"""Epoch-level conditional frame accuracy; padding and unknown frames excluded."""
import math

import torch


def tensor(value):
    return value[0] if isinstance(value, (tuple, list)) else value


class ConditionalFrameAccuracy:
    """Count correct/valid frames, deduplicating clip-condition pairs across DDP."""

    def __init__(self, threshold=0.5):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("frame_accuracy_threshold must be between 0 and 1")
        self.threshold = float(threshold)
        self.rows = {} #初始化，临时仓库

    #每个batch调一次 统计当前 batch 中每个 sample 有多少帧预测正确，然后保存结果
    def update(self, predictions, targets):
        probabilities = predictions["frame_probs"].detach()#模型预测结果[B,T]
        labels = tensor(targets["strong_labels"]).to(probabilities.device)#strong label
        mask = tensor(targets["strong_mask"]).to(device=probabilities.device, dtype=torch.bool)
        mask = mask & predictions["frame_mask"].bool()
        if not torch.isfinite(probabilities[mask]).all():
            raise FloatingPointError("Nonfinite probabilities in supervised evaluation frames")

        correct = ((probabilities >= self.threshold) == (labels >= 0.5)) & mask
        # 统计每一个batch里面的sample中的正确帧和有效帧
        counts = torch.stack((correct.sum(dim=1), mask.sum(dim=1)), dim=1).cpu().tolist()
        for clip, category, pair in zip(targets["utt_id"], targets["class_index"], counts):
            self.rows[(clip, int(category))] = tuple(pair)
        """
        rows = {
(Audio_A, Dog):      (450,500),
(Audio_B, Speech):   (420,500),
(Audio_C, Dog):      (390,480),
(Audio_D, Car):      (470,500),
...
}
        """
    #epoch完成后计算accuracy
    def compute(self):
        rows = self.rows
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, rows)
            rows = {key: value for part in gathered for key, value in part.items()}
        correct = sum(pair[0] for pair in rows.values())#所有正确帧
        total = sum(pair[1] for pair in rows.values())#有效帧
        return {
            "conditional_frame_accuracy": correct / total if total else None,
            "correct_frames": correct,
            "valid_frames": total,
            "threshold": self.threshold,
        }

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
