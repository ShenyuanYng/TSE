"""Frame-level strong-label loss for AudioSet Model 1."""

from __future__ import annotations

import torch
import torch.nn.functional as F
"""
完成五步：
取出预测和强标签
  ->合并标注mask和padding mask
  ->逐元素计算BCE
  ->正负样本分别求平均
  ->返回用于反向传播的标量loss
"""

def _tensor(value, name: str) -> torch.Tensor:
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        value = value[0]
    if not torch.is_tensor(value):
        raise TypeError(f"Expected collated tensor for {name}, got {type(value).__name__}")
    return value


def _balanced_mean(element_loss, targets, mask):
    """Average positive and negative strong targets separately."""
    positive_mask = mask * (targets >= 0.5).to(mask.dtype)
    negative_mask = mask * (targets < 0.5).to(mask.dtype)
    terms = []
    if positive_mask.sum().item() > 0:
        terms.append((element_loss * positive_mask).sum() / positive_mask.sum())
    if negative_mask.sum().item() > 0:
        terms.append((element_loss * negative_mask).sum() / negative_mask.sum())
    if not terms:
        return element_loss.sum() * 0.0
    return torch.stack(terms).mean()


def audioset_strong_loss(
    predictions,
    target_dict,
    class_positive_weight: float = 1.0,
    balance_positive_negative: bool = True,
    **kwargs,
):
    
    del kwargs
    #检查模型输出是不是字典，必须是字典
    if not isinstance(predictions, dict):
        raise TypeError("AudioSetStrongLoss expects the model output dictionary")

    frame_logits = predictions["frame_logits"]
    
    strong_targets = _tensor(target_dict["strong_labels"], "strong_labels").to(
        device=frame_logits.device, dtype=frame_logits.dtype
    )
    #strong_mask：这个强标签位置是否可信，是否参与loss
    strong_mask = _tensor(target_dict["strong_mask"], "strong_mask").to(
        device=frame_logits.device, dtype=frame_logits.dtype
    )
    #三个必须完全同形[B,T,C]
    if frame_logits.shape != strong_targets.shape or frame_logits.shape != strong_mask.shape:
        raise ValueError(
            "frame prediction, strong target, and strong mask must all have shape [B,T,C]: "
            f"{tuple(frame_logits.shape)}, {tuple(strong_targets.shape)}, "
            f"{tuple(strong_mask.shape)}"
        )

    """
    frame_mask 是模型的输出之一，有效帧掩码[B, T] 标记哪些时间帧是真实音频，
    frame_mask[b,t] = True 表示第 b 个样本的第 t 帧不是 batch padding。

    strong_mask是这个强标签位置是否可信，是否参与loss

    unsqueeze(-1)先由[B,T]->[B,T,1],随后相乘然后通过广播规则把最后一维扩展到527
    """
    valid_frames = predictions.get("frame_mask")
    if valid_frames is not None:
        strong_mask = strong_mask * valid_frames.unsqueeze(-1).to(strong_mask.dtype)

    #创建正样本权重
    positive_weight = torch.as_tensor(
        class_positive_weight, device=frame_logits.device, dtype=frame_logits.dtype
    )


    """
    BCE 二元交叉熵  逐元素计算BCE
    """
    frame_bce = F.binary_cross_entropy_with_logits(
        frame_logits,#[B,T,C]
        strong_targets,#[B,T,C]
        reduction="none",
        pos_weight=positive_weight,#标量
    )#输出frame_bce：[B,T,C]
    if balance_positive_negative:
        strong_bce = _balanced_mean(frame_bce, strong_targets, strong_mask)
    elif strong_mask.sum().item() > 0:
        strong_bce = (frame_bce * strong_mask).sum() / strong_mask.sum()
    else:
        strong_bce = frame_logits.sum() * 0.0
    return {"loss": strong_bce, "strong_bce": strong_bce.detach()}
