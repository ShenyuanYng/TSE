"""Weak-label-conditioned Model 1 for AudioSet Strong."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable, Optional

import torch
import torch.nn as nn

#辅助函数用于兼容两种输入形式
def _collated_tensor(value):
    """Extract the padded tensor from this repository's ``collate_seq`` output."""
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    return value


class ConvFrequencyBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
            #输入 卷积 归一化 SiLU激活 频率池化  dropout
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2), ceil_mode=True),# 1✖️2表示 时间轴T不压缩，F压缩一半
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AudioSetCRNN(nn.Module):
    """Predict strong frame labels from audio and its clip-level weak label.

    Model inputs are an RI spectrogram ``[B,T,1,M,2,F]`` and weak labels
    ``[B,C]``. The weak-label embedding is repeated over time and fused with
    the audio representation before the frame classifier. The only learned
    output is the strong prediction ``[B,T,C]``; there is no clip pooling,
    pseudo-label generation, or Model 2 path.
    """

    def __init__(
        self,
        num_classes: int = 527,
        num_imics: int = 1,
        conv_channels: Iterable[int] = (32, 64, 96),
        temporal_hidden: int = 128,#GRU每个方向隐藏维度128
        rnn_layers: int = 2,#两层GRU
        dropout: float = 0.2,#随机丢弃20％特征
    ) -> None:
        super().__init__()
        channels = [int(value) for value in conv_channels]
        if not channels:
            raise ValueError("conv_channels must contain at least one value")

        blocks = []
        in_channels = num_imics * 2
        """
        创建CNN，目的是增加特征通道数，并逐渐减少频率位置的数量  循环创建了三个不同参数的卷积块，把他们串联成一个encoder
        [B,2,T,F]->[B,32,T,F/2]->[B,64,T,F/4]->[B,96,T,F/8]  这96个特征是训练过程中模型自己学出来的
        """
        for out_channels in channels:  #channels=[32,64,96]  总共循环三次
            blocks.append(ConvFrequencyBlock(in_channels, out_channels, dropout))#（2,32,0.2）
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)


        
        """
        BiGUR
        输入：[B,T,96]

       正向 GRU → 128
      /
96维
      \
       反向 GRU → 128

合并：128 + 128 = 256  输出[B,T,256]
        """
        self.temporal = nn.GRU(
            input_size=channels[-1], #96
            hidden_size=temporal_hidden,#128
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        """
        [B,T,256]->层归一化[B,T,256]  对样本内部的特征维度做归一化，让这些特征的数值尺度更稳定
        对每个时间帧的256维特征做归一化
        """
        temporal_dim = temporal_hidden * 2
        self.temporal_norm = nn.LayerNorm(temporal_dim)

        #弱标签怎么进入模型
        self.weak_encoder = nn.Sequential(#把原始的527维weaklabel转化为模型更容易使用的526维语义条件特征            
            nn.Linear(num_classes, temporal_dim),#经过Linear（[527,256]）->[B,256]  这里就是一种embedding/特征映射
            nn.SiLU(),
            nn.LayerNorm(temporal_dim),
        )

        #此时拿到的是[B,T,512]
        self.fusion = nn.Sequential(
            nn.Linear(temporal_dim * 2, temporal_dim),  #让模型学习音频信息和weaklabel信息组应该怎么组合->256
            nn.SiLU(),
            nn.LayerNorm(temporal_dim),
            nn.Dropout(dropout),#训练的时候随机把大约20％的特征暂时置0，防止模型太依赖某几个特征，减轻过拟合
        )

        #融合之后 输入[B,T,256]->[B,T,527]
        self.frame_head = nn.Linear(temporal_dim, num_classes)

        #把初始化参数保存到当前模型对象里面，后面其他函数需要使用
        self.num_classes = num_classes
        self.num_imics = num_imics

    def _unpack_inputs(self, model_input):
        if not isinstance(model_input, Mapping):
            raise TypeError(
                "AudioSet Model 1 expects {'mix_ri': ..., 'weak_labels': ...} as input"
            )
        if "mix_ri" not in model_input or "weak_labels" not in model_input:
            raise KeyError("AudioSet Model 1 requires both mix_ri and weak_labels inputs")

        mix_ri = _collated_tensor(model_input["mix_ri"])
        weak_labels = _collated_tensor(model_input["weak_labels"])
        if not torch.is_tensor(mix_ri):
            mix_ri = torch.as_tensor(mix_ri, dtype=torch.float32)
        if not torch.is_tensor(weak_labels):
            weak_labels = torch.as_tensor(weak_labels, dtype=torch.float32)
        if mix_ri.ndim != 6:
            raise ValueError(f"Expected RI input [B,T,1,M,2,F], got {tuple(mix_ri.shape)}")
        if mix_ri.shape[2] != 1 or mix_ri.shape[4] != 2:
            raise ValueError(f"Invalid RI singleton/component dimensions: {tuple(mix_ri.shape)}")
        if mix_ri.shape[3] < self.num_imics:
            raise ValueError(f"Model needs {self.num_imics} microphones, input has {mix_ri.shape[3]}")
        if weak_labels.ndim != 2 or weak_labels.shape != (mix_ri.shape[0], self.num_classes):
            raise ValueError(
                f"Expected weak labels [{mix_ri.shape[0]},{self.num_classes}], "
                f"got {tuple(weak_labels.shape)}"
            )
        return mix_ri[:, :, 0, : self.num_imics], weak_labels

    @staticmethod
    def _frame_mask(model_input, num_frames: int, device: torch.device) -> torch.Tensor:
        lengths: Optional[torch.Tensor] = None
        mix_value = model_input["mix_ri"]
        if isinstance(mix_value, (tuple, list)) and len(mix_value) == 2:
            if torch.is_tensor(mix_value[1]):
                lengths = mix_value[1]
        if lengths is None:
            batch_size = _collated_tensor(mix_value).shape[0]
            return torch.ones(batch_size, num_frames, dtype=torch.bool, device=device)
        lengths = lengths.to(device=device)
        return torch.arange(num_frames, device=device)[None] < lengths[:, None]


    #数据流真正开始
    def forward(self, model_input, target_dict=None, current_epoch=None):
        del target_dict, current_epoch
        mix_ri, weak_labels = self._unpack_inputs(model_input) #读取输入 mix_ri[B,T,M,2,F]，weak_labels[B,527]
        batch_size, num_frames = mix_ri.shape[:2]
        features = mix_ri.permute(0, 2, 3, 1, 4).reshape(  #permute变为[B,M,2,T,F]，reshape变为[B,M×2,T,F]
            batch_size, self.num_imics * 2, num_frames, mix_ri.shape[-1]
        )

        #这里是CNN特征提取
        features = self.encoder(features)


        #cnn后频率池化，[B,96,T,8/F]->mean frequency->[B,96,T]  把频率F这一维平均掉，然后transpose 交换第1,2维
        features = features.mean(dim=-1).transpose(1, 2)
        frame_mask = self._frame_mask(model_input, num_frames, features.device)#处理batch里不同长度的音频

        lengths = frame_mask.sum(dim=1).to(device="cpu")
        packed = nn.utils.rnn.pack_padded_sequence(
            features, lengths, batch_first=True, enforce_sorted=False #把[B,T,96]包装成适合处理不同长度音频的packed格式
        )
        #在这里把输入送入GRU
        packed_temporal, _ = self.temporal(packed)
        temporal, _ = nn.utils.rnn.pad_packed_sequence(
            packed_temporal, batch_first=True, total_length=num_frames
        )
        temporal = self.temporal_norm(temporal) #GRU后的结果[B,T,256]

        #这里处理弱标签
        weak_context = self.weak_encoder(
            weak_labels.to(device=temporal.device, dtype=temporal.dtype)
        )
        #通过unsqueeze把[B,256]增加一个维度[B,1,256]  然后expand变为[B,T,256]
        weak_context = weak_context.unsqueeze(1).expand(-1, num_frames, -1)
        #通过最后一个维度拼接->[B,T,512]  音频特征+弱标签融合
        fused = self.fusion(torch.cat([temporal, weak_context], dim=-1))#通过最后一个维度拼接->[B,T,512]
        frame_logits = self.frame_head(fused)
        return {
            "frame_logits": frame_logits,  #原始预测值[B, T, 527]
            "frame_probs": torch.sigmoid(frame_logits),#转换为概率[B, T, 527]
            "frame_mask": frame_mask,#有效帧掩码[B, T] 标记哪些时间帧是真实音频，哪些只是为了组成 batch 而补出来的 padding。
        }

"""
                    音频分支
                       │
             [B,2,T,F]
                       │
                      CNN
                       ↓
             [B,96,T,F/8]
                       │
                 频率平均
                       ↓
                [B,T,96]
                       │
                    BiGRU
                       ↓
                [B,T,256]
                       │
                       │
                       │
                       │        弱标签
                       │       [B,527]
                       │           │
                       │       Linear
                       │       527→256
                       │           ↓
                       │        [B,256]
                       │           │
                       │       复制T次
                       │           ↓
                       │       [B,T,256]
                       │           │
                       └────┬──────┘
                            │
                           cat
                            ↓
                       [B,T,512]
                            │
                          fusion
                            ↓
                       [B,T,256]
                            │
                       frame_head
                         256→527
                            ↓
                       [B,T,527]
                            │
                         sigmoid
                            ↓
                    每帧527类概率
"""
