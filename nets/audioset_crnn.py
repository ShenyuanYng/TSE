"""Target-class conditioned logMel CRNN, without microphone or RI axes."""
import torch
import torch.nn as nn


def _collated_tensor(value):
    return value[0] if isinstance(value, (tuple, list)) else value


class ConvFrequencyBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        #输入 卷积 归一化 SiLU激活 频率池化
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2), ceil_mode=True),# 1✖️2表示 时间轴T不压缩，F压缩一半
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AudioSetCRNN(nn.Module):
    def __init__(self, num_classes=527, conv_channels=(32, 64, 96),
                 temporal_hidden=128, rnn_layers=2, embedding_dim=128):
        super().__init__()
        if not isinstance(embedding_dim, int) or embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer")
        self.num_classes = num_classes
        channels = list(conv_channels)
        if not channels:
            raise ValueError("conv_channels cannot be empty")
        blocks, previous = [], 1
        #根据[32,64,96]建立三层CNN
        for channel in channels:
            blocks.append(ConvFrequencyBlock(previous, channel))
            previous = channel
        self.encoder = nn.Sequential(*blocks) #将三个CNN block拼成频谱编码器
        #创建时序GRU
        self.temporal = nn.GRU(channels[-1], temporal_hidden, rnn_layers,
                               batch_first=True, bidirectional=True,
                               dropout=0.0)
        width = temporal_hidden * 2  #128*2
        #对每一个时间帧的256维GRU输出做LayerNorm，使时序特征分布更稳定
        self.temporal_norm = nn.LayerNorm(width)

        # 可训练矩阵 [C,D]，D 独立于 BiGRU 的输出维度（默认 256）。
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        # 拼接音频特征和类别 embedding，再映射回音频特征维度。
        self.fusion = nn.Sequential(nn.Linear(width + embedding_dim, width),
                                    nn.SiLU(), nn.LayerNorm(width))
        self.frame_head = nn.Linear(width, 1)#[B,T,256]->[B,T,1]

    def forward(self, model_input, target_dict=None, current_epoch=None):

        #取出模型的两个输入
        logmel = _collated_tensor(model_input["logmel"])#[B,T,64]

        condition = _collated_tensor(model_input["weak_labels"])#[B,527]


        #验证环节 验证log-mel必须是三维张量  目标类别向量必须有正确尺寸
        if logmel.ndim != 3:
            raise ValueError("logmel must have shape [B,T,n_mels]")
        batch, frames, _ = logmel.shape
        if condition.shape != (batch, self.num_classes):
            raise ValueError("weak_labels must have shape [B,C]")
        #验证condition是one-hot [0,0,0,1,0,0,...] 只能有一个1
        if not (((condition == 0) | (condition == 1)).all()
                and (condition.sum(-1) == 1).all()):
            raise ValueError("Each condition must be one-hot")

        value = model_input["logmel"]

        lengths = value[1] if isinstance(value, (tuple, list)) else torch.full((batch,), frames)
        """
        frames=5  lengths=[5,3]frame_mask =
[[True, True, True, True, True ],
 [True, True, True, False, False]]  会让loss忽略补0帧
        """
        frame_mask = torch.arange(frames, device=logmel.device)[None] < lengths.to(logmel.device)[:, None]
        features = self.encoder(logmel.unsqueeze(1)).mean(-1).transpose(1, 2)


        packed = nn.utils.rnn.pack_padded_sequence(features, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        packed, _ = self.temporal(packed)
        temporal, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=frames)
        # [B,C] @ [C,D] -> [B,D]，one-hot 乘法等价于查找对应 embedding。
        weight = self.class_embedding.weight
        context = condition.to(device=weight.device, dtype=weight.dtype) @ weight
        context = context.unsqueeze(1).expand(-1, frames, -1)  # [B,T,D]

        combined = torch.cat((self.temporal_norm(temporal), context), dim=-1)
        fused = self.fusion(combined)  # [B,T,256+D] -> [B,T,256]（默认）
        logits = self.frame_head(fused).squeeze(-1)
        #每帧输出一个。对于当前指定的target class，这一帧有没有这个类别。target class 本身通常来自 weak label 中已有的类别
        return {"frame_logits": logits, "frame_probs": logits.sigmoid(), "frame_mask": frame_mask}
