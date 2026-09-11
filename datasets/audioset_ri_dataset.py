"""AudioSet Strong inputs and frame-level ground truth for Model 1."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from datasets.paths.audioset_paths import load_class_map
from utils.audio_utils_torch import do_stft

"""
真正读取单条音频，把WAV转为单声道波形和RI STFT，同时把弱标签，强标签和监督掩码编码成Numpy数组
把audioset_path整理好的sample，进一步转换成模型真正使用的输入和groundtruth
audioset_paths.py
        ↓
      entry
        ↓
AudioSetRIDataset
        │
        ├──── audio ──→ STFT ──→ RI
        │                       [T,1,1,2,F]
        │
        ├──── weak_mids ──────→ [527]
        │
        └──── events ─────────→ [T,527]
                                strong GT
"""
class AudioSetRIDataset(Dataset):
    

    def __init__(
        self,
        path_list: List[dict],
        class_map_path: str,
        num_classes: int = 527,
        sr: int = 16000,
        chunk_size: Optional[float] = None,
        rms_norm: int = 2,
        training: bool = True,
        window_length: int = 512,
        hop_length: int = 128,
        fft_size: Optional[int] = None,
        window_type: str = "sqrt_hann",
        strong_annotation_complete: bool = True,
    ) -> None:
        self.path_list = path_list
        """
        返回的是[
       entry1,
       entry2,
       entry3,
       ...
        ]
        """
        self.mid_to_index, self.class_names = load_class_map(class_map_path)
        if len(self.class_names) != num_classes:
            raise ValueError(
                f"num_classes={num_classes}, but {class_map_path!r} defines "
                f"{len(self.class_names)} classes"
            )
        self.num_classes = num_classes
        self.sr = sr
        self.chunk_size = int(chunk_size * sr) if chunk_size is not None else None
        self.rms_norm = rms_norm
        self.training = training
        self.window_length = window_length
        self.hop_length = hop_length
        self.fft_size = fft_size
        self.window_type = window_type
        self.strong_annotation_complete = strong_annotation_complete

    def __len__(self) -> int:
        return len(self.path_list)

    #真正读取WAV
    def _read_audio(self, entry: dict) -> Tuple[np.ndarray, int]:
        info = sf.info(entry["audio"])
        if info.samplerate != self.sr:
            raise ValueError(
                f"Sample-rate mismatch for {entry['audio']}: got {info.samplerate}, expected {self.sr}"
            )
        n_samples = info.frames
        #根据音频长度不同情况的处理
        if self.chunk_size is None or n_samples <= self.chunk_size:
            start_sample = 0
            end_sample = n_samples
        elif self.training:
            start_sample = int(np.random.randint(0, n_samples - self.chunk_size + 1))
            end_sample = start_sample + self.chunk_size
        else:
            start_sample = 0
            end_sample = self.chunk_size
         #真正读取waveform
        waveform, _ = sf.read(
            entry["audio"],
            start=start_sample,
            stop=end_sample,
            dtype="float32",
            always_2d=True,
        )
        waveform = waveform.mean(axis=1)
        if self.rms_norm == 2:
            scale = float(np.std(waveform))
            if scale > 0.0:
                waveform = waveform / scale
        elif self.rms_norm != 0:
            raise ValueError("AudioSetRIDataset supports rms_norm=0 or rms_norm=2")
        return waveform.astype(np.float32, copy=False), start_sample


    #STFT 转换为频谱图
    def _ri_features(self, waveform: np.ndarray) -> np.ndarray:
        stft = do_stft(
            torch.from_numpy(waveform)[None],
            self.window_length,
            self.hop_length,
            self.window_type,
            self.fft_size,
        )  # [1, T, F, 2]
        ri = stft.permute(1, 0, 3, 2).contiguous().numpy()  # [T, 1, 2, F]
        return ri[:, None].astype(np.float32, copy=False)  # [T, 1, 1, 2, F]

    #生成弱标签[527] 这里的weak label不带时间T
    def _weak_target(self, entry: dict) -> np.ndarray:
        target = np.zeros(self.num_classes, dtype=np.float32) #先生成全0
        #然后根据id，将对应位置赋值为1
        for mid in entry.get("weak_mids", []):
            index = self.mid_to_index.get(mid)
            if index is not None:
                target[index] = 1.0
        return target

    #计算STFT的每一帧对应原始音频的什么时间点
    """
    根据 STFT 的帧数、窗口长度、步长和音频裁剪起点，计算每个 STFT 帧中心在原始 WAV 中对应的时间（单位：秒）。
它返回一个形状为 [T] 的 NumPy 数组。默认 10 秒音频时形状是 [1253]，这些时间随后用于把强标签的 onset/offset 区间对齐到具体 STFT 帧。
    """
    def _frame_centers(self, num_frames: int, crop_start_sample: int) -> np.ndarray:
        # do_stft left-pads by window_length-hop_length (see audio_utils_torch.py).
        left_padding = self.window_length - self.hop_length
        centers = (
            np.arange(num_frames, dtype=np.float32) * self.hop_length
            - left_padding
            + self.window_length / 2.0
        )
        return (centers + crop_start_sample) / float(self.sr)


    #把 event-level strong annotation 转成模型训练真正需要的 [T,527]
    def _strong_target(
        self, entry: dict, num_frames: int, crop_start_sample: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        labels = np.zeros((num_frames, self.num_classes), dtype=np.float32)
        mask_value = 1.0 if self.strong_annotation_complete else 0.0
        mask = np.full_like(labels, mask_value)
        centers = self._frame_centers(num_frames, crop_start_sample)

        for event in entry.get("events", []):
            index = self.mid_to_index.get(event["mid"])
            if index is None:
                continue
            active = (centers >= float(event["onset"])) & (centers < float(event["offset"]))
            if event.get("present", True):
                labels[active, index] = 1.0
                if not self.strong_annotation_complete:
                    mask[active, index] = 1.0
            else:
                labels[active, index] = 0.0
                mask[active, index] = 1.0
        return labels, mask

    def __getitem__(self, index: int):
        #拿一条数据
        entry = self.path_list[index]
        #得到waveform
        waveform, crop_start_sample = self._read_audio(entry)
        #得到[T,1,1,2,F]
        mix_ri = self._ri_features(waveform)
        num_frames = mix_ri.shape[0]
        #得到[527]
        weak_labels = self._weak_target(entry)

        #得到strong_labels[T,527]  strong_mask=[T,527]
        strong_labels, strong_mask = self._strong_target(
            entry, num_frames, crop_start_sample
        )
        #模型真正的输入[T,1,1,2,F]，[527]
        model_input = {
            "mix_ri": mix_ri,
            "weak_labels": weak_labels,
        }

        #ground truth
        target = {
            "strong_labels": strong_labels,
            "strong_mask": strong_mask,
            "frame_times": self._frame_centers(num_frames, crop_start_sample),
            "utt_id": entry["id"],
            "audio_path": entry["audio"],
        }
        return model_input, target
