"""Single-class conditioned logMel examples for AudioSet Strong."""
from __future__ import annotations
import warnings
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from datasets.paths.audioset_paths import load_class_map
"""
准备数据，读取AudioSet音频，并把WAV音频转换成模型能够直接使用的log-mel特征，同时准备对应的weak label 和strong label
"""
class AudioSetLogMelDataset(Dataset):

    #初始化数据集。读取类别表、设置采样率、STFT参数、Mel滤波器；同时整理每条音频对应哪些类别
    def __init__(self, path_list, class_map_path, num_classes=527, sr=16000,
                 chunk_size=None, rms_norm=2, training=True, window_length=512,
                 hop_length=256, fft_size=None, window_type="hann",
                 strong_annotation_complete=True, n_mels=64):
        self.path_list = path_list
        self.mid_to_index, self.class_names = load_class_map(class_map_path)
        if len(self.class_names) != num_classes:
            raise ValueError("num_classes must match the class map")
        self.num_classes, self.sr = num_classes, sr
        self.chunk_size = int(chunk_size * sr) if chunk_size is not None else None
        self.rms_norm, self.training = rms_norm, training
        self.window_length, self.hop_length = window_length, hop_length
        self.fft_size = fft_size or window_length
        self.strong_annotation_complete = strong_annotation_complete
        if window_type != "hann":
            raise ValueError("logMel uses a periodic Hann window")
        if not 0 < hop_length <= window_length <= self.fft_size or n_mels < 1:
            raise ValueError("Invalid logMel dimensions")
        self.window = torch.hann_window(window_length, periodic=True)
        # HTK Mel scale, triangular filters, power input, natural logarithm.
        #滤波器的构造  2595× log10(1 + f / 700)
        mel_max = 2595 * np.log10(1 + (sr / 2) / 700)
        #在mel坐标上均匀取66个点，然后转回Hz
        edges = 700 * (10 ** (torch.linspace(0, mel_max, n_mels + 2) / 2595) - 1)
        #构造FFT每个频率bin的实际频率
        frequencies = torch.linspace(0, sr / 2, self.fft_size // 2 + 1)[:, None]#[257,1]
        #分别计算每个三角滤波器的上升边和下降边
        rising = (frequencies - edges[:-2]) / (edges[1:-1] - edges[:-2])
        falling = (edges[2:] - frequencies) / (edges[2:] - edges[1:-1])
        #得到标准的三角形Mel滤波器矩阵
        self.mel_filters = torch.minimum(rising, falling).clamp_min(0)
        if (self.mel_filters.sum(0) == 0).any():
            raise ValueError("Empty Mel filter; reduce n_mels or increase fft_size")
        self.samples = []
        unknown_weak, unknown_strong = set(), set()

        """
        self.samples里存的是(音频索引i，当前目标类别mid)，一条音频有几个week label 就会生成几个（i,mid）对
        """
        for i, entry in enumerate(path_list):
            for mid in sorted(set(entry.get("weak_mids", []))):
                if mid in self.mid_to_index:
                    self.samples.append((i, mid))
                else:
                    unknown_weak.add(mid)
            unknown_strong.update(e["mid"] for e in entry.get("events", [])
                                  if e["mid"] not in self.mid_to_index)
        if unknown_weak or unknown_strong:
            warnings.warn(f"Unmapped MIDs: weak={sorted(unknown_weak)}, strong={sorted(unknown_strong)}")
        if not self.samples:
            raise ValueError("No mapped weak-label conditions in this split")

    #这个dataset一共有多少个样本
    def __len__(self):
        return len(self.samples)

    #真正读取WAV
    def _read_audio(self, entry: dict):
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
            raise ValueError("AudioSetLogMelDataset supports rms_norm=0 or rms_norm=2")
        return waveform.astype(np.float32, copy=False), start_sample

    #把waveform转换成Log-Mel Spectrogram
    def _logmel_features(self, waveform):
        spectrum = torch.stft(torch.from_numpy(waveform), n_fft=self.fft_size,
                              hop_length=self.hop_length, win_length=self.window_length,
                              window=self.window, center=True, pad_mode="constant",
                              return_complex=True)
        mel = spectrum.abs().square().transpose(0, 1) @ self.mel_filters
        return mel.clamp_min(1e-10).log().numpy().astype(np.float32)

    #计算 Log-Mel 中每一个时间帧对应原音频中的时间位置
    def _frame_centers(self, num_frames, crop_start_sample):
        return (np.arange(num_frames, dtype=np.float32) * self.hop_length
                + crop_start_sample) / self.sr

    #真正读取一条音频
    def __getitem__(self, index):
        #确定音频和目标类别   entry_index = 音频片段编号   mid= 当前要检测的类别
        entry_index, mid = self.samples[index]
        entry = self.path_list[entry_index]
        #读取音频
        waveform, crop_start = self._read_audio(entry)
        if len(waveform) == 0:
            raise ValueError(f"Empty audio: {entry['audio']}")
        #计算logmel
        features = self._logmel_features(waveform)
        #将秒级标注转成逐帧标签
        centers = self._frame_centers(len(features), crop_start)
        labels = np.zeros(len(features), dtype=np.float32)

        mask = np.full_like(labels, float(self.strong_annotation_complete))

        #只选择当前目标类别的事件
        events = [e for e in entry.get("events", []) if e["mid"] == mid]
        for event in events:
            active = (centers >= event["onset"]) & (centers < event["offset"])
            if event.get("present", True):
                labels[active] = 1
                mask[active] = 1
        for event in events:
            if not event.get("present", True):
                active = (centers >= event["onset"]) & (centers < event["offset"])
                labels[active], mask[active] = 0, 1
        # A centered STFT may include a last frame exactly at audio end.
        mask *= (centers < (crop_start + len(waveform)) / self.sr)

        #创建类别条件
        condition = np.zeros(self.num_classes, dtype=np.float32)
        class_index = self.mid_to_index[mid]
        condition[class_index] = 1
        return {"logmel": features, "weak_labels": condition}, {
            "strong_labels": labels, "strong_mask": mask, "frame_times": centers,
            "class_index": class_index, "utt_id": entry["id"], "audio_path": entry["audio"],
        }
