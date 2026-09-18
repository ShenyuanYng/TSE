"""AudioSet dataset and path factories."""
"""
数据集工厂，把实验配置转换成一个可供DAtaLoader使用的AudioSetLogMelDataset对象
"""


#根据数据区分
def _partition_config(hparams, partition, name):
    prefix = {"tr": "train", "cv": "val", "test": "val"}[partition]
    return getattr(hparams, f"{prefix}_{name}")

#把以“秒”为单位的 STFT 参数换算成以“采样点”为单位的参数
def _stft_config(hparams):
    window_length = int(hparams.inwintime * hparams.samplerate)
    hop_length = int(hparams.hoptime * hparams.samplerate)
    fft_size = (
        int(hparams.fft_size * hparams.samplerate)
        if hparams.fft_size != 0.0
        else None
    )
    return window_length, hop_length, fft_size


#收集当前分区的路径配置
def datapath_creator(hparams, partition):
    """Join strong annotations, weak labels, and available audio files."""
    if hparams.dataset != "audioset_strong":
        raise ValueError(f"Unsupported dataset: {hparams.dataset}")

    from datasets.paths.audioset_paths import get_audioset_paths

    entries = get_audioset_paths(
        data_root=hparams.data_path,
        audio_dir=_partition_config(hparams, partition, "audio_dir"),
        weak_label_path=_partition_config(hparams, partition, "weak_manifest"),
        strong_label_path=_partition_config(hparams, partition, "strong_manifest"),
        strict=hparams.audioset_strict_paths,
    )

    if partition == "tr":
        return entries
    from datasets.eval_split import split_eval
    return split_eval(entries, hparams.data_path, hparams.eval_split_path,
                      hparams.eval_test_fraction, hparams.eval_split_seed)[partition]


#根据配置创建一个完整的AudioSetLogMelDataset 对象
"""
dataset_creator()
  ├─ 检查 dataset / feature_type / partition
  ├─ datapath_creator()
  │    └─ get_audioset_paths()
  ├─ _stft_config()
  └─ AudioSetLogMelDataset(...)
"""
def dataset_creator(hparams, partition, path_list=None, is_training=None):
    """Create one AudioSet Strong split without importing unrelated tasks."""
    if hparams.dataset != "audioset_strong":
        raise ValueError(f"Unsupported dataset: {hparams.dataset}")
    if hparams.feature_type != "logmel":
        raise ValueError(f"Unsupported feature_type: {hparams.feature_type}")
    if partition not in {"tr", "cv", "test"}:
        raise ValueError(f"Unsupported partition: {partition}")

    if path_list is None:
        path_list = datapath_creator(hparams, partition)
    if is_training is None:
        is_training = partition == "tr"

    from datasets.audioset_logmel_dataset import AudioSetLogMelDataset
    from datasets.paths.audioset_paths import resolve_path

    window_length, hop_length, fft_size = _stft_config(hparams)
    chunk_size = hparams.train_on_chunk if partition == "tr" else None
#创建AudioSetLogMelDataset
    return AudioSetLogMelDataset(
        path_list,
        class_map_path=resolve_path(hparams.class_map_path, hparams.data_path),
        num_classes=hparams.num_classes, #527类别
        sr=hparams.samplerate,
        chunk_size=chunk_size,
        rms_norm=hparams.rms_norm,
        training=is_training,
        window_length=window_length,
        hop_length=hop_length,
        fft_size=fft_size,
        window_type=hparams.window_type,
        strong_annotation_complete=hparams.strong_annotation_complete,
        n_mels=getattr(hparams, "n_mels", 64),
    )
