#!/usr/bin/env bash

# AudioSet Strong Model 1 experiment configuration.
# Change effective hyperparameters here; exp_defs/audioset_params.yaml only
# defines the command-line schema used by train.py.

set -euo pipefail

# ---------------------------------------------------------------------------
# Task and data
# ---------------------------------------------------------------------------
dataset="audioset_strong"
task_type="audioset_strong"
feature_type="ri"
collate_type="sequence"

data_path="./data/audioset"
class_map_path="metadata/class_labels_indices.csv"
train_audio_dir='["audio/train"]'
val_audio_dir='["audio/eval"]'
train_weak_manifest='["metadata/unbalanced_train_segments.csv"]'
val_weak_manifest='["metadata/eval_segments.csv"]'
train_strong_manifest="metadata/audioset_train_strong.tsv"
val_strong_manifest="metadata/audioset_eval_strong.tsv"
audioset_strict_paths=true
strong_annotation_complete=true

# Machine-specific paths live here with the other effective settings.
if [[ "${HOSTNAME:-}" == ailab01* ]]; then
    data_path="/hdd2/${USER}/AudioSet"
    train_audio_dir='["/hdd2/cse30029918/corpora/AudioSetRaw/balance_wav", "/hdd2/cse30029918/corpora/AudioSetRaw/unbalanced_wav"]'
    val_audio_dir='["/hdd2/cse30029918/corpora/AudioSetRaw/valid_wav"]'
    train_weak_manifest='["metadata/balanced_train_segments.csv", "metadata/unbalanced_train_segments.csv"]'
    val_weak_manifest="$train_weak_manifest"
    val_strong_manifest="$train_strong_manifest"
    audioset_strict_paths=false
elif [[ "${HOSTNAME:-}" == ailab03* ]]; then
    data_path="/data/${USER}/AudioSet"
    train_audio_dir='["audio/balance_wav", "audio/unbalanced_wav"]'
    val_audio_dir='["audio/valid_wav"]'
    train_weak_manifest='["metadata/balanced_train_segments.csv", "metadata/unbalanced_train_segments.csv"]'
    val_weak_manifest="$train_weak_manifest"
    val_strong_manifest="$train_strong_manifest"
    audioset_strict_paths=false
fi

# ---------------------------------------------------------------------------
# Audio features and Model 1
# ---------------------------------------------------------------------------
samplerate=16000
inwintime=0.032
hoptime=0.008
fft_size=0
window_type="sqrt_hann"
rms_norm=2
train_on_chunk=10.0
pad_back_or_front=0

model_name="AudioSetCRNN"
num_classes=527
num_imics=1
audioset_conv_channels='[32, 64, 96]'
audioset_temporal_hidden=128
audioset_rnn_layers=2
audioset_dropout=0.2

# ---------------------------------------------------------------------------
# Loss and optimization
# ---------------------------------------------------------------------------
loss_type="audioset_strong"
class_positive_weight=1.0
balance_positive_negative=true
learning_rate=0.001
decay_factor=0.5
patience=5
gradient_clip_val=5.0

# ---------------------------------------------------------------------------
# Training and runtime
# ---------------------------------------------------------------------------
batch_size=8
epochs=50
seed=7
shuffle=true
num_workers=4
train_percent_check=1.0
val_percent_check=1.0
cudnn_deterministic=1
cudnn_benchmark=0

accelerator="auto"
devices=1
distributed_backend="auto"
gpu_index=0

save_path="experiments"
tag=""

# Command-line overrides remain available, for example:
#   ./run.sh --batch-size 4 --train-percent-check 0.01
. tools/parse_options.sh || exit 1

./spksep_run.sh \
    --dataset "$dataset" --task-type "$task_type" \
    --feature-type "$feature_type" --collate-type "$collate_type" \
    --data-path "$data_path" --class-map-path "$class_map_path" \
    --train-audio-dir "$train_audio_dir" --val-audio-dir "$val_audio_dir" \
    --train-weak-manifest "$train_weak_manifest" --val-weak-manifest "$val_weak_manifest" \
    --train-strong-manifest "$train_strong_manifest" --val-strong-manifest "$val_strong_manifest" \
    --audioset-strict-paths "$audioset_strict_paths" \
    --strong-annotation-complete "$strong_annotation_complete" \
    --samplerate "$samplerate" --inwintime "$inwintime" \
    --hoptime "$hoptime" --fft-size "$fft_size" --window-type "$window_type" \
    --rms-norm "$rms_norm" --train-on-chunk "$train_on_chunk" \
    --pad-back-or-front "$pad_back_or_front" \
    --model-name "$model_name" --num-classes "$num_classes" \
    --num-imics "$num_imics" --audioset-conv-channels "$audioset_conv_channels" \
    --audioset-temporal-hidden "$audioset_temporal_hidden" \
    --audioset-rnn-layers "$audioset_rnn_layers" --audioset-dropout "$audioset_dropout" \
    --loss-type "$loss_type" --class-positive-weight "$class_positive_weight" \
    --balance-positive-negative "$balance_positive_negative" \
    --learning-rate "$learning_rate" --decay-factor "$decay_factor" \
    --patience "$patience" --gradient-clip-val "$gradient_clip_val" \
    --batch-size "$batch_size" --epochs "$epochs" --seed "$seed" \
    --shuffle "$shuffle" --num-workers "$num_workers" \
    --train-percent-check "$train_percent_check" --val-percent-check "$val_percent_check" \
    --cudnn-deterministic "$cudnn_deterministic" --cudnn-benchmark "$cudnn_benchmark" \
    --accelerator "$accelerator" --devices "$devices" \
    --distributed-backend "$distributed_backend" --gpu-index "$gpu_index" \
    --save-path "$save_path" --tag "$tag"
