#!/usr/bin/env bash

# AudioSet-only launcher retaining the original framework's two-level
# run.sh -> launcher structure.

set -euo pipefail
. path.sh

exp_def="exp_defs/audioset_params.yaml"
dataset="audioset_strong"
task_type="audioset_strong"
feature_type="logmel"
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
n_mels=64
eval_test_fraction=0.5
eval_split_seed=7
eval_split_path="metadata/eval_split.json"
test_after_fit=false
frame_accuracy_threshold=0.5

samplerate=16000
inwintime=0.032
hoptime=0.016
fft_size=0
window_type="hann"
rms_norm=2
train_on_chunk=10.0
pad_back_or_front=0

model_name="AudioSetCRNN"
num_classes=527
audioset_conv_channels='[32, 64, 96]'
audioset_temporal_hidden=128
audioset_embedding_dim=128
audioset_rnn_layers=2

loss_type="audioset_strong"
class_positive_weight=1.0
balance_positive_negative=true
learning_rate=0.001
decay_factor=0.5
patience=5
gradient_clip_val=5.0

batch_size=8
epochs=50
num_jobs=1
slurm_partition=""
slurm_gres=""
slurm_cpus_per_task=6
slurm_mem="32G"
slurm_time="2-00:00:00"
slurm_account=""
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

. tools/parse_options.sh || exit 1

if [[ "$dataset" != "audioset_strong" || "$task_type" != "audioset_strong" ]]; then
    echo "This repository supports only the audioset_strong task." >&2
    exit 1
fi
if [[ "$feature_type" != "logmel" ]]; then
    echo "AudioSet Model 1 currently supports only feature_type=logmel." >&2
    exit 1
fi
if [[ "$samplerate" != "16000" ]]; then
    echo "AudioSet Model 1 expects samplerate=16000." >&2
    exit 1
fi
if [[ -z "$data_path" ]]; then
    echo "data_path must be configured." >&2
    exit 1
fi

if [[ ! "$num_jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "num_jobs must be a positive integer (total jobs including this job)." >&2
    exit 1
fi
if (( num_jobs > 1 )) && [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "num_jobs > 1 requires launching through sbatch." >&2
    exit 1
fi

# Respect CUDA_VISIBLE_DEVICES supplied by Slurm. Select gpu_index only for a
# direct local launch where no scheduler has constrained device visibility.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$gpu_index"
fi

save_name="${dataset}/${model_name}_logmel_embedding_concat_d${audioset_embedding_dim}_hop${hoptime}"
save_name+="_h${audioset_temporal_hidden}_rnn${audioset_rnn_layers}"
save_name+="_bs${batch_size}_chunk${train_on_chunk}_lr${learning_rate}"
if [[ "$class_positive_weight" != "1.0" ]]; then
    save_name+="_posw${class_positive_weight}"
fi
if [[ "$balance_positive_negative" != "true" ]]; then
    save_name+="_unbalanced"
fi
save_name+="$tag"
if [[ -n "${SLURM_JOB_ID:-}" && "$save_name" != *"_job${SLURM_JOB_ID}" ]]; then
    save_name+="_job${SLURM_JOB_ID}"
fi

logdir="${save_path}/${save_name}/logs"
mkdir -p "$logdir"
experiment_log="${logdir}/training.log"
if [[ -e "$experiment_log" ]]; then
    echo "$experiment_log exists; choose a new --tag before proceeding." >&2
    exit 1
fi

echo "Experiment logfile: $(pwd)/$experiment_log"
echo "Begin Training"

python_bin="${PYTHON_BIN:-python}"
if (( num_jobs > 1 )); then
    echo "Scheduling $((num_jobs - 1)) continuation jobs; total num_jobs=$num_jobs"
    "$python_bin" slurm/submit_continuation.py \
        --run-dir "${save_path}/${save_name}" \
        --after-job "$SLURM_JOB_ID" \
        --additional-jobs "$((num_jobs - 1))" \
        --node "${SLURMD_NODENAME:-${HOSTNAME%%.*}}" \
        --partition "$slurm_partition" --gres "$slurm_gres" \
        --cpus-per-task "$slurm_cpus_per_task" --mem "$slurm_mem" \
        --time "$slurm_time" --account "$slurm_account" \
        --starting-job
fi
perl "$PWD/tools/run.pl" "$experiment_log" \
    "$python_bin" train.py \
    --exp-def "$exp_def" \
    --dataset "$dataset" --task-type "$task_type" \
    --feature-type "$feature_type" --collate-type "$collate_type" \
    --data-path "$data_path" --class-map-path "$class_map_path" \
    --train-audio-dir "$train_audio_dir" --val-audio-dir "$val_audio_dir" \
    --train-weak-manifest "$train_weak_manifest" --val-weak-manifest "$val_weak_manifest" \
    --train-strong-manifest "$train_strong_manifest" --val-strong-manifest "$val_strong_manifest" \
    --audioset-strict-paths "$audioset_strict_paths" \
    --strong-annotation-complete "$strong_annotation_complete" \
    --n-mels "$n_mels" --eval-test-fraction "$eval_test_fraction" \
    --eval-split-seed "$eval_split_seed" --eval-split-path "$eval_split_path" \
    --test-after-fit "$test_after_fit" \
    --frame-accuracy-threshold "$frame_accuracy_threshold" \
    --samplerate "$samplerate" --inwintime "$inwintime" \
    --hoptime "$hoptime" --fft-size "$fft_size" --window-type "$window_type" \
    --rms-norm "$rms_norm" --train-on-chunk "$train_on_chunk" \
    --pad-back-or-front "$pad_back_or_front" \
    --model-name "$model_name" --num-classes "$num_classes" \
    --audioset-conv-channels "$audioset_conv_channels" \
    --audioset-temporal-hidden "$audioset_temporal_hidden" \
    --audioset-embedding-dim "$audioset_embedding_dim" \
    --audioset-rnn-layers "$audioset_rnn_layers" \
    --loss-type "$loss_type" --class-positive-weight "$class_positive_weight" \
    --balance-positive-negative "$balance_positive_negative" \
    --learning-rate "$learning_rate" --decay-factor "$decay_factor" \
    --patience "$patience" --gradient-clip-val "$gradient_clip_val" \
    --batch-size "$batch_size" --epochs "$epochs" --seed "$seed" \
    --shuffle "$shuffle" --num-workers "$num_workers" \
    --train-percent-check "$train_percent_check" --val-percent-check "$val_percent_check" \
    --cudnn-deterministic "$cudnn_deterministic" --cudnn-benchmark "$cudnn_benchmark" \
    --accelerator "$accelerator" --devices "$devices" \
    --distributed-backend "$distributed_backend" \
    --save-path "$save_path" --save-name "$save_name"
