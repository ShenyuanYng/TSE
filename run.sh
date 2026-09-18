#!/usr/bin/env bash

# AudioSet Strong Model 1 experiment configuration.
# Change effective hyperparameters here; exp_defs/audioset_params.yaml only
# defines the command-line schema used by train.py.

set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"

# Submit this same configuration with: ./run.sh --submit true
# No per-experiment sbatch file is needed. The scheduler stores the resolved
# launcher arguments, so later edits to run.sh do not change a queued job.
submit=false
slurm_node="ailab01"
slurm_partition=""  # Empty: select L40 / RTXPRO5000 from slurm_node.
slurm_gres=""       # Empty: select the matching single GPU.
slurm_cpus_per_task=6
slurm_mem="32G"
slurm_time="2-00:00:00"
slurm_account=""    # Empty: use the scheduler's default account.
slurm_job_name="audioset-embedding"
python_bin="${PYTHON_BIN:-}"  # Empty: audioset conda Python on ailab, python locally.
help_message="Edit settings in run.sh, then run ./run.sh locally or ./run.sh --submit true for Slurm."

# ---------------------------------------------------------------------------
# Task and data
# ---------------------------------------------------------------------------
dataset="audioset_strong"
task_type="audioset_strong"
feature_type="logmel"
collate_type="sequence"

# Empty paths use the machine profile below; explicit values override it.
data_path=""
class_map_path="metadata/class_labels_indices.csv"
train_audio_dir=""
val_audio_dir=""
train_weak_manifest=""
val_weak_manifest='["metadata/eval_segments.csv"]'
train_strong_manifest="metadata/audioset_train_strong.tsv"
val_strong_manifest="metadata/audioset_eval_strong.tsv"
audioset_strict_paths=""
strong_annotation_complete=true
n_mels=64
eval_test_fraction=0.5
eval_split_seed=7
eval_split_path="metadata/eval_split.json"
test_after_fit=false
frame_accuracy_threshold=0.5

# ---------------------------------------------------------------------------
# Audio features and Model 1
# ---------------------------------------------------------------------------
samplerate=16000
inwintime=0.032
hoptime=0.016  # 256 samples: 50% overlap with the 512-sample window.
fft_size=0
window_type="hann"
rms_norm=2
train_on_chunk=10.0
pad_back_or_front=0

model_name="AudioSetCRNN"
num_classes=527
audioset_conv_channels='[32, 64, 96]'
audioset_temporal_hidden=128
audioset_embedding_dim=128  # D: independent of the BiGRU output width.
audioset_rnn_layers=2

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
# Total Slurm jobs, including the first. Set 3 for up to three two-day jobs.
num_jobs=1
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

# Resolve paths using the requested compute node when submitting from a login
# host. Only fill empty values, preserving explicit settings and CLI overrides.
data_profile="${SLURMD_NODENAME:-${HOSTNAME:-local}}"
if [[ "$submit" == true ]]; then
    data_profile="$slurm_node"
fi
case "$data_profile" in
    ailab01*)
        data_path="${data_path:-/hdd2/${USER}/AudioSet}"
        if [[ -z "$train_audio_dir" ]]; then
            train_audio_dir='["/hdd2/cse30029918/corpora/AudioSetRaw/balance_wav", "/hdd2/cse30029918/corpora/AudioSetRaw/unbalanced_wav"]'
        fi
        if [[ -z "$val_audio_dir" ]]; then
            val_audio_dir='["/hdd2/cse30029918/corpora/AudioSetRaw/eval_wav"]'
        fi
        ;;
    ailab03*)
        data_path="${data_path:-/data/${USER}/AudioSet}"
        if [[ -z "$train_audio_dir" ]]; then
            train_audio_dir='["audio/balance_wav", "audio/unbalanced_wav"]'
        fi
        if [[ -z "$val_audio_dir" ]]; then
            val_audio_dir='["audio/eval_wav"]'
        fi
        ;;
esac
case "$data_profile" in
    ailab01*|ailab03*)
        if [[ -z "$train_weak_manifest" ]]; then
            train_weak_manifest='["metadata/balanced_train_segments.csv", "metadata/unbalanced_train_segments.csv"]'
        fi
        audioset_strict_paths="${audioset_strict_paths:-false}"
        python_bin="${python_bin:-/home/${USER}/anaconda3/envs/audioset/bin/python}"
        ;;
esac
data_path="${data_path:-./data/audioset}"
if [[ -z "$train_audio_dir" ]]; then
    train_audio_dir='["audio/train"]'
fi
if [[ -z "$val_audio_dir" ]]; then
    val_audio_dir='["audio/eval"]'
fi
if [[ -z "$train_weak_manifest" ]]; then
    train_weak_manifest='["metadata/unbalanced_train_segments.csv"]'
fi
audioset_strict_paths="${audioset_strict_paths:-true}"
export PYTHON_BIN="${python_bin:-python}"
export PYTHONUNBUFFERED=1

resource_node="$slurm_node"
if [[ "$submit" == false && -n "${SLURMD_NODENAME:-}" ]]; then
    resource_node="$SLURMD_NODENAME"
fi
case "$resource_node" in
    ailab01)
        slurm_partition="${slurm_partition:-L40}"
        slurm_gres="${slurm_gres:-gpu:l40:1}"
        ;;
    ailab03)
        slurm_partition="${slurm_partition:-RTXPRO5000}"
        slurm_gres="${slurm_gres:-gpu:rtxpro5000:1}"
        ;;
esac
if [[ ! "$num_jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "num_jobs must be a positive integer (including the first job)." >&2
    exit 1
fi

launch_args=( \
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
    --distributed-backend "$distributed_backend" --gpu-index "$gpu_index" \
    --save-path "$save_path" --tag "$tag" --num-jobs "$num_jobs" \
    --slurm-partition "$slurm_partition" --slurm-gres "$slurm_gres" \
    --slurm-cpus-per-task "$slurm_cpus_per_task" --slurm-mem "$slurm_mem" \
    --slurm-time "$slurm_time" --slurm-account "$slurm_account" \
)

if [[ "$submit" == true ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "Already inside a Slurm job; use ./run.sh without --submit true." >&2
        exit 1
    fi
    if [[ -z "$slurm_node" || -z "$slurm_partition" || -z "$slurm_gres" ]]; then
        echo "Configure slurm_node, slurm_partition and slurm_gres in run.sh." >&2
        exit 1
    fi
    if (( num_jobs > 1 )) && [[ "$slurm_node" != ailab01 && "$slurm_node" != ailab03 ]]; then
        echo "Multi-job continuation currently supports ailab01 and ailab03." >&2
        exit 1
    fi
    mkdir -p "$project_root/logs"
    sbatch_args=(--parsable --export=ALL --chdir="$project_root"
        --job-name="$slurm_job_name" --nodelist="$slurm_node"
        --partition="$slurm_partition" --gres="$slurm_gres"
        --cpus-per-task="$slurm_cpus_per_task" --mem="$slurm_mem" --time="$slurm_time"
        --output="$project_root/logs/%x-%j.out" --error="$project_root/logs/%x-%j.err")
    if [[ -n "$slurm_account" ]]; then
        sbatch_args+=(--account="$slurm_account")
    fi
    # Submit the existing launcher directly, with every effective argument.
    exec sbatch "${sbatch_args[@]}" "$project_root/spksep_run.sh" "${launch_args[@]}"
fi
exec bash "$project_root/spksep_run.sh" "${launch_args[@]}"
