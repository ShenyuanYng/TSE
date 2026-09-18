#!/usr/bin/env bash
#SBATCH --job-name=audioset-current-01
#SBATCH --partition=L40
#SBATCH --nodelist=ailab01
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Submit once from the project root; queue two continuations for this experiment.
set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
export PYTHON_BIN="/home/syy12632604/anaconda3/envs/audioset/bin/python"
data_root="/hdd2/syy12632604/AudioSet"
raw_root="/hdd2/cse30029918/corpora/AudioSetRaw"
num_jobs=3  # Total jobs including the first; each has a two-day time limit.

cd "$project_root"
test -x "$PYTHON_BIN"
test -x run.sh
test -x spksep_run.sh
test -f resume_training.py
test -f slurm/submit_continuation.py
test -f slurm/continue_training.sbatch
test -d "$raw_root/balance_wav"
test -d "$raw_root/unbalanced_wav"
test -d "$raw_root/eval_wav"
for manifest in class_labels_indices.csv balanced_train_segments.csv unbalanced_train_segments.csv eval_segments.csv audioset_train_strong.tsv audioset_eval_strong.tsv; do
    if [[ ! -f "$data_root/metadata/$manifest" ]]; then
        echo "Missing metadata: $data_root/metadata/$manifest" >&2
        exit 1
    fi
done

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$HOSTNAME"
echo "project_root=$project_root"
echo "Full training: lr=0.0005, epochs=50, batch_size=8, clip=5.0, num_jobs=$num_jobs"
echo "Gradient hook in source_separator.py at job start:"
sed -n '/    def on_after_backward(self):/,/    def training_step(self, batch, batch_idx):/p' source_separator.py
sha256sum source_separator.py nets/audioset_crnn.py datasets/audioset_logmel_dataset.py \
    loss_functions/audioset_loss.py train.py run.sh spksep_run.sh
"$PYTHON_BIN" --version
nvidia-smi

export PYTHONUNBUFFERED=1
# The job ID keeps this run separate from previous checkpoints and logs.
exec ./run.sh \
    --accelerator gpu \
    --devices 1 \
    --data-path "$data_root" \
    --train-audio-dir "[\"$raw_root/balance_wav\", \"$raw_root/unbalanced_wav\"]" \
    --val-audio-dir "[\"$raw_root/eval_wav\"]" \
    --train-weak-manifest '["metadata/balanced_train_segments.csv", "metadata/unbalanced_train_segments.csv"]' \
    --audioset-strict-paths false \
    --learning-rate 0.0005 \
    --gradient-clip-val 5.0 \
    --batch-size 8 \
    --epochs 50 \
    --train-on-chunk 10.0 \
    --seed 7 \
    --num-workers 4 \
    --train-percent-check 1.0 \
    --val-percent-check 1.0 \
    --test-after-fit false \
    --num-jobs "$num_jobs" \
    --tag "_teacherhook_full_ailab01_job${SLURM_JOB_ID}"
