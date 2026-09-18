#!/usr/bin/env bash
#SBATCH --job-name=audioset-m1-smoke-03
#SBATCH --partition=RTXPRO5000
#SBATCH --nodelist=ailab03
#SBATCH --gres=gpu:rtxpro5000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
export PYTHON_BIN="/home/syy12632604/anaconda3/envs/audioset/bin/python"
data_root="/data/syy12632604/AudioSet"

cd "$project_root"
test -x run.sh
test -d "$data_root/audio/balance_wav"
test -d "$data_root/audio/unbalanced_wav"
test -d "$data_root/audio/eval_wav"
for manifest in class_labels_indices.csv balanced_train_segments.csv unbalanced_train_segments.csv eval_segments.csv audioset_train_strong.tsv audioset_eval_strong.tsv; do
    if [[ ! -f "$data_root/metadata/$manifest" ]]; then
        echo "Missing metadata: $data_root/metadata/$manifest" >&2
        exit 1
    fi
done

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$HOSTNAME"
echo "project_root=$project_root"
"$PYTHON_BIN" --version
nvidia-smi

export PYTHONUNBUFFERED=1
./run.sh \
    --accelerator gpu \
    --devices 1 \
    --test-after-fit false \
    --epochs 1 \
    --num-jobs 1 \
    --batch-size 2 \
    --train-percent-check 10 \
    --val-percent-check 10 \
    --tag "_smoke_ailab03_job${SLURM_JOB_ID}"
