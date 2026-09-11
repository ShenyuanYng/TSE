#!/usr/bin/env bash
#SBATCH --job-name=audioset-m1-smoke-01
#SBATCH --partition=L40
#SBATCH --nodelist=ailab01
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
export PYTHON_BIN="/home/syy12632604/anaconda3/envs/audioset/bin/python"
data_root="/hdd2/syy12632604/AudioSet"
raw_root="/hdd2/cse30029918/corpora/AudioSetRaw"

cd "$project_root"
test -x run.sh
test -d "$raw_root/balance_wav"
test -d "$raw_root/unbalanced_wav"
test -d "$raw_root/valid_wav"
test -f "$data_root/metadata/audioset_train_strong.tsv"

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$HOSTNAME"
echo "project_root=$project_root"
"$PYTHON_BIN" --version
nvidia-smi

export PYTHONUNBUFFERED=1
./run.sh \
    --accelerator gpu \
    --devices 1 \
    --epochs 1 \
    --batch-size 2 \
    --train-percent-check 0.001 \
    --val-percent-check 0.001 \
    --tag "_smoke_ailab01_job${SLURM_JOB_ID}"
