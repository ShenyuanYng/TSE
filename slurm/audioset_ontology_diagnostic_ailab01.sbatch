#!/usr/bin/env bash
#SBATCH --job-name=ontology7
#SBATCH --partition=L40
#SBATCH --nodelist=ailab01
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/ontology_confusion_37074/slurm-%j.log

# From the repository root, create the log directory before submitting:
# mkdir -p outputs/ontology_confusion_37074
# sbatch slurm/audioset_ontology_diagnostic_ailab01.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

exec /home/syy12632604/anaconda3/envs/audioset/bin/python -u diagnose_ontology_confusions.py \
  --checkpoint "experiments/audioset_strong/AudioSetCRNN_logmel_embedding_concat_d128_hop0.016_h128_rnn2_bs256_chunk10.0_lr0.00010922_20:18_psrsion+recall+F1_job37074/version_37074/_ckpt_epoch_30.ckpt" \
  --reference-report probability_reference_37074.json \
  --ontology resources/audioset_ontology.json \
  --device cuda \
  --class-chunk-size 32 \
  --frame-chunk-size 256 \
  --output-dir outputs/ontology_confusion_37074
