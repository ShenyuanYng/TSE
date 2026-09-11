# AudioSet Strong Model 1

This repository trains one weak-label-conditioned CRNN against real frame-level
AudioSet Strong annotations. It intentionally contains no weak-only training,
pseudo-label generation, Model 2, or speech-separation implementation.

## Framework

```text
run.sh                         all effective settings and machine paths
  -> spksep_run.sh             validates settings, names and launches the run
    -> train.py                creates the Lightning trainer and checkpoints
      -> SourceSeparator       task-independent training loop
        -> dataset_creator
        -> model_creator
        -> loss_creator
        -> task_creator
        -> collate_creator
```

The creator boundary is retained from the original speech-separation framework:
the training loop does not import or construct a concrete dataset, model, loss,
task adapter, or collator itself.

## Training data

The training split is the intersection of:

1. clip IDs in `metadata/audioset_train_strong.tsv`;
2. entries in the balanced and unbalanced weak manifests;
3. audio files available in the configured balanced and unbalanced directories.

Each dataset item contains RI STFT features and a weak-label vector as model
inputs, plus aligned frame-level strong labels as the training target.

Machine-specific paths for `ailab01` and `ailab03`, model settings, loss
settings, and training hyperparameters are all defined in `run.sh`.
`exp_defs/audioset_params.yaml` is only the argparse schema.

## Local launch

Activate the environment and run:

```bash
./run.sh
```

Temporary overrides are supported without editing other files:

```bash
./run.sh --batch-size 2 --epochs 1 --train-percent-check 0.001
```

## Slurm

Create the outer log directory before submitting:

```bash
mkdir -p logs
```

Smoke tests:

```bash
sbatch slurm/audioset_model1_smoke_ailab01.sbatch
sbatch slurm/audioset_model1_smoke_ailab03.sbatch
```

Full training:

```bash
sbatch slurm/audioset_model1_train_ailab01.sbatch
sbatch slurm/audioset_model1_train_ailab03.sbatch
```

Submit from the repository root. The scripts use `SLURM_SUBMIT_DIR` as the
project directory, so the local folder name can change without editing them.

## Verification

```bash
python -m unittest \
  tests.test_config_parser \
  tests.test_creator_isolation \
  tests.test_audioset_pipeline \
  -v
```

## Relevant files

- `datasets/paths/audioset_paths.py`: joins strong annotations, weak labels,
  and audio files.
- `datasets/audioset_ri_dataset.py`: produces RI features and strong targets.
- `nets/audioset_crnn.py`: weak-label-conditioned frame classifier.
- `loss_functions/audioset_loss.py`: masked strong frame BCE.
- `creators/`: isolated constructors and task/collate adapters.
- `prepare_audioset_strong_subset.py`: optional strongly annotated subset
  preparation utility.
