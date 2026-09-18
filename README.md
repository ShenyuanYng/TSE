# AudioSet Strong Model 1

This repository trains one weak-label-conditioned CRNN against real frame-level
AudioSet Strong annotations. It intentionally contains no weak-only training,
pseudo-label generation, Model 2, or speech-separation implementation.

## Framework

```text
run.sh                         all settings, machine paths and Slurm submission
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

Each clip with weak labels `[A,B,C]` becomes three examples: `mix+A`,
`mix+B`, `mix+C`. Each one-hot condition selects a learned class embedding,
which is multiplied elementwise with the temporal audio representation.
Targets and outputs are `[B,T]` for
that condition only; weak labels are never copied into the frame targets.

Features are natural-log Mel power spectra (HTK scale, 64 triangular filters,
0–8 kHz, floor 1e-10), extracted at 16 kHz with a periodic Hann window of 512
samples (32 ms) and hop 256 (16 ms), giving 50% overlap. STFT uses centered
constant padding; frame centers are `t * 256 / 16000`, with a crop offset during
training. A ten-second clip has 626 frames, including the final frame centered
exactly at 10 seconds. Frames at/after audio end
are excluded from supervision. Dataset features are `[T,64]`, collated as
`[B,T,64]`; CNN adds a feature channel `[B,1,T,64]`. There are no microphone
or real/imaginary dimensions. Learning rate defaults to 0.001 everywhere.

## Class embedding and fusion

The one-hot condition multiplies a learned embedding matrix `[C,D]` to produce
`[B,D]`. Its width `D` is independent of the audio representation and is configured
with `audioset_embedding_dim` in `run.sh` (default 128). The embedding is expanded
across time and concatenated with the normalized BiGRU output. A linear layer
projects the combined features back to the audio width before the existing SiLU,
LayerNorm, Dropout and frame head.

With the default settings, the shapes are:

```text
logmel [B,T,64] -> CNN + frequency mean [B,T,96]
                -> BiGRU + LayerNorm   [B,T,256] --------+
                                                        concat -> [B,T,384]
one-hot [B,527] @ embedding matrix [527,128]             |             |
                -> [B,128] -> expand to [B,T,128] -------+             v
                                                     Linear(384,256), SiLU,
                                                     LayerNorm, Dropout
                                                             |
                                                     frame head -> [B,T]
```

The audio width is `2 * audioset_temporal_hidden` (256 by default); the fusion
layer maps `audio_width + D` to `audio_width`, so changing either dimension works.
The embedding is learned jointly through strong BCE. Inputs remain single-class
one-hot conditions.
The masked, positive/negative-balanced BCE is unchanged: `loss` is the training
scalar and `strong_bce` is its detached logging copy, so their values agree.

## Validation and held-out test

The official eval manifests and audio directories supply both validation and
test. Before training or condition expansion, eval is split by YouTube video ID
with deterministic multilabel balancing (target test fraction 0.5, seed 7).
Exact balance is not guaranteed for rare labels or large video groups. The
partition is persisted in `metadata/eval_split.json` under the data root.
Changing the inventory, fraction or seed requires a new split path. Train/eval
video overlap is rejected. Machine-specific defaults now use `eval_wav` and
official eval manifests; adjust these paths to match local storage.

Validation controls checkpoint selection and learning-rate scheduling through
`val_loss`. Test data is never used by fitting or checkpoint selection.
`./run.sh --test-after-fit true` evaluates the best validation-loss checkpoint
on the held-out test set after the final chosen experiment; the default is false.

Validation/test report `val_conditional_frame_accuracy` /
`test_conditional_frame_accuracy`. A frame is predicted present when its sigmoid
probability is at least `frame_accuracy_threshold` (default 0.5, set in `run.sh`).
Accuracy is the number of correctly predicted valid frames divided by the total
valid frames across the split, including both positive and negative labels. It is
not an average of batch or class accuracies. Padding and unknown positions are
excluded using `strong_mask & frame_mask`. Classes without positives still count
if they have supervised negative frames.

Only correct/valid counts are stored for each clip-condition pair. Distributed
evaluation gathers these counts and deduplicates repeated pairs. Results, counts
and threshold are saved as `val_frame_accuracy.json` / `test_frame_accuracy.json`.
If no valid frames exist, JSON accuracy is null and the logged metric is NaN.
The old mAP implementation and logging code are commented out for reference;
no AP computation or AP report runs during validation or test.

This is frame accuracy conditioned on a weak-label target class. The configured
class map defines the vocabulary; unmapped weak/strong MIDs are reported as
warnings. Set `strong_annotation_complete=false` for sparse PRESENT/NOT_PRESENT
annotations to mask unspecified positions.

The new concatenation model requires a fresh experiment: old elementwise-product
or earlier RI/concatenation checkpoints are incompatible. Keep old runs on their
original code. Launcher names include `logmel_embedding_concat_d128_hop0.016`
(with the configured dimension and hop) to distinguish this architecture.

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
./run.sh --batch-size 2 --epochs 1 --train-percent-check 0.001 \
  --audioset-embedding-dim 64 --frame-accuracy-threshold 0.5
```

## Slurm

Edit `run.sh` only: training settings (`learning_rate`, `batch_size`, `epochs`,
`num_jobs`, `tag`, etc.), `python_bin` if needed, and these scheduler settings:

```bash
slurm_node="ailab01"          # or ailab03
slurm_partition=""           # auto from node; explicit values override
slurm_gres=""                # auto from node; explicit values override
slurm_cpus_per_task=6
slurm_mem="32G"
slurm_time="2-00:00:00"
slurm_account=""             # scheduler default
slurm_job_name="audioset-embedding"
```

Then submit every experiment with the same command:

```bash
./run.sh --submit true
```

The entry point creates `logs/` and submits the existing `spksep_run.sh` directly
to Slurm. No new sbatch file is written. The requested node determines machine
paths even when submission happens on a login host. Python defaults to
`/home/$USER/anaconda3/envs/audioset/bin/python` on the two ailab profiles;
set `python_bin` in `run.sh` or export `PYTHON_BIN` to override it. Local launches
on other hosts use `python` from the active environment.

All resolved training and continuation resource parameters are passed as
arguments at submission time; Slurm also captures the Python environment
selection. Later edits to `run.sh` do not change a queued job's settings.
Source code and data are still read from the shared project when the job runs.
Every Slurm run gets a `_jobJOBID` suffix to avoid checkpoint/log collisions.
The old `slurm/audioset_*.sbatch` files are historical templates, not the
submission entry point; their embedded overrides do not apply to this command.

For a smoke run, change these values in `run.sh` before using the same command:

```bash
epochs=1
num_jobs=1
batch_size=2
train_percent_check=10        # integer = number of batches
val_percent_check=10
slurm_time="00:30:00"
tag="_smoke"
```

Restore the desired full-training values in `run.sh` for the next experiment.

## Verification

```bash
python -m unittest \
  tests.test_config_parser \
  tests.test_creator_isolation \
  tests.test_audioset_pipeline \
  tests.test_audioset_metrics_split \
  tests.test_audioset_training \
  tests.test_run_submission \
  tests.test_continuation_submission \
  -v
```

## Inspect one prediction

Compare a trained checkpoint's frame probabilities with the real strong label
for one validation example:

```bash
python inspect_prediction.py \
  --checkpoint experiments/EXPERIMENT/version_N/_ckpt_epoch_N.ckpt \
  --partition cv \
  --output prediction_comparison.html
```

The default selection is an example with positive strong frames. Use
`--sample-index`, or combine `--clip-id` and `--class-mid`, to inspect a
specific clip-condition pair. The HTML report plots the probability curve and
true/predicted intervals and frame accuracy; a CSV beside it contains every frame.
The threshold defaults to the checkpoint setting; use `--threshold` to override it. Use the held-out
`test` partition only for the final selected experiment.

## Relevant files

- `datasets/paths/audioset_paths.py`: joins strong annotations, weak labels,
  and audio files.
- `datasets/audioset_logmel_dataset.py`: produces logMel and single-class targets.
- `nets/audioset_crnn.py`: weak-label-conditioned frame classifier.
- `loss_functions/audioset_loss.py`: masked strong frame BCE.
- `creators/`: isolated constructors and task/collate adapters.
- `prepare_audioset_strong_subset.py`: optional strongly annotated subset
  preparation utility.
- `inspect_prediction.py`: compares one checkpoint prediction with its strong
  frame target.

## Automatic multi-job training

Set `num_jobs=3` in `run.sh`, then run `./run.sh --submit true` once. This
means **three jobs in total**, including the first; the default `num_jobs=1`
keeps a single job. The first allocated job queues its `num_jobs - 1`
continuations before training starts. Each continuation inherits the partition,
GPU, CPU, memory, account and time settings from `run.sh` (two days by default),
uses the same experiment, and resumes the checkpoint until the saved total
epoch target is reached. The single existing `slurm/continue_training.sbatch`
is reused for every continuation; no per-experiment scripts are generated.

`num_jobs` is a launcher setting in `run.sh`, not a model hyperparameter in
YAML. It requires Slurm for values above one. Continuations bypass the launcher,
so they cannot recursively submit new chains. Changing this file cannot add
jobs to a process already running; attach a chain once as below instead.

## Continue an existing two-day Slurm run

This applies to runs trained with the current configurable concatenation model.
Keep earlier architectures on their original checkout. While a current run is
still active, a continuation chain can also be attached manually once:

```bash
python slurm/submit_continuation.py \
  --after-job 36595 --additional-jobs 3 --node ailab03 \
  --run-dir experiments/audioset_strong/AudioSetCRNN_logmel_embedding_concat_d128_hop0.016_h128_rnn2_drop0.2_bs8_chunk10.0_lr0.001_job36595
```

`--additional-jobs` is the number of **additional** jobs, excluding the running job.
For manual attachment, resource overrides are available via `--partition`,
`--gres`, `--cpus-per-task`, `--mem`, `--time` and `--account`.
Here the three additional jobs each have the default two-day limit and use sequential
`afterany` dependencies, so a TIMEOUT releases the next job too. They still
wait for cluster resources. Use the activated training Python environment;
`--python-bin /path/to/python` can override it. Keep the same compute node and
project directory because checkpoint settings retain the original data paths.
The script checks predecessor ownership, node and working directory and records
submitted IDs in `continuation_jobs.json`. A duplicate invocation is rejected.
If submission partially fails, inspect that record before taking any action.

Continuation bypasses `run.sh` to restore the actual saved hyperparameters,
model, Adam state, scheduler, best-checkpoint state and epoch progress from
`mylast.ckpt`. The `.pt` sidecar is no longer required for recovery. Best-model
files retain the original checkpoint directory; TensorBoard and metric outputs
use a new version directory for each job. Do not rename the original experiment
or delete its checkpoint version directory. Test evaluation remains disabled.

Each continuation writes `logs/training_jobJOBID.log` inside the original
experiment; Slurm startup logs are in project `logs/continue-JOBID.out/.err`.
It continues to the saved total epoch target (e.g. 50), not 50 extra epochs.
Once complete, `training_complete.json` is written; further queued jobs check
the checkpoint and exit without training. Ordinary training exceptions create
`continuation/failed`, so later jobs skip training pending investigation.
Timeout or SIGKILL recovery uses the latest completed-epoch checkpoint; the
unfinished epoch can be replayed. This does not implement exact mid-batch resume.
No existing running job is cancelled, resubmitted, or modified by submission.

Verify continuation locally or on the server with:

```bash
python -m unittest tests.test_audioset_training tests.test_continuation_submission -v
```

The integration test verifies an actual one-epoch checkpoint resumes to a total
of two epochs, preserves Adam step counts, works without the legacy sidecar,
and performs no more optimizer steps when invoked after completion. Slurm
submission dependencies are tested with simulated scheduler responses.
