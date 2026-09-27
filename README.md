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
which is concatenated with the temporal audio representation.
Targets and outputs are `[B,T]` for
that condition only; weak labels are never copied into the frame targets.

Features use `torchaudio.transforms.MelSpectrogram`, followed by a floor of
`1e-10` and the natural logarithm. The library handles STFT, power and Mel
filtering; the output is transposed from `[n_mels,T]` to `[T,n_mels]`.
These are natural-log Mel power spectra (HTK scale, 64 triangular filters,
0–8 kHz, floor 1e-10), extracted at 16 kHz with a periodic Hann window of 512
samples (32 ms) and hop 256 (16 ms), giving 50% overlap. STFT uses centered
constant padding; frame centers are `t * 256 / 16000`, with a crop offset during
training. A ten-second clip has 626 frames, including the final frame centered
exactly at 10 seconds. Frames at/after audio end
are excluded from supervision. Dataset features are `[T,64]`, collated as
`[B,T,64]`; CNN adds a feature channel `[B,1,T,64]`. There are no microphone
or real/imaginary dimensions. Learning rate defaults to 0.001 everywhere.

`requirements.txt` pins `torchaudio==2.8.0` to match `torch==2.8.0`. Install the
matching TorchAudio CPU/CUDA build in the training environment before launching.
The existing `run.sh` feature parameters still control the library transform.

## Class embedding and fusion

The one-hot condition multiplies a learned embedding matrix `[C,D]` to produce
`[B,D]`. Its width `D` is independent of the audio representation and is configured
with `audioset_embedding_dim` in `run.sh` (default 128). The embedding is expanded
across time and concatenated with the normalized BiGRU output. A linear layer
projects the combined features back to the audio width before the existing SiLU,
LayerNorm and frame head. CNN and fusion dropout layers are removed, and GRU
inter-layer dropout is fixed at zero.

With the default settings, the shapes are:

```text
logmel [B,T,64] -> CNN + frequency mean [B,T,96]
                -> BiGRU + LayerNorm   [B,T,256] --------+
                                                        concat -> [B,T,384]
one-hot [B,527] @ embedding matrix [527,128]             |             |
                -> [B,128] -> expand to [B,T,128] -------+             v
                                                     Linear(384,256), SiLU,
                                                     LayerNorm
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

Official Strong Train supplies training and validation. Before condition
expansion, all usable clips are grouped by YouTube video ID. Video IDs are
ordered by a SHA-256 hash of seed and ID; 15% (rounded to the nearest integer,
with at least one video in each partition) are held out for validation and the
remaining 85% are used for training. All clips and class conditions from one
video stay together. These percentages refer to independent videos; clip and
class counts may differ from 85/15. No label stratification is applied.

The defaults are `train_val_fraction=0.15`, `train_split_seed=7` and
`train_split_path=metadata/train_val_split.json`, relative to the data root.
The fixed clip lists are persisted and reused; changing the inventory, fraction
or seed requires a new split path. The old `metadata/eval_split.json` is not used.
Video overlap between any two partitions is rejected, and startup logs report
both video and clip counts.

Official Strong Eval is kept entirely as test, without further splitting.
Configure its sources with `test_audio_dir`, `test_weak_manifest` and
`test_strong_manifest` (CLI: `--test-audio-dir`, etc.); these replace the old
`val_*` source settings. Validation always uses the `train_*` sources.
Machine profiles use `eval_wav` for test. As before, usable clips must have
matching strong labels, weak labels and audio; `audioset_strict_paths=false`
skips missing matches, so retaining all eval means all available usable eval clips.

Start a new experiment from scratch for this split. Old checkpoints trained on
the full Strong Train have already seen the new validation videos; resuming
their legacy configuration is rejected rather than silently changing holdouts.

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

Validation additionally logs `val_frame_accuracy`, `val_frame_precision`,
`val_frame_recall`, and `val_frame_f1`. Precision, recall and F1 are computed from
the full epoch's TP/FP/FN/TN counts after deduplication, never averaged over
batches. Their zero-denominator values are 0 when valid frames exist.

Only TP/FP/FN/TN counts are stored for each clip-condition pair. Distributed
evaluation gathers these counts and deduplicates repeated pairs. Results, counts
and threshold are saved as `val_frame_accuracy.json` / `test_frame_accuracy.json`.
Counts reset after each epoch. If no valid frames exist, JSON metrics are null
and logged metrics are NaN.
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

## Validation threshold sweep (no training)

Use the `val_loss` best checkpoint recorded in an existing experiment:

```bash
python -u sweep_validation_thresholds.py \
  --run-dir experiments/audioset_strong/EXPERIMENT \
  --device cuda \
  --output-dir outputs/threshold_sweep_best
```

`--run-dir` reads `mylast.ckpt` only to locate the saved `best_model_path`;
the actual predictions use the best checkpoint's weights. Alternatively, pass
`--checkpoint /path/to/best.ckpt` instead of `--run-dir`.
Use the same compute node/data paths as the original run. For a run trained on
ailab01, request a separate validation allocation from the project directory:

```bash
srun --partition=L40 --nodelist=ailab01 --gres=gpu:1 \
  --cpus-per-task=6 --mem=32G --time=02:00:00 \
  /home/$USER/anaconda3/envs/audioset/bin/python -u sweep_validation_thresholds.py \
  --run-dir experiments/audioset_strong/EXPERIMENT --device cuda \
  --output-dir outputs/threshold_sweep_best
```

The default thresholds are `0.3 0.4 0.5 0.6 0.7 0.8 0.9`. The script runs in
evaluation/inference mode and traverses the full saved validation split once.
Each batch produces one `frame_probs` tensor shared by all thresholds. By default
only TP/FP/FN/TN counts are kept, using the existing `strong_mask & frame_mask`,
clip-condition deduplication and epoch-level metric formulas. No loss, optimizer,
training loop or checkpoint write is involved, and the configured threshold stays
unchanged. Missing split files or a changed clip inventory cause an error.

It prints Precision/Recall/F1/Accuracy for every threshold and writes
`threshold_sweep.json` and `threshold_sweep.csv`, including confusion counts.
The JSON also records the chosen checkpoint, epoch, validation split hash and
number of prediction batches. Defaults reuse the checkpoint's batch size and
worker count; `--batch-size` and `--num-workers` may override inference resources.
Zero denominators and empty supervision follow the existing validation metrics.
At threshold 0.5, compare with the training log at the **best checkpoint's epoch**,
which may differ from the final epoch. The sweep does not select or apply a new
training threshold automatically.

## Per-class and Macro F1 from a fixed checkpoint

Add `--per-class` to the same validation command. This mode fixes the threshold
at **0.7**, runs the saved validation set once, and keeps the model weights fixed:

```bash
python -u sweep_validation_thresholds.py \
  --run-dir experiments/audioset_strong/EXPERIMENT \
  --per-class --device cuda \
  --output-dir outputs/validation_per_class_0.7
```

`--run-dir` selects the recorded `val_loss` best checkpoint as above; use
`--checkpoint /path/to/exact.ckpt` to evaluate an explicit frozen checkpoint.
The optional `--thresholds` argument must be `0.7` in this mode. On Slurm, use
the same `srun` allocation shown above, adding `--per-class` to the Python command.

After global clip-condition deduplication, TP/FP/FN/TN are summed separately for
each class, and Precision/Recall/F1 are computed from those full-validation
counts. Micro Precision/Recall/F1/Accuracy still use the pooled counts over all
valid frames. Macro F1 is the **unweighted mean** of class F1 values for classes
with `valid_frames > 0`; neither batch nor rank F1 values are averaged.
Negative-only classes participate with F1 = 0 when its denominator is zero.
Classes with no valid frames are excluded. If none are valid, `macro_f1` is null
and `valid_class_count` is 0.

The console prints the Micro metrics, `macro_f1`, `valid_class_count`, and the
per-class table. The output directory contains:

- `per_class_metrics.csv`: every class in the checkpoint-configured class map,
  with MID, class name, class index, TP/FP/FN/TN, Precision/Recall/F1/Accuracy,
  valid frame count, and `included_in_macro`. Classes without valid frames have
  zero counts and blank metric cells, so they cannot be mistaken for zero-F1
  evaluated classes.
- `threshold_sweep.csv`: the single threshold's Micro metrics, Macro F1, valid
  class count, and pooled confusion counts.
- `threshold_sweep.json`: the same summary plus per-class rows, checkpoint and
  validation split provenance, and the Macro F1 inclusion rule.

This only extends standalone checkpoint evaluation; training hooks, loss,
optimizer, saved weights and training threshold are unchanged.

Per-class evaluation also exports Strong-positive class statistics, using
`positive_frames = TP + FN > 0` and `predicted_positive_frames = TP + FP` on the
same supervised valid frames. The number of classes is computed from the data,
not fixed at 353. Both frame counts retain the original threshold, masking and
DDP deduplication. Four additional CSV files are written:

- `strong_positive_class_metrics.csv`: MID, class_name, positive_frames,
  predicted_positive_frames, TP, FP, FN, Precision, Recall, F1.
- `zero_f1_classes.csv`: all Strong-positive classes with F1 = 0, plus a
  `zero_f1_reason` column (`no_predictions` or `predicted_but_tp_zero`).
- `zero_f1_no_predictions.csv`: `TP + FP = 0` (the model predicts no valid frame
  positive for this class at the evaluation threshold).
- `zero_f1_predicted_but_tp_zero.csv`: `TP = 0` and `FP > 0` (predictions exist,
  but none match positive labels).

`positive_class_summary.json` records the group sizes, mean F1 over just the
Strong-positive classes, definitions and provenance. These groups exclude
classes with no Strong-positive frames, even if those classes have false positives.
Zero F1 is identified from exact counts, not rounded CSV metric values.

Existing results can be summarized without loading a checkpoint or PyTorch:

```bash
python summarize_positive_classes.py \
  --input-csv outputs/validation_per_class_0.7/per_class_metrics.csv \
  --output-dir outputs/strong_positive_classes_0.7
```

## Probability diagnostics for classes with no predictions

Use the original **per-class** `threshold_sweep.json` to select classes with
`TP = FP = 0` and `FN > 0`. For the job 37074 report at **0.7**, this selects 76
classes. This does **not** mean that they also have no predictions at 0.5.
The count-only CSVs cannot reconstruct probability means, maxima or percentiles;
run the frozen best checkpoint once on the same saved validation split:

```bash
python -u sweep_validation_thresholds.py \
  --run-dir experiments/audioset_strong/EXPERIMENT \
  --device cuda --thresholds 0.5 0.7 \
  --probability-reference outputs/validation_per_class_0.7/threshold_sweep.json \
  --diagnostic-threshold 0.5 \
  --output-dir outputs/no_prediction_probability_37074
```

Use the `srun` allocation shown above on Slurm. The reference report's checkpoint
epoch, version/checkpoint filename, split hash, validation inventory and class map
must match. After inference, each selected class must reproduce the reference
valid-frame/positive-frame counts and zero predictions at the selection threshold.
Mismatch raises an error instead of silently analyzing a different population.

This optional evaluation keeps valid frame probabilities and GT-positive flags
only for the selected classes on CPU. It reuses the batch's existing `frame_probs`,
applies `strong_mask & frame_mask`, and merges DDP objects using the same
`(clip, class_index)` deduplication as the existing metrics. Means and exact P95
(linear interpolation) are computed over **all** deduplicated class frames, never
averaged across batches, clips or ranks. The retained arrays are reset after
computation. Training's counts-only metrics, model and loss are unchanged.

Additional outputs, ordered by descending `positive_frames` (MID breaks ties):

- `no_prediction_probability_stats.csv`: all selected classes, with MID/name,
  positive/valid frame counts, all-valid max/mean/P95, GT-positive mean/max,
  GT-negative mean and the positive-minus-negative mean gap. Selection and
  diagnostic thresholds are explicit columns. TP/FP/FN, predicted-positive frame
  count and Precision/Recall/F1 refer to the diagnostic threshold (default 0.5).
- `no_prediction_probability_top20.csv`: the 20 classes with the most positive
  frames, including the same probability statistics; also printed to the console.
- `no_prediction_probability_summary.json`: reference provenance, threshold
  observation counts and statistic definitions. These are also recorded under
  `probability_diagnostics` in the new run's `threshold_sweep.json`.

`threshold_observation` reports `gt_positives_recovered`, `only_false_positives`
or `all_scores_below_threshold` at the diagnostic threshold. Low probabilities
alone cannot establish that a class was not learned. Compare positive and
negative frame scores and the resulting precision/recall/F1 before attributing
failure to the threshold. A missing negative-frame mean stays blank, not zero.

## Ontology diagnostic: other classes on target GT-positive frames

`diagnose_ontology_confusions.py` ranks the other model classes on the same
GT-positive frames of a target class. The default targets are Speech, Animal,
Vehicle, Ocean, Narration/monologue, Rail transport and Emergency vehicle.
Use `--target-mids` to supply a different explicit list of MIDs.

```bash
python -u diagnose_ontology_confusions.py \
  --run-dir experiments/audioset_strong/EXPERIMENT \
  --reference-report probability_reference_37074.json \
  --ontology resources/audioset_ontology.json \
  --device cuda \
  --output-dir outputs/ontology_confusion_37074
```

The reference is the original **per-class** `threshold_sweep.json`, not the
later two-threshold probability summary. It fixes checkpoint identity/epoch,
saved split hash, validation membership, class map and expected positive/valid
frame counts. The full original validation loader and reference batch size are
retained, because the encoder's GroupNorm also sees padded time positions.
No sampling or ontology propagation changes the evaluation population.

This checkpoint has a conditional binary output, not a 527-channel output.
The diagnostic runs the normal forward once per batch and observes its
`temporal_norm` audio features with a temporary hook. It then scores all 527
one-hot conditions through the existing embedding/fusion/frame head, exclusively
on `strong_labels >= 0.5 & strong_mask & frame_mask` for each selected target.
The target's rescored probabilities must match the normal forward within
`atol=1e-4, rtol=1e-3`; the maximum observed difference is reported. Model code,
weights, loss and training hooks are unchanged. No softmax, probability
renormalization or binary threshold is applied to the ranking.

`--class-chunk-size 32` and `--frame-chunk-size 256` bound temporary head tensors.
They do not alter the full encoder batch size or padding. Only sums, maxima and
annotation/count evidence per `(clip, target MID)` survive each batch. These
summaries are deduplicated across DDP ranks before class-level aggregation and
reset after computation. Ties in mean probability use MID order; frame-level
winner ties use model class index. The target itself is excluded from Top-10.

The fixed official ontology snapshot, source attribution, license and SHA-256
are under `resources/`. Relationships use the full 632-node directed graph,
including intermediate nodes outside the model's class map and multiple parents.
Relation labels describe the **other class relative to the target**:
`other_is_child`, `other_is_descendant`, `other_is_parent`, `other_is_ancestor`,
`no_lineage_relation`, or `unknown_mid`. A shared ancestor alone does not make
two classes ancestors/descendants of each other. Shared direct parents are
reported separately, and relationship paths run from parent to child.

Outputs:

- `ontology_confusion_top10.csv`: 10 other classes per target (70 rows for the
  seven defaults), ordered by their **frame-weighted mean probability over all
  target GT-positive frames**. Includes target and other MID/name, positive frame
  and clip counts, target/other mean and maximum, mean gap, fraction of frames
  where the other score exceeds the target, frame-level Top-1 frequency among
  other classes, ontology relationship, distance and path.
- The same CSV includes weak-label presence (clip count and covered target
  positive frames) and exact-MID strong annotation overlap on those target frames.
  `NOT_PRESENT` overrides `PRESENT`, as in the dataset. Original float64 frame
  centers are reconstructed from sample indices to avoid collated float32 time
  rounding at annotation boundaries. No alternate-class mask restricts scoring.
- `ontology_confusion_all_classes.csv`: the full ranking of all other model
  classes, allowing alternative views without another inference run.
- `ontology_target_summary.csv`: support and target probabilities, Top-10
  lineage-related/unknown class counts and numerical comparison per target.
- `ontology_confusion_summary.json`: checkpoint/split/ontology provenance,
  inference settings, definitions and interpretation limits.

High alternate probabilities plus a lineage relationship are diagnostic evidence,
not proof of a strong/weak label error: simultaneous sounds can legitimately
co-occur, and independently conditioned scores need not be calibrated across
classes. This analysis includes only exact-MID GT-positive frames, so it cannot
explain all weak-label targets lacking a same-MID PRESENT annotation. Do not mix
counts from a different split/checkpoint, including earlier matching audits.

For the existing job 37074 checkpoint on ailab01, after syncing the new files:

```bash
cd ~/projects/new_example_code
mkdir -p outputs/ontology_confusion_37074
sbatch slurm/audioset_ontology_diagnostic_ailab01.sbatch
# Replace JOBID with the number returned by sbatch:
tail -F outputs/ontology_confusion_37074/slurm-JOBID.log
```

The Slurm script only performs validation diagnostics and writes reports/logs.
All scoring inputs must already exist on the compute node; ontology resolution
uses the local snapshot and needs no network access there.

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

## Export aligned audio for Audacity

Both prediction scripts now automatically export three WAV files per sample in
an `audio/` folder beside the HTML report: `*_original.wav`, `*_predicted.wav`,
and `*_ground_truth.wav`. Predictions preserve the original audio where frame
probability reaches the threshold; ground truth preserves audio inside the raw
same-MID PRESENT annotation intervals (NOT_PRESENT intervals take precedence).
All other samples are zero. Files retain the full source length, sample rate,
channels and amplitude; they are written as float WAV without normalization,
resampling, concatenation or fades. This is time gating, not source separation.

Run on the training server with a checkpoint from the new split:

```bash
python inspect_matched_predictions.py \
  --checkpoint experiments/EXPERIMENT/mylast.ckpt \
  --partition tr cv --num-samples 20 --seed 7 \
  --threshold 0.5 --device cpu \
  --output-dir outputs/matched_audio_01
```

The output directory must be new or empty. The old `--copy-audio` flag is still
accepted but is no longer required. An example set is
`cv_004_0.78_original.wav`, `cv_004_0.78_predicted.wav`, and
`cv_004_0.78_ground_truth.wav`; the number is that sample's frame accuracy.
Download the whole output folder and import all three WAVs for the same sample
into one Audacity project, aligned at time zero. Compare the tracks in spectrogram
view with the same display settings. HTML reports also provide playback and
download links; `summary.json` records the relative audio paths.

The ground-truth audio uses raw annotation timestamps, while the HTML truth
curve is aligned to feature frames and can differ by less than one frame at a
boundary. Predictions are exported on all valid audio frames regardless of
strong-label masks; accuracy is still calculated only on supervised frames.

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
  --run-dir experiments/audioset_strong/AudioSetCRNN_logmel_embedding_concat_d128_hop0.016_h128_rnn2_bs8_chunk10.0_lr0.001_job36595
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
