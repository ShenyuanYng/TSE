"""Continue one existing experiment, using its saved settings and training state."""
import argparse
import fcntl
import json
import os
from pathlib import Path

import torch

from train import main as train_main


def resume_settings(checkpoint, run_dir):
    """Keep the saved configuration, including the original data split and LR."""
    hp = argparse.Namespace(**dict(checkpoint["hyper_parameters"]))
    run_dir = Path(run_dir).resolve()
    hp.save_path = str(run_dir.parent)
    hp.save_name = run_dir.name
    hp.test_after_fit = False
    callbacks = [state for state in checkpoint.get("callbacks", {}).values()
                 if isinstance(state, dict) and "best_model_path" in state and "dirpath" in state]
    if len(callbacks) != 1:
        raise ValueError("Expected one saved ModelCheckpoint callback")
    # Reuse the old callback directory so Lightning restores best-k bookkeeping
    # instead of resetting it when TensorBoard creates a new per-job version.
    checkpoint_dir = Path(callbacks[0]["dirpath"])
    if not checkpoint_dir.is_dir() or checkpoint_dir.resolve().parent != run_dir:
        raise ValueError("Original checkpoint directory is missing or outside this run; do not move the experiment")
    return hp, str(checkpoint_dir)


def completed_epochs(checkpoint):
    # completed may lag epoch end while a ModelCheckpoint hook is saving.
    # 'processed' marks a completed training+validation epoch before that hook.
    progress = checkpoint["loops"]["fit_loop"]["epoch_progress"]["total"]
    return max(int(progress["completed"]), int(progress["processed"]))


def continue_run(run_dir):
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    # Protect against accidentally submitting two independent continuation chains.
    with (run_dir / ".resume.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkpoint = torch.load(str(run_dir / "mylast.ckpt"), map_location="cpu", weights_only=False)
        hp, checkpoint_dir = resume_settings(checkpoint, run_dir)
        done = completed_epochs(checkpoint)
        print(f"Restoring {run_dir}: completed_epochs={done}, target_epochs={hp.epochs}, "
              f"global_step={checkpoint['global_step']}", flush=True)
        if done < hp.epochs:
            del checkpoint
            train_main(hp, resume_run_dir=str(run_dir), checkpoint_dir=checkpoint_dir)
            checkpoint = torch.load(str(run_dir / "mylast.ckpt"), map_location="cpu", weights_only=False)
            done = completed_epochs(checkpoint)
            if done < hp.epochs:
                raise RuntimeError(f"Training exited before target: {done}/{hp.epochs}")
        temporary = run_dir / f".training_complete.{os.getpid()}.tmp"
        temporary.write_text(json.dumps({"completed_epochs": done, "target_epochs": hp.epochs,
                                         "job_id": os.environ.get("SLURM_JOB_ID")}, indent=2))
        os.replace(temporary, run_dir / "training_complete.json")
        print(f"Training complete: {done}/{hp.epochs}. No additional epochs needed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    continue_run(args.run_dir)
