"""Train AudioSet Strong Model 1."""

import argparse
import os.path
import random

import loguru
import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer

from source_separator import SourceSeparator
from utils.backtracking_model_checkpoint import BacktrackingModelCheckpointUpgrade
from utils.config import yaml_to_parser
from utils.mytensorboardlogger import MyTensorBoardLogger


def trainer_device_config(hparams):
    """Return portable Lightning accelerator, device, and strategy settings."""
    accelerator = getattr(hparams, "accelerator", "auto")
    requested_devices = getattr(hparams, "devices", 1)
    if accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("accelerator=gpu was requested, but CUDA is unavailable")

    if accelerator == "cpu":
        devices = 1 if requested_devices in (None, 0, "auto") else requested_devices
        strategy = "auto"
    else:
        devices = requested_devices
        if isinstance(devices, int) and devices <= 0:
            devices = 1
        strategy = getattr(hparams, "distributed_backend", "auto")
        if strategy in {"ddp", "ddp_find_unused_parameters_true"} and devices == 1:
            strategy = "auto"
    return accelerator, devices, strategy


def _checkpoint_callback(save_path):
    return BacktrackingModelCheckpointUpgrade(
        dirpath=save_path,
        filename="_ckpt_epoch_{epoch}",
        auto_insert_metric_name=False,
        monitor="val_loss",
        save_last=True,
        save_top_k=1,
        save_weights_only=False,
        mode="min",
        every_n_epochs=1,
        verbose=False,
    )


def _resume_checkpoint(callback, run_dir):
    checkpoint_path = os.path.join(run_dir, "mylast.ckpt")
    if not os.path.exists(checkpoint_path):
        return None

    info_path = os.path.join(run_dir, "mylast_best_info.pt")
    loguru.logger.info("loading mylast.ckpt from {}", checkpoint_path)
    info = torch.load(info_path, weights_only=False)
    callback._last_global_step_saved = info[0]
    callback.current_score = info[1]
    callback.best_k_models = info[2]
    callback.kth_best_model_path = info[3]
    callback.best_model_score = info[4]
    callback.best_model_path = info[5]
    callback.last_model_path = info[6]
    loguru.logger.info("last_epoch={}", info[7])
    return checkpoint_path


def main(hparams):
    if hparams.seed is None:
        hparams.seed = random.getrandbits(16)
    hparams.cudnn_deterministic = bool(hparams.cudnn_deterministic)
    hparams.cudnn_benchmark = bool(hparams.cudnn_benchmark)
    cudnn.deterministic = hparams.cudnn_deterministic
    cudnn.benchmark = hparams.cudnn_benchmark
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.set_float32_matmul_precision("high")

    tensorboard = MyTensorBoardLogger(
        save_dir=hparams.save_path,
        name=hparams.save_name,
    )
    hparams.save_path = os.path.join(
        hparams.save_path,
        tensorboard.name,
        f"version_{tensorboard.version}",
    )
    run_dir = os.path.dirname(hparams.save_path)
    checkpoint = _checkpoint_callback(hparams.save_path)
    resume_from = _resume_checkpoint(checkpoint, run_dir)
    accelerator, devices, strategy = trainer_device_config(hparams)

    trainer = Trainer(
        min_epochs=hparams.epochs,
        max_epochs=hparams.epochs,
        logger=tensorboard,
        limit_train_batches=hparams.train_percent_check,
        limit_val_batches=hparams.val_percent_check,
        enable_checkpointing=True,
        callbacks=[checkpoint, pl.callbacks.ModelSummary(max_depth=5)],
        default_root_dir=hparams.save_path,
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        gradient_clip_val=hparams.gradient_clip_val,
        enable_progress_bar=False,
        benchmark=hparams.cudnn_benchmark,
        deterministic=hparams.cudnn_deterministic,
        num_sanity_val_steps=10,
    )
    trainer.fit(SourceSeparator(hparams), ckpt_path=resume_from)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-def",
        type=str,
        default="exp_defs/audioset_params.yaml",
    )
    args, other_options = parser.parse_known_args()
    hyperparams = yaml_to_parser(args.exp_def).parse_args(other_options)
    main(hyperparams)
