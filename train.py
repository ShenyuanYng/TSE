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



#checkpoint保存和恢复
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

    # Lightning restores model, optimizer, scheduler, loops and callback states
    # directly from this checkpoint. The historical .pt sidecar may be stale.
    loguru.logger.info("loading mylast.ckpt from {}", checkpoint_path)
    return checkpoint_path


def main(hparams, resume_run_dir=None, checkpoint_dir=None):
    if hparams.seed is None:
        hparams.seed = random.getrandbits(16)
    hparams.cudnn_deterministic = bool(hparams.cudnn_deterministic)
    hparams.cudnn_benchmark = bool(hparams.cudnn_benchmark)
    cudnn.deterministic = hparams.cudnn_deterministic
    cudnn.benchmark = hparams.cudnn_benchmark

    #固定随机状态，让模型初始化，数据随机裁剪等行为尽量可重复
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.set_float32_matmul_precision("high")

    tensorboard = MyTensorBoardLogger(
        save_dir=hparams.save_path,
        name=hparams.save_name,
    )

    #创建日志目录
    hparams.save_path = os.path.join(
        hparams.save_path,
        tensorboard.name,
        f"version_{tensorboard.version}",
    )

    #检查是否续训
    run_dir = os.path.dirname(hparams.save_path)
    if resume_run_dir is not None:
        run_dir = resume_run_dir
    checkpoint = _checkpoint_callback(checkpoint_dir or hparams.save_path)
    resume_from = _resume_checkpoint(checkpoint, run_dir)
    if resume_run_dir is not None and resume_from is None:
        raise FileNotFoundError(f"Resume checkpoint missing in {run_dir}")
    accelerator, devices, strategy = trainer_device_config(hparams)


    #创建lightning trainer
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

    #sourcespearator 是训练外壳
    model = SourceSeparator(hparams)

    trainer.fit(model, ckpt_path=resume_from)
    if hparams.test_after_fit:
        trainer.test(model, ckpt_path="best")


#程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-def",
        type=str,
        default="exp_defs/audioset_params.yaml",
    )
    """
    取参数，只把--exp-def取出来，知道应该读取那个YAML,剩余参数放入other_options
    """
    args, other_options = parser.parse_known_args()
    hyperparams = yaml_to_parser(args.exp_def).parse_args(other_options)
    main(hyperparams)
