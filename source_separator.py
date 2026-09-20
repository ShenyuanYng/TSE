"""Creator-driven Lightning training module for AudioSet Strong."""

import random
from argparse import Namespace

import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from loguru import logger
from torch import optim

from creators.collate import collate_creator
from creators.dataset import dataset_creator, datapath_creator
from datasets.train_split import video_id, validate_split_config
# from utils.audioset_metrics import ConditionalFrameMAP  # mAP 暂停使用
from utils.audioset_metrics import ConditionalFrameAccuracy
from creators.loss import loss_creator
from creators.model import model_creator
from creators.task import task_creator
from utils.my_optimizer import ReduceLROnPlateau_verbose

"""Lightning 根据约定自动调用 SourceSeparator 的方法。
SourceSeparator.setup()
SourceSeparator.configure_optimizers()
SourceSeparator.val_dataloader()
SourceSeparator.validation_step()   # sanity check
SourceSeparator.train_dataloader()
SourceSeparator.training_step()
自动 backward
自动 optimizer.step()
SourceSeparator.on_train_epoch_end()
SourceSeparator.validation_step()   # 正式验证
SourceSeparator.on_validation_epoch_end()
checkpoint callback
"""


class SourceSeparator(pl.LightningModule):
    """Shared training loop; concrete components come only from creators."""

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = model_creator(hparams)
        self.loss = loss_creator(hparams)
        self.task = task_creator(hparams)
        self.collate_fn = collate_creator(hparams)
        #累计每个batch的损失记录，不参与AudioSetCRNN.forward的张量计算
        self.training_step_output = []
        self.validation_step_output = []
        # self.val_map = ConditionalFrameMAP()
        # self.test_map = ConditionalFrameMAP()
        self.val_accuracy = ConditionalFrameAccuracy(hparams.frame_accuracy_threshold)
        self.test_accuracy = ConditionalFrameAccuracy(hparams.frame_accuracy_threshold)
        self.split_paths = None

    def on_load_checkpoint(self, checkpoint):
        # train.py can also auto-resume a checkpoint in an existing run directory.
        saved = Namespace(**dict(checkpoint["hyper_parameters"]))
        validate_split_config(saved)
        for name in ("data_path", "train_audio_dir", "train_weak_manifest",
                     "train_strong_manifest", "train_val_fraction",
                     "train_split_seed", "train_split_path", "test_audio_dir",
                     "test_weak_manifest", "test_strong_manifest", "audioset_strict_paths"):
            if getattr(saved, name) != getattr(self.hparams, name):
                raise ValueError(f"Checkpoint data split setting changed: {name}; "
                                 "start a new experiment from scratch")

    #进入真正模型
    def forward(self, model_input, target_dict=None):
        target_dict = {} if target_dict is None else target_dict
        #进入任务适配器
        return self.task.forward(
            self.model,#又会进入model
            model_input,
            target_dict,
            current_epoch=self.current_epoch,
        )

    #_step() 是训练、验证和测试共享的计算函数
    def _step(self, batch, metric=None):
        #进入损失函数时的数据先拆开batch
        model_input, target_dict = batch
        #模型预测
        predictions = self.forward(model_input, target_dict)
        if metric is not None:
            metric.update(predictions, target_dict)
        #然后调用损失函数，计算loss
        return self.loss(
            predictions,
            target_dict,
            training=self.model.training,
        )


    #反向传播后的钩子
    def on_after_backward(self):
        try:
            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            val = torch.nn.utils.get_total_norm(
                grads,
                norm_type=2.0,
                error_if_nonfinite=True,
                foreach=None,
            )
        except RuntimeError:
            logger.info(
                'detected inf or nan values in gradients. not updating model parameters'
            )
            self.zero_grad()

    #每个训练batch的入口，每拿到一次就调用一次
    def training_step(self, batch, batch_idx):
        self._train_batch_info = {
            "batch_idx": batch_idx,
            "clip_ids": batch[1].get("utt_id"),
            "class_indices": batch[1].get("class_index"),
        }
        loss_dict = self._step(batch)
        train_loss = loss_dict["loss"]
        if not torch.isfinite(train_loss).all():
            raise FloatingPointError(
                f"Nonfinite training loss: epoch={self.current_epoch}, "
                f"global_step={self.global_step}, batch={self._train_batch_info}, "
                f"loss={train_loss.detach().item()}"
            )
        logged_losses = {
            f"train_{name}": value.detach()
            for name, value in loss_dict.items()
        }

        if self.trainer.current_epoch == 0:
            values = [float(value.item()) for value in logged_losses.values()]
            fields = "".join(
                name.removeprefix("train_") + "={:.3f} "
                for name in logged_losses
            )
            logger.info(("idx={}, " + fields).format(batch_idx, *values))

        #每个训练batch都会保存一个不带梯度的记录
        self.training_step_output.append(
            {"loss": train_loss.detach(), "other_losses": logged_losses}
        )
        return {"loss": train_loss}


    #验证开始的入口
    def validation_step(self, batch, batch_idx):
        del batch_idx
        if self.hparams.cudnn_benchmark:
            cudnn.benchmark = False
        loss_dict = self._step(batch, self.val_accuracy)
        val_loss = loss_dict["loss"]
        logged_losses = {
            f"valid_{name}": value.detach()
            for name, value in loss_dict.items()
        }
        self.validation_step_output.append(
            {"loss": val_loss.detach(), "other_losses": logged_losses}
        )
        return {"loss": val_loss}


    #一轮结束时，对所有batch求平均
    def on_train_epoch_end(self):
        train_loss, other_losses = self._average_losses(
            self.training_step_output
        )
        if not other_losses:
            return
        fields = "".join(
            name.removeprefix("train_") + "={:.3f} "
            for name in other_losses
        )
        values = [value.item() for value in other_losses.values()]
        logger.info(
            ("Epoch {}, Train Loss={:.3f} " + fields).format(
                self.current_epoch,
                train_loss.item(),
                *values,
            )
        )
        self.training_step_output.clear()

    def on_validation_epoch_end(self):
        # self._log_map("val", self.val_map)
        self._log_accuracy("val", self.val_accuracy)
        cudnn.benchmark = bool(self.hparams.cudnn_benchmark)
        val_loss, other_losses = self._average_losses(
            self.validation_step_output
        )
        if not other_losses:
            return
        fields = "".join(
            name.removeprefix("valid_") + "={:.3f} "
            for name in other_losses
        )
        values = [value.item() for value in other_losses.values()]
        logger.info(
            ("Epoch {}, Valid Loss={:.3f} " + fields).format(
                self.current_epoch,
                val_loss.item(),
                *values,
            )
        )
        self.log("val_loss", val_loss, sync_dist=True)
        self.validation_step_output.clear()

    @staticmethod
    def _average_losses(outputs):
        if not outputs:
            return 0.0, {}

        loss = sum(output["loss"] for output in outputs) / len(outputs)
        names = outputs[0]["other_losses"]
        other_losses = {
            name: sum(output["other_losses"][name] for output in outputs)
            / len(outputs)
            for name in names
        }
        return loss, other_losses


    #创建Adam和学习率调度器
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        #学习率调度器
        scheduler = ReduceLROnPlateau_verbose(
            optimizer,
            patience=self.hparams.patience,
            factor=self.hparams.decay_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    #给dataloader worker设置随机种子
    def _worker_init_fn(self, worker_id):
        worker_seed = (
            self.hparams.seed
            + worker_id
            + self.current_epoch * 101
            + self.global_rank * 100003
        )
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    #从片段清单变成可读的样本，真正的构建dataloader
    def _build_data_loader(self, partition):
        dataset = dataset_creator(self.hparams, partition, path_list=self.split_paths[partition])
        return data.DataLoader(
            dataset,#单个样本的数据集
            self.hparams.batch_size,
            collate_fn=self.collate_fn,#将单样本合并为batch，并处理补0
            shuffle=bool(self.hparams.shuffle) if partition == "tr" else False,#训练集按配置打乱，验证和测试不打乱
            num_workers=self.hparams.num_workers,#加载数据的子进程数量
            worker_init_fn=self._worker_init_fn,#初始化加载进程的随机种子
            persistent_workers=self.hparams.num_workers > 0,#保留加载进程，避免每轮重启
        )
    def train_dataloader(self):
        return self._build_data_loader("tr")



    def val_dataloader(self):
        return self._build_data_loader("cv")

    """
    先整理数据清单，先确定train/validation/test分别有哪些样本
    """
    def setup(self, stage=None):
        if self.hparams.pad_back_or_front != 0:
            raise ValueError("AudioSet packed sequences require padding at the back")
        if self.split_paths is None:
            self.split_paths = {part: datapath_creator(self.hparams, part)
                                for part in ("tr", "cv", "test")}
            videos = {part: {video_id(e) for e in entries}
                      for part, entries in self.split_paths.items()}
            for left, right in (("tr", "cv"), ("tr", "test"), ("cv", "test")):
                if videos[left] & videos[right]:
                    raise ValueError(f"{left} and {right} share YouTube video IDs")
            logger.info("Video split sizes (Strong Train -> tr/cv; Strong Eval -> test): {}",
                        {k: len(v) for k, v in videos.items()})
            logger.info("Clip split sizes: {}", {k: len(v) for k, v in self.split_paths.items()})

    def on_validation_epoch_start(self):
        self.val_accuracy.reset()

    def on_test_epoch_start(self):
        self.test_accuracy.reset()

    def test_step(self, batch, batch_idx):
        losses = self._step(batch, self.test_accuracy)
        self.log("test_loss", losses["loss"], on_epoch=True, sync_dist=True,
                 batch_size=len(batch[1]["utt_id"]))

    def _log_accuracy(self, prefix, metric):
        if self.trainer.sanity_checking:
            metric.reset()
            return
        result = metric.compute()
        value = result["conditional_frame_accuracy"]
        # compute() already gathers and deduplicates all ranks.
        self.log(f"{prefix}_conditional_frame_accuracy",
                 value if value is not None else float("nan"), sync_dist=False)
        logger.info("{} conditional frame accuracy={}, correct/valid frames={}/{}",
                    prefix, value, result["correct_frames"], result["valid_frames"])
        if self.global_rank == 0:
            import json
            from pathlib import Path
            path = Path(self.hparams.save_path)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"{prefix}_frame_accuracy.json").write_text(json.dumps(result, indent=2))
        metric.reset()

    # 原 mAP 日志逻辑暂时注释保留。
    # def _log_map(self, prefix, metric):
    #     if self.trainer.sanity_checking:
    #         metric.reset()
    #         return
    #     value, aps = metric.compute()
    #     self.log(f"{prefix}_conditional_frame_mAP", value, sync_dist=False)
    #     self.log(f"{prefix}_mAP_classes", float(len(aps)), sync_dist=False)
    #     logger.info("{} conditional frame mAP={}, valid classes={}", prefix, value, len(aps))
    #     if self.global_rank == 0:
    #         import json
    #         from pathlib import Path
    #         path = Path(self.hparams.save_path)
    #         path.mkdir(parents=True, exist_ok=True)
    #         (path / f"{prefix}_frame_ap.json").write_text(json.dumps({
    #             "conditional_frame_mAP": value if aps else None,
    #             "valid_classes": len(aps), "per_class_AP": aps,
    #         }, indent=2))
    #     metric.reset()
    #

    def test_dataloader(self):
        return self._build_data_loader("test")

    def on_test_epoch_end(self):
        # self._log_map("test", self.test_map)
        self._log_accuracy("test", self.test_accuracy)
