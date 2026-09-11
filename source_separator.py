"""Creator-driven Lightning training module for AudioSet Strong."""

import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from loguru import logger
from torch import optim

from creators.collate import collate_creator
from creators.dataset import dataset_creator
from creators.loss import loss_creator
from creators.model import model_creator
from creators.task import task_creator
from utils.my_optimizer import ReduceLROnPlateau_verbose


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

    def forward(self, model_input, target_dict=None):
        target_dict = {} if target_dict is None else target_dict
        #进入任务适配器
        return self.task.forward(
            self.model,#又会进入mode
            model_input,
            target_dict,
            current_epoch=self.current_epoch,
        )

    """
    完整路径是：
    SourceSeparator.forward
    输入：
    mix_ri [B,T,1,M,2,F]
    weak_labels [B,527]
        ↓
    AudioSetStrongTask.forward
    不改变数据
        ↓
    AudioSetCRNN.forward
    执行 CNN、GRU、弱标签融合
        ↓
    返回预测字典
    """

    def _step(self, batch):
        #进入损失函数时的数据先拆开batch
        model_input, target_dict = batch
        predictions = self.forward(model_input, target_dict)
        #然后调用损失函数
        return self.loss(
            predictions,
            target_dict,
            training=self.model.training,
        )

    def on_after_backward(self):
        try:
            grads = [parameter.grad for parameter in self.model.parameters()
                     if parameter.grad is not None]
            torch.nn.utils.get_total_norm(
                grads,
                norm_type=2.0,
                error_if_nonfinite=True,
                foreach=None,
            )
        except RuntimeError:
            logger.info("detected inf or nan gradients; skipping this update")
            self.zero_grad()

    def training_step(self, batch, batch_idx):
        loss_dict = self._step(batch)
        train_loss = loss_dict["loss"]
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

        if torch.isnan(train_loss):
            logger.info("idx={}, train_loss is NaN", batch_idx)
            return None

        self.training_step_output.append(
            {"loss": train_loss.detach(), "other_losses": logged_losses}
        )
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        del batch_idx
        if self.hparams.cudnn_benchmark:
            cudnn.benchmark = False
        loss_dict = self._step(batch)
        val_loss = loss_dict["loss"]
        logged_losses = {
            f"valid_{name}": value.detach()
            for name, value in loss_dict.items()
        }
        self.validation_step_output.append(
            {"loss": val_loss.detach(), "other_losses": logged_losses}
        )
        return {"loss": val_loss}

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
        cudnn.benchmark = self.hparams.cudnn_benchmark
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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
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

    def _build_data_loader(self, partition):
        dataset = dataset_creator(self.hparams, partition)
        return data.DataLoader(
            dataset,
            self.hparams.batch_size,
            collate_fn=self.collate_fn,
            shuffle=bool(self.hparams.shuffle) if partition == "tr" else False,
            num_workers=self.hparams.num_workers,
            worker_init_fn=self._worker_init_fn,
            persistent_workers=self.hparams.num_workers > 0,
        )
    """dataloader 取出单样本  dataset[index]
    单样本结构为：
    (
    {
        "mix_ri": np.ndarray[T, 1, 1, 2, 257],
        "weak_labels": np.ndarray[527],
    },
    {
        "strong_labels": np.ndarray[T, 527],
        "strong_mask": np.ndarray[T, 527],
        "frame_times": np.ndarray[T],
        "utt_id": str,
        "audio_path": str,
    },

    收集到batch个单样本后，调用self.collate_fn(batch)，执行collate_seq(
    batch,
    pad_back_or_front=0,
)
)

    """

    def train_dataloader(self):
        return self._build_data_loader("tr")

    def val_dataloader(self):
        return self._build_data_loader("cv")
