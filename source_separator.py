import os
import torch
import pytorch_lightning as pl
import random
import numpy as np
from torch import optim
import torch.utils.data as data
from utils.collate import collate_seq
from loguru import logger
from creators.dataset import dataset_creator
from creators.loss import loss_creator
from creators.model import model_creator
import torch.backends.cudnn as cudnn
from functools import partial

from utils.my_optimizer import ReduceLROnPlateau_verbose


class SourceSeparator(pl.LightningModule):

    def __init__(self, hparams, last_epoch=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = model_creator(hparams)
        self.loss = loss_creator(hparams)
        self.last_epoch = last_epoch

        self.training_step_output = [] # following lightning 2.0.0 release notes https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.0.0
        self.validation_step_output = [] # following lightning 2.0.0 release notes https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.0.0

    def forward(self, x, target_dict={}):
        return self.model(x, target_dict=target_dict, current_epoch=self.current_epoch)

    def _step(self, batch):
        input_features, target_dict = batch
        input_features, lens = input_features
        #y = self.forward(input_features, target_dict)
        y = self.forward(None, target_dict)
        loss = self.loss(y, target_dict, training=self.model.training)
        return loss

    #def on_after_backward(self):
    #    # https://github.com/Lightning-AI/lightning/issues/4956
    #    valid_gradients = True
    #    for name, param in self.named_parameters():
    #        if param.grad is not None:
    #            valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
    #            #valid_gradients = not torch.isnan(param.grad).any()
    #            if not valid_gradients:
    #                break
    #    if not valid_gradients:
    #        logger.info('detected inf or nan values in gradients. not updating model parameters')
    #        self.zero_grad()
    def on_after_backward(self):
        """
        # https://github.com/Lightning-AI/lightning/issues/4956
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                #valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                #valid_gradients = not (~torch.isfinite(param.grad)).any()
                valid_gradients = torch.isfinite(param.grad).all()
                if not valid_gradients:
                    break
        if not valid_gradients:
            logger.info('detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()
        """
        try:
            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            val = torch.nn.utils.get_total_norm(grads, norm_type=2.0, error_if_nonfinite=True, foreach=None)
        except RuntimeError:
            logger.info('detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

    def training_step(self, batch, batch_idx):
        loss_dict = self._step(batch)
        train_loss = loss_dict['loss']
        tqdm_dict = {f'train_{k}': v.detach() for k, v in loss_dict.items()}
        if self.trainer.current_epoch == 0:
            logger_strs, logger_vals = '', []
            for k, v in tqdm_dict.items():
                logger_strs += (k.replace('train_','')+'={:.3f} ')
                logger_vals.append(float(v.item()))
            logger.info(('idx={}, '+logger_strs).format(batch_idx, *logger_vals))
        if torch.isnan(train_loss):
            logger.info(('idx={}, train_loss is NaN').format(batch_idx))
            return None # skip this batch
        else:
            self.training_step_output.append({
                    'loss': train_loss.detach(),
                    'other_losses': tqdm_dict,})
            return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):
        if self.hparams.cudnn_benchmark:
            cudnn.benchmark = False
        loss_dict = self._step(batch)
        val_loss = loss_dict['loss']
        other_losses = {f'valid_{k}': v.detach() for k, v in loss_dict.items()}

        self.validation_step_output.append({
                'loss': val_loss.detach(),
                'other_losses': other_losses,})
        return {'loss': val_loss}

    def on_train_epoch_end(self):
        # what if training_step returns None
        ret = []
        trn_loss, trn_other_losses = self.average_loss_over_epoch(self.training_step_output)
        if len(trn_other_losses.keys()) == 0:
            return
        logger_strs_other_losses = ''
        logger_vals_other_losses = []
        for k, v in trn_other_losses.items():
            logger_strs_other_losses += (k.replace('train_','')+'={:.3f} ')
            logger_vals_other_losses.append(v.item())
        logger.info(('Epoch {}, Train Loss={:.3f} '+logger_strs_other_losses).format(self.trainer.current_epoch, trn_loss.item(), *logger_vals_other_losses))
        self.training_step_output.clear()

    def on_validation_epoch_end(self):
        cudnn.benchmark = self.hparams.cudnn_benchmark
        ret = 0.
        val_loss, val_other_losses = self.average_loss_over_epoch(self.validation_step_output)
        if len(val_other_losses.keys()) == 0:
            return
        logger_strs_other_losses = ''
        logger_vals_other_losses = []
        for k, v in val_other_losses.items():
            logger_strs_other_losses += (k.replace('valid_','')+'={:.3f} ')
            logger_vals_other_losses.append(v.item())
        logger.info(('Epoch {}, Valid Loss={:.3f} '+logger_strs_other_losses).format(self.trainer.current_epoch, val_loss.item(), *logger_vals_other_losses))
        ret = ret + val_loss
        self.log('val_loss', ret)
        self.validation_step_output.clear()

    def average_loss_over_epoch(self, outputs):
        loss_num = 0.
        loss_den = 0.
        other_losses_num = {}
        for output in outputs:
            loss_num += output['loss']
            loss_den += 1
            other_losses = output['other_losses']
            if loss_den == 1:
                for k, v in other_losses.items():
                    other_losses_num[k] = v
            else:
                for k, v in other_losses.items():
                    other_losses_num[k] += v
        if loss_den > 0:
            avg_loss = loss_num / loss_den
            for k in other_losses.keys():
                other_losses_num[k] = other_losses_num[k] / loss_den
        else:
            avg_loss = 0.
            other_losses_num = {}
        return avg_loss, other_losses_num

    def configure_optimizers(self):
        hparams = self.hparams
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau_verbose(
                optimizer, patience=self.hparams.patience,
                factor=self.hparams.decay_factor)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            },
        }

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, default_params=None):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        ckpt_hparams = checkpoint['hyper_parameters']

        if default_params is not None:
            import yaml
            default_hyperparams = yaml.safe_load(open(default_params))
            for k, v in default_hyperparams.items():
                if k not in ckpt_hparams:
                    ckpt_hparams[k] = v

        from argparse import Namespace
        hparams = Namespace(**ckpt_hparams)
        model = cls(hparams)
        model.load_state_dict(checkpoint['state_dict'])
        model.on_load_checkpoint(checkpoint)
        return model

    def _worker_init_fn(self, worker_id):
        random.seed(self.hparams.seed+worker_id+self.current_epoch*101)
        np.random.seed(self.hparams.seed+worker_id+self.current_epoch*101)
        torch.manual_seed(self.hparams.seed+worker_id+self.current_epoch*101)

    def _build_data_loader(self, partition):
        shuffle = self.hparams.shuffle if partition == 'tr' else None
        d = dataset_creator(self.hparams, partition)
        return data.DataLoader(
                d, self.hparams.batch_size,
                collate_fn=partial(
                    collate_seq,
                    pad_back_or_front=self.hparams.pad_back_or_front,
                    ),
                shuffle=shuffle,
                num_workers=self.hparams.num_workers,
                worker_init_fn=self._worker_init_fn,
                persistent_worker=True,
                )

    def train_dataloader(self):
        return self._build_data_loader('tr')

    def val_dataloader(self):
        return self._build_data_loader('cv')
