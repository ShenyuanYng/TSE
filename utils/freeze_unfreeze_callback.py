import torch
from torch import optim
from pytorch_lightning.callbacks import Callback
from loguru import logger
import glob


def load_model_state_dict(model, hparams):
    checkpoints = glob.glob('experiments/'+hparams.save_name+'/version_*/_ckpt_epoch_*.ckpt')
    if len(checkpoints) == 0:
        return
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    logger.info('loading checkpoint = {}'.format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    # remove the model. prefix in pytorch_lightning
    model.load_state_dict({
            key.replace('model.', ''): value for key, value in checkpoint['state_dict'].items() if key.startswith('model.')
            })


class FreezeUnfreezeCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        hparams = pl_module.hparams
        init_epochs_net1 = hparams.init_epochs_net1
        init_epochs_net_stack = hparams.init_epochs_net_stack
        num_iters = hparams.num_iters

        current_epoch = trainer.current_epoch
        #logger.info('current_epoch = {}'.format(current_epoch))

        if init_epochs_net1 == 0 and init_epochs_net_stack == 0:
            pass

        else:

            #raise # upgraded to torch 2.0

            assert init_epochs_net1 >= 0 and init_epochs_net_stack > 0
            update_flag = False
            model = pl_module.model

            if current_epoch == 0:
                logger.info('training starts here, use init_epochs_net1={}, init_epochs_net_stack={}, num_iters={}'.format(
                            init_epochs_net1, init_epochs_net_stack, num_iters))

            if current_epoch == init_epochs_net1:
                load_model_state_dict(model, hparams) # load the best model
                logger.info('At epoch {}, set net1 to eval'.format(current_epoch))
                for param in model.net1.parameters():
                    param.requires_grad = False
                update_flag = True
                learning_rate = hparams.learning_rate

            if init_epochs_net1 < current_epoch and current_epoch < init_epochs_net1+num_iters*init_epochs_net_stack:
                if (current_epoch-init_epochs_net1) % init_epochs_net_stack == 0:
                    load_model_state_dict(model, hparams) # load the best model
                    net_idx = (current_epoch - init_epochs_net1) // init_epochs_net_stack - 1
                    logger.info('At epoch {}, set iter_nets {} to eval'.format(current_epoch, net_idx))
                    for param in model.iter_nets[net_idx].parameters():
                        param.requires_grad = False
                    update_flag = True
                    learning_rate = hparams.learning_rate

            if current_epoch == init_epochs_net1+num_iters*init_epochs_net_stack:

                #raise # no joint training on Feb. 16, 2024

                load_model_state_dict(model, hparams) # load the best model
                logger.info('At epoch {}, set all to train'.format(current_epoch))
                for param in model.parameters():
                    param.requires_grad = True
                update_flag = True
                #learning_rate = hparams.learning_rate / (hparams.lr_scale_factor*1.)
                learning_rate = hparams.learning_rate / 4.
                logger.info('At epoch {}, set learning_rate to {}'.format(current_epoch, learning_rate))

            if update_flag:
                logger.info('At epoch {}, update optimizer and lr_scheduler'.format(current_epoch))
                optimizer = optim.Adam(
                        filter(
                            lambda p: p.requires_grad, model.parameters(),
                            ),
                        lr=learning_rate)

                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, patience=hparams.patience,
                        factor=hparams.decay_factor, verbose=True)

                #trainer.optimizers = [optimizer]
                ##trainer.lr_schedulers = [new_schedulers] # not working
                #trainer.lr_schedulers = trainer.configure_schedulers(
                #                    [scheduler],
                #                    monitor='val_loss',
                #                    is_manual_optimization=False,
                #                )

                # https://github.com/Lightning-AI/pytorch-lightning/blob/3bee81960a6f8979c8e1b5e747a17124feee652d/src/pytorch_lightning/trainer/trainer.py#L1754
                trainer.strategy.optimizers = [optimizer]

                #tmp = trainer.strategy.lr_scheduler_configs
                #assert len(tmp) == 1
                #logger.info(tmp[0])

                """
                # works for pytorch-lightning==1.9.4 
                # https://github.com/Lightning-AI/pytorch-lightning/blob/3bee81960a6f8979c8e1b5e747a17124feee652d/src/pytorch_lightning/core/optimizer.py#L176
                from pytorch_lightning.core.optimizer import _configure_schedulers_automatic_opt, _set_scheduler_opt_idx, _validate_scheduler_api
                trainer.strategy.lr_scheduler_configs = _configure_schedulers_automatic_opt([scheduler], monitor='val_loss')
                _set_scheduler_opt_idx(trainer.strategy.optimizers, trainer.strategy.lr_scheduler_configs)
                _validate_scheduler_api(trainer.strategy.lr_scheduler_configs, pl_module)
                """

                # https://lightning.ai/docs/pytorch/2.0.9/_modules/lightning/pytorch/core/optimizer.html
                from pytorch_lightning.core.optimizer import _configure_schedulers_automatic_opt, _validate_scheduler_api, _validate_multiple_optimizers_support, _validate_optimizers_attached
                trainer.strategy.lr_scheduler_configs = _configure_schedulers_automatic_opt([scheduler], monitor='val_loss')
                _validate_multiple_optimizers_support(trainer.strategy.optimizers, pl_module)
                _validate_optimizers_attached(trainer.strategy.optimizers, trainer.strategy.lr_scheduler_configs)
                _validate_scheduler_api(trainer.strategy.lr_scheduler_configs, pl_module)

                #tmp = trainer.strategy.lr_scheduler_configs
                #assert len(tmp) == 1
                #logger.info(tmp[0])
