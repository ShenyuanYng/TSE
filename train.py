import os
import numpy as np
import pytorch_lightning as pl
import loguru
import random
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
import argparse
from utils.config import yaml_to_parser
from utils.mytensorboardlogger import MyTensorBoardLogger
from utils.backtracking_model_checkpoint import BacktrackingModelCheckpointUpgrade
from source_separator import SourceSeparator
import os.path


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

    logger = MyTensorBoardLogger(save_dir=hparams.save_path, name=hparams.save_name)
    orig_save_path = hparams.save_path
    hparams.save_path = os.path.join(hparams.save_path, logger.name, f'version_{logger.version}')

    #assert pl.__version__ == '2.0.9'
    assert pl.__version__ == '2.5.2'
    torch.set_float32_matmul_precision('high')

    ckpt_callback = BacktrackingModelCheckpointUpgrade(
            dirpath=hparams.save_path, filename='_ckpt_epoch_{epoch}', auto_insert_metric_name=False, #_ckpt_epoch_0.ckpt, FILE_EXTENSION = ".ckpt"
            monitor='val_loss',
            save_last=True, save_top_k=1, save_weights_only=False, mode='min',
            every_n_epochs=1, # change here
            verbose=False,
            )

    resume_from_checkpoint = None
    last_ckpt_dir = os.path.dirname(hparams.save_path)
    last_ckpt_path = last_ckpt_dir+'/mylast.ckpt'
    if os.path.exists(last_ckpt_path): # if true, we will resume training by loading the checkpoint
        #
        # check utils/backtracking_model_checkpoint.py for what flags are saved in mylast_best_info.pt
        #
        loguru.logger.info('loading mylast.ckpt from {}'.format(last_ckpt_path))
        resume_from_checkpoint = last_ckpt_path
        info = torch.load(last_ckpt_dir+'/mylast_best_info.pt')
        ckpt_callback._last_global_step_saved = info[0]
        ckpt_callback.current_score = info[1]
        ckpt_callback.best_k_models = info[2]
        ckpt_callback.kth_best_model_path = info[3]
        ckpt_callback.best_model_score = info[4]
        ckpt_callback.best_model_path = info[5]
        ckpt_callback.last_model_path = info[6]
        last_epoch = info[7]

    # only using the following code only when there are non-trainable parameters in the DNN
    #if hparams.distributed_backend == 'ddp':
    #    strategy = 'ddp_find_unused_parameters_true'
    #else:
    #    raise
    strategy = hparams.distributed_backend

    trainer = Trainer(min_epochs=hparams.epochs, max_epochs=hparams.epochs,
                      logger=logger,
                      limit_train_batches=hparams.train_percent_check, limit_val_batches=hparams.val_percent_check, limit_test_batches=hparams.test_percent_check,
                      enable_checkpointing=True,
                      callbacks=[ckpt_callback, pl.callbacks.ModelSummary(max_depth=5),],
                      default_root_dir=hparams.save_path,
                      devices=hparams.gpus,
                      accelerator="gpu", # accelerator=hparams.distributed_backend, change here
                      #strategy=hparams.distributed_backend, # change here
                      strategy=strategy,
                      gradient_clip_val=hparams.gradient_clip_val,
                      enable_progress_bar=False,
                      benchmark=hparams.cudnn_benchmark,
                      deterministic=hparams.cudnn_deterministic, # change here
                      num_sanity_val_steps=10,
                      ) #

    if 'last_epoch' in locals():
        loguru.logger.info('last_epoch={}'.format(last_epoch))
        #set_model_param_require_grad(trainer, model, hparams, last_epoch)
        model = SourceSeparator(hparams, last_epoch=last_epoch)
    else:
        model = SourceSeparator(hparams, last_epoch=None)

    #torch._dynamo.config.verbose=True
    #compiled_model = torch.compile(model)
    #trainer.fit(compiled_model, ckpt_path=resume_from_checkpoint)
    trainer.fit(model, ckpt_path=resume_from_checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-def', type=str, default='default_params.yaml')
    args, other_options = parser.parse_known_args()
    hparam_parser = yaml_to_parser(args.exp_def)
    hyperparams = hparam_parser.parse_args(other_options)
    main(hyperparams)
