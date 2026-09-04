import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import os.path


class BacktrackingModelCheckpointUpgrade(ModelCheckpoint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.old_lr = None
        self.CHECKPOINT_NAME_LAST = "mylast"
        self.FILE_EXTENSION = ".ckpt"
        self.sanity_check_flag = False

    def _save_last_checkpoint(self, trainer, monitor_candidates):
        if not self.save_last:
            return

        filepath = self._format_checkpoint_name(self.CHECKPOINT_NAME_LAST, monitor_candidates)
        filepath = os.path.join(os.path.dirname(self.dirpath), f"{filepath}{self.FILE_EXTENSION}") # changed here

        # https://lightning.ai/docs/pytorch/1.9.4/_modules/pytorch_lightning/callbacks/model_checkpoint.html#ModelCheckpoint
        # set the last model path before saving because it will be part of the state.
        # https://lightning.ai/docs/pytorch/2.0.0/_modules/lightning/pytorch/callbacks/model_checkpoint.html#ModelCheckpoint
        previous, self.last_model_path = self.last_model_path, filepath
        self._save_checkpoint(trainer, filepath)
        if previous and previous != filepath:
            self._remove_checkpoint(trainer, previous)

    #def get_best_model_path(self):
    #    if self.mode == 'min':
    #        return min(self.best_k_models, key=self.best_k_models.get)
    #    else:
    #        return max(self.best_k_models, key=self.best_k_models.get)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        """
        self._last_global_step_saved = -1
        self.current_score = None
        self.best_k_models = {}
        self.kth_best_model_path = ""
        self.best_model_score = None
        self.best_model_path = ""
        self.last_model_path = ""
        """
        if self.sanity_check_flag == True:
            # https://github.com/Lightning-AI/lightning/blob/fbc8b209f2fbb980a979835b4ff0f0b16e75fae3/pytorch_lightning/callbacks/model_checkpoint.py#L202
            last_ckpt_dir = os.path.dirname(self.dirpath)
            torch.save([self._last_global_step_saved,
                    self.current_score,
                    self.best_k_models,
                    self.kth_best_model_path,
                    self.best_model_score,
                    self.best_model_path,
                    self.last_model_path,
                    trainer.current_epoch,],
                    last_ckpt_dir+'/mylast_best_info.pt')
        self.sanity_check_flag = True
