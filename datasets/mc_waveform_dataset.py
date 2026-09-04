import os
from collections import OrderedDict
import numpy as np
import soundfile as sf
import torch.utils.data as data
import pickle
from loguru import logger
import itertools
import time


class MCWaveformDataset(data.Dataset):
    def __init__(self, path_list,
            sr=16000, chunk_size=None,
            rms_norm=0,
            n_imics=1, n_omics=1,
            max_num_mics=1, max_num_omics=1,
            training=True,
            partition='tr',
            n_srcs=1,
            ):
        self.path_list = path_list
        self.sr = sr
        if chunk_size is not None:
            self.chunk_size = int(chunk_size * sr)
        else:
            self.chunk_size = None
        self.training = training 

        self.rms_norm = rms_norm
        self.max_num_mics = max_num_mics
        self.max_num_omics = max_num_omics
        #assert 1 <= n_imics and n_imics <= self.max_num_mics
        assert 1 <= n_omics and n_omics <= self.max_num_omics
        self.n_imics = n_imics
        self.n_omics = n_omics
        assert self.n_omics == 1
        self.omics_idx = [0] # also assuming ref_mic is 0

        self.partition = partition

        assert self.n_imics in [1,8,]
        self.n_srcs = n_srcs

    def __getitem__(self, index):
        raise

    def _read_audio(self, index):
        path_dict = self.path_list[index]
        utt_id = path_dict['id']
        mix_path = path_dict['mix']
        n_samples = sf.info(mix_path).frames

        if self.chunk_size is None or n_samples <= self.chunk_size:
            start_samp = 0
            end_samp = None
        else:
            start_samp = np.random.randint(0, n_samples - self.chunk_size)
            end_samp = start_samp + self.chunk_size

        y_mix = self._read_wav(mix_path, start_samp, end_samp).T # [MAX_NUM_MICS, -1]
        #assert y_mix.shape[0] == 8

        if self.rms_norm == 2:
            std_ff = np.std(y_mix, axis=-1, keepdims=True) # [n_imics, 1]
            std_ff[std_ff == 0.] = 1.
            y_mix /= std_ff
        else:
            raise

        y_srcs = np.stack([self._read_wav(src_path, start_samp, end_samp).T \
                           for src_path in path_dict['srcs'].values()], axis=0) # [n_srcs, n_imics, -1], or [n_srcs, 1, -1], depending on the dataset

        assert len(self.omics_idx) == 1
        y_srcs = y_srcs[:, self.omics_idx, :] / std_ff[0]

        return y_mix, y_srcs, std_ff, utt_id

    def __len__(self):
        return len(self.path_list)

    def _read_wav(self, path, start_samp=0, end_samp=None):
        if os.path.exists(path[:-4]+'.pickle'):
            with open(path[:-4]+'.pickle','rb') as tmp_f:
                y = pickle.load(tmp_f)
            if len(y.shape) == 1:
                y = y[:,None]
            y = y[start_samp:(end_samp if end_samp is not None else y.shape[0])] # 2D
        elif os.path.exists(path):
            y, sr = sf.read(path, start=start_samp, stop=end_samp, dtype=np.float32, always_2d=True) # 2D
            assert sr == self.sr
        else:
            logger.info(path)
            raise
        return y
