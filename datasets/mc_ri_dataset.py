import numpy as np
from utils.audio_utils_torch import do_stft, do_istft
from datasets.mc_waveform_dataset import MCWaveformDataset
from datasets.mc_inference_target import InferenceTargetSTFT
import torch
from loguru import logger
import random


def stft_to_ri(stft_):
    # stft_ is [...,n_freqs]
    # return [...,2,n_freqs]
    return np.stack([np.real(stft_), np.imag(stft_)], axis=-2)

class MCRIDataset(MCWaveformDataset):
    def __init__(self,
            path_list,
            n_srcs=2,
            sr=16000, chunk_size=None,
            rms_norm=0,
            n_imics=1, n_omics=1,
            max_num_mics=1, max_num_omics=1,
            training=True,
            window_length=512, hop_length=128, fft_size=None, window_type='sqrt_hann',
            masking_or_mapping=0,
            input_feature='ri',
            partition='tr',
            ):
        super(MCRIDataset, self).__init__(
                path_list,
                sr=sr, chunk_size=chunk_size,
                rms_norm=rms_norm,
                n_imics=n_imics, n_omics=n_omics, 
                max_num_mics=max_num_mics, max_num_omics=max_num_omics,
                training=training,
                partition=partition,
                n_srcs=n_srcs,
                )
        self.window_length = window_length
        self.hop_length = hop_length
        self.fft_size = fft_size
        self.window_type = window_type
        self.masking_or_mapping = masking_or_mapping
        self.input_feature = input_feature

        assert n_imics in [1,8,]

    def __getitem__(self, index):

        y_mix, y_srcs, y_mix_std_, utt_id = self._read_audio(index)
        #
        # [n_imics, -1]
        # [n_srcs, 1, -1]
        #
        # [n_imics, 1], string
        #
        def func(sig):
            sig_stft = self._stft(sig) # [-1, n_frames, n_freqs]
            sig_ri = stft_to_ri(sig_stft.swapaxes(0,1)) # [n_frames, -1, 2, n_freqs]
            return sig_ri[:,None], sig_stft # [n_frames, 1, -1, 2, n_freqs] and [-1, n_frames, n_freqs]

        mix_ri, mix_stft = func(y_mix) # [n_frames, 1, n_imics, 2, n_freqs], [-1, n_frames, n_freqs]

        if 'ri' in self.input_feature:
            feat = [mix_ri]
        else:
            raise
        feat = np.concatenate(feat, axis=-3)

        srcs_stft = self._stft(y_srcs) # [n_srcs, 1, n_frames, n_freqs]
        srcs_ri = stft_to_ri(srcs_stft) # [n_srcs, 1, n_frames, 2, n_freqs]
        srcs_ri = srcs_ri.transpose([2,0,1,3,4]) # [n_frames, n_srcs, 1, 2, n_freqs]

        if self.training:
            y_mix = y_mix.swapaxes(0,1) # [-1, n_imics]
            y_srcs = y_srcs.swapaxes(0,2) # [-1, 1, n_srcs] 
            srcs = {
                'mix_ri': mix_ri,
                'srcs_ri': srcs_ri,
                #
                'mix': y_mix,
                'srcs': y_srcs,
                #
                'window_size': self.window_length,
                'hop_size': self.hop_length,
                'fft_size': self.fft_size,
                'samplerate': self.sr,
                'window_type': self.window_type,
                #
                'utt_id': utt_id,
            }
            return feat, srcs
        else:
            mix_path = self.path_list[index]['mix']
            return feat, InferenceTargetSTFT(
                    utt_id,
                    y_mix, y_srcs,
                    mix_stft, srcs_stft,
                    mix_ri, srcs_ri,
                    mix_path=mix_path,
                    mix_std_=y_mix_std_,
                    samplerate=self.sr,
                    )

    def _stft(self, samples):
        if type(samples) is not torch.Tensor:
            samples = torch.from_numpy(samples)
        stft_ri = do_stft(samples, self.window_length, self.hop_length, self.window_type, self.fft_size).numpy()
        return stft_ri[...,0]+1j*stft_ri[...,1]

    def _istft(self, yout):
        """
        yout.shape is numpy [...,T,F] or tensor [...,T,F,2]
        return numpy [...,-1]
        """
        if type(yout) is np.ndarray:
            yout = torch.from_numpy(np.stack([np.real(yout),np.imag(yout)],axis=-1))
        return do_istft(yout, self.window_length, self.window_length, self.hop_length, self.window_type, self.fft_size).numpy()
