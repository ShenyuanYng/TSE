import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.audio_utils_torch import do_istft, do_stft
from loss_functions.general import perms


def riinf_enh_loss(est_ri, target_dict,
        loss_type="l1",
        loss_items=["RIx2"],
        rix2_flag=0, mag_flag=0,
        masking_or_mapping=1,
        ref_mic=0,
        window_type="sqrt_hann",
        stft_istft_proj_flag=0,
        permutation_invariant=0,
        **kwargs):

    assert permutation_invariant == 0

    if loss_type in ['ratiol1',]:
        loss_func = torch.abs
    else:
        raise

    n_batch, n_frames, n_srcs, n_omics, _, n_freqs = est_ri.shape # [n_batch, n_frames, n_srcs, n_omics, 2, n_freqs]
    #assert n_batch == 1 and n_omics == 1
    assert n_omics == 1
    assert n_srcs in [1,]
    assert ref_mic == 0

    mix_ri = target_dict['mix_ri'][0] # [n_batch, n_frames, 1, n_imics, 2, n_freqs]
    srcs_ri = target_dict['srcs_ri'][0] # [n_batch, n_frames, n_srcs, 1, 2, n_freqs]

    if stft_istft_proj_flag:
        # optionally do an iSTFT-STFT projection
        window_size = target_dict['window_size'][0]
        hop_size = target_dict['hop_size'][0]
        fft_size = target_dict['fft_size'][0]
        y_mix = target_dict['mix'][0] # [n_batch, n_samples, n_imics]
        n_orig_samples = y_mix.shape[1]
        #y_srcs = target_dict['srcs'][0] # [n_batch, n_samples, 1, n_srcs]
        est_wfm = do_istft(
                est_ri.permute(0,3,2,1,5,4), # [n_batch, n_omics, n_srcs, n_frames, n_freqs, 2]
                window_size, window_size, hop_size, window_type=window_type, fft_size=fft_size) # [n_batch, n_omics, n_srcs, -1]
        est_wfm = est_wfm[...,:n_orig_samples] # [n_batch, n_omics, n_srcs, -1]
        est_ri = do_stft(est_wfm, window_size, hop_size, window_type=window_type) # [n_batch, n_omics, n_srcs, n_frames, n_freqs, 2]
        est_ri = est_ri.permute(0,3,2,1,5,4) # [n_batch, n_frames, n_srcs, n_omics, 2, n_freqs]

    def unify_tf_loss(est_, src_, weight_=None):
        # est_.shape is [n_batch, n_frames, n_omics, n_freqs] or [n_batch, n_frames, n_freqs]
        # src_.shape is [n_batch, n_frames, n_omics, n_freqs] or [n_batch, n_frames, n_freqs]
        err_ = loss_func(est_ - src_)
        err_ = torch.sum(err_.reshape([n_batch,-1]),dim=-1) # [n_batch,]
        weight_ = loss_func(weight_)
        weight_ = torch.sum(weight_.reshape([n_batch,-1]),dim=-1) # [n_batch,]
        return (err_/weight_).mean() # scalar

    est_r, est_i = est_ri[...,0,:], est_ri[...,1,:] # [n_batch, n_frames, n_srcs, 1, n_freqs]        
    srcs_r, srcs_i = srcs_ri[...,0,:], srcs_ri[...,1,:] # [n_batch, n_frames, n_srcs, 1, n_freqs]

    mix_r, mix_i = mix_ri[...,0,0,:], mix_ri[...,0,1,:] # [n_batch, n_frames, 1, n_freqs]
    mix_mag = torch.sqrt(mix_r**2+mix_i**2+1e-5) # [n_batch, n_frames, 1, n_freqs]
    utt_weight = torch.stack([mix_mag for _ in range(n_srcs)], dim=2) # [n_batch, n_frames, n_srcs, 1, n_freqs]

    assert n_srcs == 1
    loss_enh, loss_names = [], []

    for ss in range(n_srcs):
        if rix2_flag:
            loss_r = unify_tf_loss(est_r[:,:,ss], srcs_r[:,:,ss], weight_=utt_weight[:,:,ss])
            loss_i = unify_tf_loss(est_i[:,:,ss], srcs_i[:,:,ss], weight_=utt_weight[:,:,ss])
            loss_enh.append(loss_r)
            loss_enh.append(loss_i)
            loss_names.append('sR' if ss == 0 else 'nR')
            loss_names.append('sI' if ss == 0 else 'nI')
        if mag_flag:
            est_mag = torch.sqrt(est_r[:,:,ss]**2+est_i[:,:,ss]**2+1e-5)
            srcs_mag = torch.sqrt(srcs_r[:,:,ss]**2+srcs_i[:,:,ss]**2+1e-5)
            loss_mag = unify_tf_loss(est_mag, srcs_mag, weight_=utt_weight[:,:,ss])
            loss_enh.append(loss_mag)
            loss_names.append('sM' if ss == 0 else 'nM')

    loss_enh = torch.stack(loss_enh)
    loss_L = loss_enh.detach()
    loss_enh = torch.sum(loss_enh)
    ret = {'loss': loss_enh,}
    loss_indv = list(loss_L.unbind(dim=0)) # list
    for aa,bb in zip(loss_names,loss_indv):
        ret[aa] = bb

    return ret


def riinf_pit_loss(est_ri, target_dict,
        loss_type="l1",
        loss_items=["RIx2"],
        rix2_flag=0, mag_flag=0,
        masking_or_mapping=1,
        ref_mic=0,
        window_type="sqrt_hann",
        stft_istft_proj_flag=0,
        permutation_invariant=1,
        **kwargs):

    assert permutation_invariant == 1

    if loss_type in ['ratiol1',]:
        loss_func = torch.abs
    else:
        raise

    n_batch, n_frames, n_srcs, n_omics, _, n_freqs = est_ri.shape # [n_batch, n_frames, n_srcs, n_omics, 2, n_freqs]
    #assert n_batch == 1 and n_omics == 1
    assert n_omics == 1
    assert n_srcs > 1
    assert ref_mic == 0

    mix_ri = target_dict['mix_ri'][0] # [n_batch, n_frames, 1, n_imics, 2, n_freqs]
    srcs_ri = target_dict['srcs_ri'][0] # [n_batch, n_frames, n_srcs, 1, 2, n_freqs]

    if stft_istft_proj_flag:
        # optionally do an iSTFT-STFT projection
        window_size = target_dict['window_size'][0]
        hop_size = target_dict['hop_size'][0]
        fft_size = target_dict['fft_size'][0]
        y_mix = target_dict['mix'][0] # [n_batch, n_samples, n_imics]
        n_orig_samples = y_mix.shape[1]
        #y_srcs = target_dict['srcs'][0] # [n_batch, n_samples, 1, n_srcs]
        est_wfm = do_istft(
                est_ri.permute(0,3,2,1,5,4), # [n_batch, n_omics, n_srcs, n_frames, n_freqs, 2]
                window_size, window_size, hop_size, window_type=window_type, fft_size=fft_size) # [n_batch, n_omics, n_srcs, -1]
        est_wfm = est_wfm[...,:n_orig_samples] # [n_batch, n_omics, n_srcs, -1]
        est_ri = do_stft(est_wfm, window_size, hop_size, window_type=window_type) # [n_batch, n_omics, n_srcs, n_frames, n_freqs, 2]
        est_ri = est_ri.permute(0,3,2,1,5,4) # [n_batch, n_frames, n_srcs, n_omics, 2, n_freqs]

    if permutation_invariant:
        all_perms = perms(n_srcs)
    else:
        all_perms = [range(n_srcs)]

    def unify_uPIT_loss(est_, srcs_, weight_):
        # est_.shape is [n_batch, n_frames, n_srcs, n_omics, n_freqs]
        # src_.shape is [n_batch, n_frames, n_srcs, n_omics, n_freqs]
        # weight_.shape is [n_batch, n_frames, 1, n_omics, n_freqs]
        ret_ = loss_func(est_[:,:,None] - srcs_[:,:,all_perms]) # [n_batch, n_pad_frames, n_perms, n_srcs, n_omics, n_freqs]
        ret_ = torch.mean(ret_/weight_[:,:,None], dim=(1,4,5)) # [n_batch, n_perms, n_srcs]
        ret_ = ret_.mean(dim=-1)
        return ret_ # [n_batch, n_perms]

    est_r, est_i = est_ri[...,0,:], est_ri[...,1,:] # [n_batch, n_frames, n_srcs, 1, n_freqs]        
    srcs_r, srcs_i = srcs_ri[...,0,:], srcs_ri[...,1,:] # [n_batch, n_frames, n_srcs, 1, n_freqs]

    mix_r, mix_i = mix_ri[...,0,0,:], mix_ri[...,0,1,:] # [n_batch, n_frames, 1, n_freqs]
    mix_mag = torch.sqrt(mix_r**2+mix_i**2+1e-5) # [n_batch, n_frames, 1, n_freqs]
    utt_weight = mix_mag[:,:,None] # [n_batch, n_frames, 1 (n_srcs), 1, n_freqs]

    loss_pit, loss_names = [], []

    if rix2_flag:
        loss_r = unify_uPIT_loss(est_r, srcs_r, weight_=utt_weight)
        loss_i = unify_uPIT_loss(est_i, srcs_i, weight_=utt_weight)
        loss_pit.append(loss_r)
        loss_pit.append(loss_i)
        loss_names.append('R')
        loss_names.append('I')
    if mag_flag:
        est_mag = torch.sqrt(est_r**2+est_i**2+1e-5)
        srcs_mag = torch.sqrt(srcs_r**2+srcs_i**2+1e-5)
        loss_mag = unify_uPIT_loss(est_mag, srcs_mag, weight_=utt_weight)
        loss_pit.append(loss_mag)
        loss_names.append('M')

    loss_pit = torch.stack(loss_pit, dim=-1) # [n_batch, n_perms, #loss]
    loss_B_P_L = loss_pit.detach()
    loss_pit = torch.mean(loss_pit, dim=-1) # [n_batch, n_perms]
    best_perm_idx = loss_pit.detach().argmin(dim=-1) # [n_batch,]
    loss_pit, _ = loss_pit.min(dim=-1) # [n_batch,]
    loss_pit = loss_pit.mean()
    ret = {'loss': loss_pit}

    loss_indv = torch.stack([loss_B_P_L[bb,pp] for bb, pp in enumerate(best_perm_idx)], dim=0) # [n_batch, #loss]
    loss_indv = loss_indv.mean(dim=0) # [#loss]
    loss_indv = list(loss_indv.unbind(dim=0)) # list
    for aa,bb in zip(loss_names,loss_indv):
        ret[aa] = bb

    return ret
