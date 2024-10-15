import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.audio_utils_torch import do_istft, do_stft, scaling_magnitude
from loss_functions.general import perms


def get_loss_func(loss_type):
    if loss_type in ['ratiol1',]:
        loss_func = torch.abs
    elif loss_type in ['ratiol2',]:
        loss_func = torch.square
    else:
        raise
    return loss_func


def riinf_enh_loss(est_ri, target_dict,
        loss_type="l1",
        loss_items=["RIx2"],
        rix2_flag=0, mag_flag=0,
        sisnrse_flag=0,
        masking_or_mapping=1,
        ref_mic=0,
        window_type="sqrt_hann",
        stft_istft_proj_flag=0,
        permutation_invariant=0,
        mc_flag=0,
        rix2_weight=1.,
        mag_sf=1.0,
        **kwargs):

    assert permutation_invariant == 0
    assert sisnrse_flag == 0

    loss_func = get_loss_func(loss_type)

    n_batch, n_frames, n_srcs, n_omics, _, n_freqs = est_ri.shape # [n_batch, n_frames, n_srcs, n_omics, 2, n_freqs]
    #assert n_batch == 1
    assert n_omics == 1
    assert n_srcs in [1,2,] # 2 means we predict both speech and noise
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

    if mag_sf == 1.0:
        est_r, est_i = est_ri[...,0,:], est_ri[...,1,:] # [n_batch, n_frames, n_srcs, 1, n_freqs]        
        srcs_r, srcs_i = srcs_ri[...,0,:], srcs_ri[...,1,:] # [n_batch, n_frames, n_srcs, 1, n_freqs]
        mix_r, mix_i = mix_ri[...,0,0,:], mix_ri[...,0,1,:] # [n_batch, n_frames, 1, n_freqs]
        mix_mag = torch.sqrt(mix_r**2+mix_i**2+1e-5) # [n_batch, n_frames, 1, n_freqs]
    else:
        assert mc_flag == 0 # should not do mixture-constraint loss if scaling magnitude via power compression
        est_r, est_i, _ = scaling_magnitude(
                torch.complex(est_ri[...,0,:], est_ri[...,1,:]),
                mag_sf, eps=1e-5) # [n_batch, n_frames, n_srcs, 1, n_freqs]
        srcs_r, srcs_i, _ = scaling_magnitude(
                torch.complex(srcs_ri[...,0,:], srcs_ri[...,1,:]),
                mag_sf, eps=1e-5) # [n_batch, n_frames, n_srcs, 1, n_freqs]
        _, _, mix_mag = scaling_magnitude(
                torch.complex(mix_ri[...,0,0,:], mix_ri[...,0,1,:]),
                mag_sf, eps=1e-5) # [n_batch, n_frames, 1, n_freqs]
    utt_weight = torch.stack([mix_mag for _ in range(n_srcs)], dim=2) # [n_batch, n_frames, n_srcs, 1, n_freqs]

    loss_enh, loss_names = [], []

    for ss in range(n_srcs):
        if rix2_flag:
            loss_r = unify_tf_loss(est_r[:,:,ss], srcs_r[:,:,ss], weight_=utt_weight[:,:,ss])
            loss_i = unify_tf_loss(est_i[:,:,ss], srcs_i[:,:,ss], weight_=utt_weight[:,:,ss])
            loss_enh.append(loss_r * rix2_weight)
            loss_enh.append(loss_i * rix2_weight)
            loss_names.append('sR' if ss == 0 else 'nR')
            loss_names.append('sI' if ss == 0 else 'nI')
        if mag_flag:
            est_mag = torch.sqrt(est_r[:,:,ss]**2+est_i[:,:,ss]**2+1e-5)
            srcs_mag = torch.sqrt(srcs_r[:,:,ss]**2+srcs_i[:,:,ss]**2+1e-5)
            loss_mag = unify_tf_loss(est_mag, srcs_mag, weight_=utt_weight[:,:,ss])
            loss_enh.append(loss_mag)
            loss_names.append('sM' if ss == 0 else 'nM')

    if mc_flag:
        assert n_srcs == 2
        est_r = est_r.sum(dim=2, keepdim=True) # [n_batch, n_frames, 1 (n_srcs), 1, n_freqs]
        est_i = est_i.sum(dim=2, keepdim=True) # [n_batch, n_frames, 1 (n_srcs), 1, n_freqs]
        srcs_r = srcs_r.sum(dim=2, keepdim=True) # [n_batch, n_frames, 1 (n_srcs), 1, n_freqs]
        srcs_i = srcs_i.sum(dim=2, keepdim=True) # [n_batch, n_frames, 1 (n_srcs), 1, n_freqs]
        # mix_mag is [n_batch, n_frames, 1, n_freqs]
        if rix2_flag:
            loss_r = unify_tf_loss(est_r[:,:,0], srcs_r[:,:,0], weight_=mix_mag)
            loss_i = unify_tf_loss(est_i[:,:,0], srcs_i[:,:,0], weight_=mix_mag)
            loss_enh.append(loss_r * rix2_weight)
            loss_enh.append(loss_i * rix2_weight)
            loss_names.append('mR')
            loss_names.append('mI')
        if mag_flag:
            est_mag = torch.sqrt(est_r[:,:,0]**2+est_i[:,:,0]**2+1e-5)
            loss_mag = unify_tf_loss(est_mag, mix_mag, weight_=mix_mag)
            loss_enh.append(loss_mag)
            loss_names.append('mM')
 
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
        sisnrse_flag=0,
        masking_or_mapping=1,
        ref_mic=0,
        window_type="sqrt_hann",
        stft_istft_proj_flag=0,
        permutation_invariant=1,
        mc_flag=0,
        rix2_weight=1.,
        mag_sf=1.0,
        **kwargs):

    assert permutation_invariant == 1
    assert mc_flag == 0 # not supported yet
    assert mag_sf == 1.0 # not supported yet
    loss_func = get_loss_func(loss_type)

    n_batch, n_frames, n_srcs, n_omics, _, n_freqs = est_ri.shape # [n_batch, n_frames, n_srcs, n_omics, 2, n_freqs]
    #assert n_batch == 1
    assert n_omics == 1
    assert n_srcs > 1
    assert ref_mic == 0
    if permutation_invariant:
        all_perms = perms(n_srcs)
    else:
        all_perms = [range(n_srcs)]

    mix_ri = target_dict['mix_ri'][0] # [n_batch, n_frames, 1, n_imics, 2, n_freqs]
    srcs_ri = target_dict['srcs_ri'][0] # [n_batch, n_frames, n_srcs, 1, 2, n_freqs]

    loss_pit, loss_names = [], []

    if rix2_flag or mag_flag:

        def unify_uPIT_loss(est_, srcs_, weight_):
            # est_.shape is [n_batch, n_frames, n_srcs, n_omics, n_freqs]
            # src_.shape is [n_batch, n_frames, n_srcs, n_omics, n_freqs]
            # weight_.shape is [n_batch, n_frames, 1, n_omics, n_freqs]
            ret_ = loss_func(est_[:,:,None] - srcs_[:,:,all_perms]) # [n_batch, n_pad_frames, n_perms, n_srcs, n_omics, n_freqs]
            ret_ = torch.mean(ret_/weight_[:,:,None], dim=(1,4,5)) # [n_batch, n_perms, n_srcs]
            ret_ = ret_.mean(dim=-1)
            return ret_ # [n_batch, n_perms]

        if stft_istft_proj_flag:
            # optionally do an iSTFT-STFT projection
            window_size = target_dict['window_size'][0]
            hop_size = target_dict['hop_size'][0]
            fft_size = target_dict['fft_size'][0]
            y_mix = target_dict['mix'][0] # [n_batch, n_samples, n_imics]
            n_orig_samples = y_mix.shape[1]
            est_wfm = do_istft(
                    est_ri.permute(0,3,2,1,5,4), # [n_batch, n_omics, n_srcs, n_frames, n_freqs, 2]
                    window_size, window_size, hop_size, window_type=window_type, fft_size=fft_size) # [n_batch, n_omics, n_srcs, -1]
            est_wfm = est_wfm[...,:n_orig_samples] # [n_batch, n_omics, n_srcs, -1]
            est_ri = do_stft(est_wfm, window_size, hop_size, window_type=window_type) # [n_batch, n_omics, n_srcs, n_frames, n_freqs, 2]
            est_ri = est_ri.permute(0,3,2,1,5,4) # [n_batch, n_frames, n_srcs, n_omics, 2, n_freqs]

        est_r, est_i = est_ri[...,0,:], est_ri[...,1,:] # [n_batch, n_frames, n_srcs, 1, n_freqs]        
        srcs_r, srcs_i = srcs_ri[...,0,:], srcs_ri[...,1,:] # [n_batch, n_frames, n_srcs, 1, n_freqs]
        mix_r, mix_i = mix_ri[...,0,0,:], mix_ri[...,0,1,:] # [n_batch, n_frames, 1, n_freqs]
        mix_mag = torch.sqrt(mix_r**2+mix_i**2+1e-5) # [n_batch, n_frames, 1, n_freqs]
        utt_weight = mix_mag[:,:,None] # [n_batch, n_frames, 1 (n_srcs), 1, n_freqs]

        if rix2_flag:
            loss_r = unify_uPIT_loss(est_r, srcs_r, weight_=utt_weight)
            loss_i = unify_uPIT_loss(est_i, srcs_i, weight_=utt_weight)
            loss_pit.append(loss_r * rix2_weight)
            loss_pit.append(loss_i * rix2_weight)
            loss_names.append('R')
            loss_names.append('I')
        if mag_flag:
            est_mag = torch.sqrt(est_r**2+est_i**2+1e-5)
            srcs_mag = torch.sqrt(srcs_r**2+srcs_i**2+1e-5)
            loss_mag = unify_uPIT_loss(est_mag, srcs_mag, weight_=utt_weight)
            loss_pit.append(loss_mag)
            loss_names.append('M')

    elif sisnrse_flag:

        window_size = target_dict['window_size'][0]
        hop_size = target_dict['hop_size'][0]
        fft_size = target_dict['fft_size'][0]
        y_mix = target_dict['mix'][0] # [n_batch, n_samples, n_imics]
        n_orig_samples = y_mix.shape[1]
        est_wfm = do_istft(
                est_ri.permute(0,3,2,1,5,4), # [n_batch, n_omics, n_srcs, n_frames, n_freqs, 2]
                window_size, window_size, hop_size, window_type=window_type, fft_size=fft_size) # [n_batch, n_omics, n_srcs, -1]
        est_wfm = est_wfm[...,:n_orig_samples] # [n_batch, n_omics, n_srcs, -1]

        est_wfm = est_wfm.permute(0,3,2,1) # [n_batch, n_samples, n_srcs, n_omics]
        est_wfm = est_wfm[:,:,None] # [n_batch, n_samples, 1, n_srcs, n_omics]

        srcs_wfm = target_dict['srcs'][0] # [n_batch, n_samples, 1, n_srcs]
        srcs_wfm = srcs_wfm[:,:,0,:,None] # [n_batch, n_samples, n_srcs, 1]
        srcs_wfm = srcs_wfm[:,:,all_perms,:] # [n_batch, n_samples, n_perms, n_srcs, 1]
        loss_sisnr, _ = si_snr_loss_se(est_wfm, srcs_wfm, use_scaling=True) # [n_batch, n_perms, n_srcs]
        loss_sisnr = loss_sisnr.mean(dim=-1) # [n_batch, n_perms]
        loss_pit.append(loss_sisnr)
        loss_names.append('sisnrse')

    else:
        raise

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


def _mean_center(arr, dim=None):
    mn = torch.mean(arr, dim, keepdims=True)
    return arr - mn


def _l2_square(arr, dim=None, keepdims=False):
    return torch.sum(arr ** 2, dim=dim, keepdims=keepdims)


def si_snr_loss_se(estimates, targets, use_scaling=True):
    """
    Differently from SI-SNR, we scale estimate
    See https://zqwang7.github.io/publications/TASLP2023_TF-GridNet.pdf
    estimates.shape is [n_batch, n_pad_samples,       1, n_srcs, n_omics]
    targets.shape is   [n_batch, n_pad_samples, n_perms, n_srcs, n_omics]
    """
    estimates = _mean_center(estimates, dim=(1,-1)) # [n_batch, n_pad_samples, 1, n_srcs, n_omics]
    targets = _mean_center(targets, dim=(1,-1)) # [n_batch, n_pad_samples, n_perms, n_srcs, n_omics]

    if use_scaling:
        est_power = _l2_square(estimates, dim=(1,-1), keepdims=True) # [n_batch, 1, 1, n_srcs, 1]
        dot_ = torch.sum(estimates*targets, dim=(1,-1), keepdims=True) # [n_batch, 1, n_perms, n_srcs, 1]
        scale = dot_ / (est_power+1e-3) # [n_batch, 1, n_perms, n_srcs, 1]
    else:
        n_batch, _, n_perms, n_srcs, _ = targets.shape
        scale = torch.ones([n_batch,1,n_perms,n_srcs,1],dtype=targets.dtype,device=targets.device)

    s_estimates = scale * estimates # [n_batch, n_pad_samples, n_perms, n_srcs, n_omics]
    e_noise = s_estimates - targets # [n_batch, n_pad_samples, n_perms, n_srcs, n_omics]
    sig_power = _l2_square(targets,dim=(1,-1)) # [n_batch, n_perms, n_srcs]
    noi_power = _l2_square(e_noise,dim=(1,-1)) # [n_batch, n_perms, n_srcs]
    si_snr_array = - 10. * torch.log10( (sig_power+1e-3) / (noi_power + 1e-5*(sig_power+1e-3)) + 1e-5)

    return si_snr_array, scale[:,0,:,:,0] # [n_batch, n_perms, n_srcs], [n_batch, 1, n_perms, n_srcs, 1]
