#!/usr/bin/env python

import numpy as np
import pickle
import os, sys
import glob
import scipy
import scipy.signal
import random
import soundfile as sf
from multiprocessing import Pool

from reverb_noise_order_at_MERL import REVERB_NOISE_LIST

nseed = 1 
random.seed(nseed)
np.random.seed(nseed)

def conv(sig, filter, mode='full'):
    assert mode == 'full'
    sig_len = sig.shape[0]
    return np.stack([scipy.signal.convolve(sig,filter[:,cc],mode=mode)[:sig_len] for cc in range(filter.shape[1])],axis=-1)

print('loading noises',)
REVERB_root = os.path.expanduser('~') + '/corpus/reverb_challenge/REVERB_DATA_OFFICIAL/'
sub_dirs = ['reverb_tools_for_Generate_mcTrainData', 'reverb_tools_for_Generate_SimData']
NOISES = [[], []]
for idx, sub_dir in enumerate(sub_dirs):
    file_paths = REVERB_NOISE_LIST[idx]
    for file_path in file_paths:
        print('.', end = '')
        sig, fs = sf.read(file_path.replace('/scratch/wangzhon/','/home/wangzq_lab/cse30029918/corpus/'))
        assert fs == 16000
        sig = sig.astype(np.float32)
        NOISES[idx].append(sig) # each [-1, 8]
    print('\n')

wsjcam0_root = os.path.expanduser('~') + '/corpus/WSJCAM0/data/' # on pnl18
data_root = os.path.expanduser('~') + '/corpus/MERL-dereverb_signals/fixarr/'
rir_flist = sorted(glob.glob(data_root+'/RIR_16k/RIR_*.pickle'))

rir_S = 4
rir_M = 8
S = 1
M = 8
assert M >= 1 and M <= rir_M
assert S >= 1 and S <= rir_S

set_names = ['tr','cv','tt']
n_mixtures = [7861,742,1088]
n_duplicates = [5,4,3] # 7861*5+742*4+1088*3=45537
#n_mixtures = [7861,742,1088]
#n_duplicates = [1,1,1] # 7861*1+742*1+1088*1=

to_write_dir = data_root + '/wav16k/'

def scale_to_snr(sph, noi, snr):
    scaling_factor = np.sqrt( np.sum(sph**2)/np.sum(noi**2) / (10**(snr/10.)) )
    return scaling_factor

def draw(range):
    assert range[1] > range[0]
    return (range[1]-range[0])*np.random.rand()+range[0]

for dd, set_name in enumerate(set_names):

    nseed = dd*7 
    random.seed(nseed)
    np.random.seed(nseed)

    print(set_name)

    os.system('mkdir -p '+to_write_dir+'/'+set_name+'/mix/')
    os.system('mkdir -p '+to_write_dir+'/'+set_name+'/s1/')
    os.system('mkdir -p '+to_write_dir+'/'+set_name+'/s1_reverb/')
    os.system('mkdir -p '+to_write_dir+'/'+set_name+'/s1_source/')
    os.system('mkdir -p '+to_write_dir+'/'+set_name+'/s1_rir/')
    os.system('mkdir -p '+to_write_dir+'/'+set_name+'/s1_reverb_rir/')
    os.system('mkdir -p '+to_write_dir+'/'+set_name+'/s1_early/')
    os.system('mkdir -p '+to_write_dir+'/'+set_name+'/s1_early_to_listen/')

    if set_name == 'tr':
        dry_flist = sorted(glob.glob(wsjcam0_root+'/primary_microphone/si_tr/*/*.wav'))
        noi_idx = 0
    elif set_name == 'cv':
        dry_flist = sorted(glob.glob(wsjcam0_root+'/primary_microphone/si_dt/*/*.wav'))
        noi_idx = 0
    elif set_name == 'tt':
        dry_flist = sorted(glob.glob(wsjcam0_root+'/*_microphone/si_et_*/*/*.wav'))
        noi_idx = 1
    else:
        raise
    assert len(dry_flist) == n_mixtures[dd]
    dry_flist = dry_flist * n_duplicates[dd] # duplicate

    rir_start = int(np.sum(np.asarray(n_mixtures[:dd])*np.asarray(n_duplicates[:dd])))

    n_mixture = n_mixtures[dd]*n_duplicates[dd]

    def simulate_mixture(uu):

        print("%d/%d"%(uu,n_mixture))

        rir_counter = rir_start + uu 
        nseed = rir_counter*7 
        random.seed(nseed)
        np.random.seed(nseed)

        rir_info = pickle.load(open(rir_flist[rir_counter].strip(),'rb'))
        h_anecho = rir_info['h_anecho'][:M,:S,:].astype(np.float32)
        h_reverb = rir_info['h_reverb'][:M,:S,:].astype(np.float32)
        T60 = rir_info['T60']

        # Found this bug on Jun. 26, 2021, some RIRs are all-zero, don't know why
        # Regenerated the data for dereverb, i.e. fixarr on Jun. 26, 2021
        if np.sum( np.sum(np.abs(h_anecho),axis=-1) == 0. ) > 0:
            print('Something wrong with RIR idx = '+str(rir_counter))
            return
        if np.sum( np.sum(np.abs(h_reverb),axis=-1) == 0. ) > 0:
            print('Something wrong with RIR idx = '+str(rir_counter))
            return

        sig, fs = sf.read(dry_flist[uu])
        assert fs == 16000
        sig = sig.astype(np.float32)
        assert len(sig.shape) == 1

        sig -= np.mean(sig)

        assert S == 1
        mc_ane_sig = conv(sig,h_anecho[:,0].T) # [-1, M]
        mc_rev_sig = conv(sig,h_reverb[:,0].T) # [-1, M]

        # assuming only one source, first micr as reference
        peak_idx = np.argmax(h_reverb[0,0])
        h_early = h_reverb[:,:S,:peak_idx+int(0.05*fs)] # consider 50 ms after the peak
        mc_ear_sig = conv(sig,h_early[:,0].T) # [-1, M]

        file_idx = np.random.randint(len(NOISES[noi_idx]))
        samp_idx = np.random.randint(NOISES[noi_idx][file_idx].shape[0]-len(sig))
        noi = NOISES[noi_idx][file_idx][samp_idx:samp_idx+len(sig)]
        noi = np.hstack([noi[:,(uu%noi.shape[1]):], noi[:,:(uu%noi.shape[1])]]) # shift noise

        snr_range = [5., 25.]
        ane_noi_snr = draw(snr_range)

        scaling_factor = scale_to_snr(mc_ane_sig, noi, ane_noi_snr) 
        noi = noi * scaling_factor
        mc_mix_sig = mc_rev_sig + noi

        #max_ = max(np.max(np.abs(mc_mix_sig)),np.max(np.abs(mc_ane_sig)))
        max_ = max(np.max(np.abs(mc_mix_sig)),np.max(np.abs(mc_ane_sig)),np.max(np.abs(mc_rev_sig)))
        mc_mix_sig = mc_mix_sig / max_ * 0.95
        mc_ane_sig = mc_ane_sig / max_ * 0.95
        mc_rev_sig = mc_rev_sig / max_ * 0.95
        sig = sig / max_ * 0.95
        mc_ear_sig = mc_ear_sig / max_ * 0.95

        """
        print(mc_ear_sig.dtype, sig.dtype, mc_mix_sig.dtype, mc_ane_sig.dtype, mc_rev_sig.dtype, noi.dtype)
        print(mc_ear_sig.shape, sig.shape, mc_mix_sig.shape, mc_ane_sig.shape, mc_rev_sig.shape, noi.shape)
        print(sys.getsizeof(mc_ear_sig), sys.getsizeof(sig), sys.getsizeof(mc_mix_sig))
        print(sys.getsizeof(mc_ane_sig), sys.getsizeof(mc_rev_sig), sys.getsizeof(noi))
        raise
        """

        if uu == 17059:
            # See generate_all_RIRs.py for the bug details
            T60 = 0.5188
        file_name = ('%06d'%uu)+'_'+('%.4f'%T60)+'.wav'
        file_name_pickle = ('%06d'%uu)+'_'+('%.4f'%T60)+'.pickle'
        sf.write(file=to_write_dir+'/'+set_name+'/mix/'+file_name, data=mc_mix_sig.astype(np.float32), samplerate=fs)
        sf.write(file=to_write_dir+'/'+set_name+'/s1/'+file_name, data=mc_ane_sig[:,0].astype(np.float32), samplerate=fs) # only save first channel
        sf.write(file=to_write_dir+'/'+set_name+'/s1_reverb/'+file_name, data=mc_rev_sig[:,0].astype(np.float32), samplerate=fs) # only save first channel
        #sf.write(file=to_write_dir+'/'+set_name+'/s1_source/'+file_name, data=sig, samplerate=fs)
        #sf.write(file=to_write_dir+'/'+set_name+'/s1_rir/'+file_name, data=h_anecho[:,0].T, samplerate=fs)
        #sf.write(file=to_write_dir+'/'+set_name+'/s1_reverb_rir/'+file_name, data=h_reverb[:,0].T, samplerate=fs)
        pickle.dump(sig.astype(np.float32),open(to_write_dir+'/'+set_name+'/s1_source/'+file_name_pickle, 'wb'),-1)
        pickle.dump(h_anecho[:,0].T.astype(np.float32),open(to_write_dir+'/'+set_name+'/s1_rir/'+file_name_pickle, 'wb'),-1) # only save first channel
        pickle.dump(h_reverb[:,0].T.astype(np.float32),open(to_write_dir+'/'+set_name+'/s1_reverb_rir/'+file_name_pickle, 'wb'),-1) # only save first channel
        pickle.dump(mc_ear_sig[:,0].astype(np.float32),open(to_write_dir+'/'+set_name+'/s1_early/'+file_name_pickle, 'wb'),-1) # only save first channel
        if uu < 10 and set_name == 'tr':
            sf.write(file=to_write_dir+'/'+set_name+'/s1_early_to_listen/'+file_name, data=mc_ear_sig[:,0].astype(np.float32), samplerate=fs) # only save first channel
        #print(to_write_dir+'/'+set_name+'/mix/'+file_name)

        """
        print(rir_flist[rir_counter].strip())
        print(np.sum(np.abs(h_anecho[:,0].T),axis=0).tolist())
        print(np.sum(np.abs(h_reverb[:,0].T),axis=0).tolist())
        print(np.sum(np.abs(mc_ane_sig),axis=0).tolist())
        print(np.sum(np.abs(mc_rev_sig),axis=0).tolist())
        print(np.sum(np.abs(mc_mix_sig),axis=0).tolist())
        print(np.sum(np.abs(sig)))
        """

        """
        rirs = pickle.load(open(to_write_dir+'/'+set_name+'/s1_rir/'+file_name_pickle,'rb')).T
        reverb_rirs = pickle.load(open(to_write_dir+'/'+set_name+'/s1_reverb_rir/'+file_name_pickle,'rb')).T
        rirs = rirs[0]
        reverb_rirs = reverb_rirs[0]
        #dry_wfm = sig
        #dry_wfm, fs = sf.read(to_write_dir+'/'+set_name+'/s1_source/'+file_name)
        dry_wfm = pickle.load(open(to_write_dir+'/'+set_name+'/s1_source/'+file_name_pickle,'rb'))
        targ_wfm_2 = scipy.signal.convolve(dry_wfm,rirs,mode='full')[:len(sig)]
        targ_wfm, fs = sf.read(to_write_dir+'/'+set_name+'/s1/'+file_name)
        targ_wfm = targ_wfm[:,0]
        print(np.sum(np.abs(targ_wfm_2-targ_wfm))/np.sum(np.abs(targ_wfm)))

        reverb_wfm_2 = scipy.signal.convolve(dry_wfm,reverb_rirs,mode='full')[:len(sig)]
        reverb_wfm, fs = sf.read(to_write_dir+'/'+set_name+'/s1_reverb/'+file_name)
        reverb_wfm = reverb_wfm[:,0]
        print(np.sum(np.abs(reverb_wfm_2-reverb_wfm))/np.sum(np.abs(reverb_wfm)))
        """

    #NUM_PROCESSES = 16 #zqgpuserver1
    #NUM_PROCESSES = 4 #zqgpuserver1
    NUM_PROCESSES = 16 # sustech_cse_cluster_2
    p = Pool(NUM_PROCESSES)
    p.map(simulate_mixture, range(n_mixture))
    """
    for uu in range(n_mixture):
        simulate_mixture(uu)
    """
    """
    for uu in [17059]:
        simulate_mixture(uu)
    raise
    """
