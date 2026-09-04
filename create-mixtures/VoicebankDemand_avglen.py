import soundfile as sf
import librosa
import glob
import numpy as np
import os
from multiprocessing import Pool

voicebank_dir = os.path.expanduser('~') + '/corpus/VoicebankDemand/'
all_dirs = ['clean_testset_wav_16k', 'clean_trainset_28spk_wav_16k', 'noisy_testset_wav_16k', 'noisy_trainset_28spk_wav_16k']
tgt_fs = 16000

for cur_dir in all_dirs:
    in_dir = voicebank_dir + '/' + cur_dir
    filelist = glob.glob(in_dir + '/*.wav') 
    print(in_dir, len(filelist))
    dur = []
    for filepath in filelist:
        dur.append(librosa.get_duration(filename=filepath))
    dur = np.asarray(dur)
    print(np.mean(dur), np.std(dur), np.max(dur), np.min(dur))

