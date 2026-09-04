import soundfile as sf
import librosa
import glob
import numpy as np
import os
from multiprocessing import Pool


voicebank_dir = os.path.expanduser('~') + '/corpus/VoicebankDemand/' # orginal sampling rate is 48 kHz
all_dirs = ['clean_testset_wav', 'clean_trainset_28spk_wav', 'noisy_testset_wav', 'noisy_trainset_28spk_wav']
tgt_fs = 16000

for cur_dir in all_dirs:
    in_dir = voicebank_dir + '/' + cur_dir
    out_dir = voicebank_dir + '/' + cur_dir + '_16k'
    filelist = glob.glob(in_dir + '/*.wav') 
    print(in_dir, len(filelist))
    os.system('mkdir -p '+out_dir)

    def downsample(uu):
        filepath = filelist[uu]
        sig, org_fs = sf.read(filepath, dtype=np.float32)
        sig = librosa.resample(sig, orig_sr=org_fs, target_sr=tgt_fs)
        sf.write(file=out_dir+'/'+filepath.split('/')[-1], data=sig.astype(np.float32), samplerate=tgt_fs)

    NUM_PROCESSES = 16 #
    p = Pool(NUM_PROCESSES)
    p.map(downsample, range(len(filelist)))
