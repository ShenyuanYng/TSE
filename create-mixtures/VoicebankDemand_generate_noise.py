import soundfile as sf
import glob
import numpy as np
import os
from multiprocessing import Pool


voicebank_dir = os.path.expanduser('~') + '/corpus/VoicebankDemandEXP/wav16k/' #
all_dirs = ['tr', 'cv',] # tt is linked to cv

for cur_dir in all_dirs:
    mix_dir = voicebank_dir + '/' + cur_dir + '/mix/'
    sph_dir = voicebank_dir + '/' + cur_dir + '/s1/'
    noi_dir = voicebank_dir + '/' + cur_dir + '/s2/'
    os.system('mkdir -p ' + noi_dir)

    mix_filelist = sorted(glob.glob(mix_dir + '/*.wav'))
    sph_filelist = sorted(glob.glob(sph_dir + '/*.wav'))
    print(mix_dir, len(mix_filelist))
    print(noi_dir)
    assert len(mix_filelist) == len(sph_filelist)

    def substract_mix_by_sph(uu):
        mix_filepath = mix_filelist[uu]
        sph_filepath = sph_filelist[uu]
        mix_sig, org_fs = sf.read(mix_filepath, dtype=np.float32)
        assert org_fs == 16000
        sph_sig, org_fs = sf.read(sph_filepath, dtype=np.float32)
        assert org_fs == 16000
        noi_sig = mix_sig - sph_sig
        sf.write(file=noi_dir+'/'+mix_filepath.split('/')[-1], data=noi_sig.astype(np.float32), samplerate=org_fs)

    NUM_PROCESSES = 16 #
    p = Pool(NUM_PROCESSES)
    p.map(substract_mix_by_sph, range(len(mix_filelist)))
