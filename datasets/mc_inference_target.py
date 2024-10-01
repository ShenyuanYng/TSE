class InferenceTargetSTFT:
    def __init__(self,
            id,
            mix_waveform, src_waveforms,
            mix_stft, srcs_stft,
            mix_ri, srcs_ri,
            mix_path="",
            mix_std_=1.0,
            samplerate=None,
            ):
        self.id = id

        self.mix_waveform = mix_waveform # [n_imics, -1] 
        self.src_waveforms = src_waveforms # [n_srcs, 1, -1]

        self.mix_stft = mix_stft # [-1, n_frames, n_freqs]
        self.srcs_stft = srcs_stft # [n_srcs, 1, n_frames, n_freqs]

        self.mix_ri = mix_ri # [n_frames, 1, n_imics, 2, n_freqs]
        self.srcs_ri = srcs_ri # [n_frames, n_srcs, 1, 2, n_freqs]

        self.mix_path = mix_path
        self.mix_std_ = mix_std_

        self.samplerate = samplerate
