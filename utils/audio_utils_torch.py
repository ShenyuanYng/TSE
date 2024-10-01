import scipy.signal
import numpy as np
import torch
import math
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def sqrt_hann(window_length, dtype=torch.float32, device=None):
    """ Sqrt-Hann window"""
    return torch.sqrt(torch.hann_window(window_length, periodic=True, dtype=dtype, device=device))


def hann_asym(window_length, alpha=0.1, dtype=torch.float32, device=None):
    l1 = window_length*(1.-alpha)
    assert l1 == int(l1)
    l1 = int(l1)
    l2 = int(window_length*alpha)
    assert l2 == int(l2)
    l2 = int(l2)
    first = torch.hann_window(l1*2, periodic=True, dtype=dtype, device=device)[:l1]
    second = torch.hann_window(l2*2, periodic=True, dtype=dtype, device=device)[-l2:]
    return torch.sqrt(torch.cat([first,second]))


def nextpoweroftwo(window_length):
    return int(2**np.ceil(np.log2(window_length)))


def do_stft(signal, window_length, hop_length, window_type='sqrt_hann', fft_size=None):
    """
    do_stft in pytorch
    :param signal: [...,-1]
    :param window_length:
    :param hop_length:
    :param window_type:
    :return: [...,T,F,2]
    """
    signal_shape = signal.shape
    n_samples = signal_shape[-1]
    signal = signal.reshape([-1,n_samples])

    if window_type == 'sqrt_hann':
        window = sqrt_hann(window_length,dtype=signal.dtype,device=signal.device)
    elif window_type == 'rectangular':
        window = torch.ones(window_length,dtype=signal.dtype,device=signal.device)
    else:
        raise
    if fft_size is None:
        #fft_size = nextpoweroftwo(window_length)
        fft_size = window_length
    else:
        assert fft_size >= window_length
        window = F.pad(window,(fft_size-window_length,0))
        window_length = fft_size
        
    ## padding fft_size//2 zeros on each side
    ## further padding a bunch of zeros on the right to make integer number of frames
    ## this is to align with the do_stft in utils/audio_utils.py
    #signal_len = n_samples
    #signal_len += 2*(fft_size//2)
    #pad_size = math.ceil((signal_len-window_length)*1.0/hop_length)*hop_length + window_length - signal_len
    #if pad_size > 0:
    #    signal = torch.cat([signal,
    #            torch.zeros([signal.shape[0],pad_size],dtype=signal.dtype,device=signal.device)], dim=-1)
    #ret = torch.stft(signal, fft_size,
    #        hop_length=hop_length, win_length=window_length, window=window,
    #        center=True, pad_mode='constant', normalized=False, onesided=True) # [-1, F, T, 2]
    ## padding (window_length - hop_length) zeros on each side
    ## this doesn't align with the do_stft in utils/audio_utils.py, which pad win_length//2 zeros on each side
    ## for 3/4 overlap, we have one more frame on each side
    ## further padding a bunch of zeros on the right to make integer number of frames
    signal_len = n_samples
    overlap_len = window_length-hop_length
    signal_len += 2*overlap_len
    pad_size = math.ceil((signal_len-window_length)*1.0/hop_length)*hop_length + window_length - signal_len
    signal = F.pad(signal,(overlap_len,pad_size+overlap_len))
    ret = torch.stft(signal, fft_size,
            hop_length=hop_length, win_length=window_length, window=window,
            center=False, normalized=False, onesided=True, return_complex=True) # [-1, F, T]
    ret = torch.view_as_real(ret) # [-1, F, T, 2]
 
    ret = ret.transpose(-3,-2) # [-1, T, F, 2]
    ret = ret.view(signal_shape[:-1]+ret.shape[-3:]) # [..., T, F, 2]
    return ret


def do_istft(stft_, window_length, output_window_length, hop_length, window_type='sqrt_hann', fft_size=None):
    """
    :param stft_: [...,T,F,2]
    :param window_length:
    :param hop_length:
    :param window_type:
    :param dtype:
    :return: [...,-1]
    """
    assert window_length % hop_length == 0
    assert output_window_length % hop_length == 0

    if window_type == 'sqrt_hann':
        window = sqrt_hann(window_length,dtype=stft_.dtype,device=stft_.device)
    elif window_type == 'rectangular':
        window = torch.ones(window_length,dtype=stft_.dtype,device=stft_.device)
    else:
        raise
    window = synthwin(window[-output_window_length:], hop_length) # we will drop samples
    if fft_size is None:
        #fft_size = nextpoweroftwo(window_length)
        fft_size = window_length
    else:
        assert fft_size >= window_length
        window_length = fft_size

    stft_shape = list(stft_.shape)
    n_frames, n_freqs, last_dim = stft_shape[-3:]
    other_dims = list(stft_shape[:-3])
    stft_ = stft_.reshape([-1,n_frames,n_freqs,last_dim])
    n_batch = stft_.shape[0]

    if last_dim == 2:
        # ifft
        signal = torch.fft.irfft(torch.complex(stft_[...,0],stft_[...,1]), n=None, dim=-1, norm='backward') # [-1,T,window_length]

        # apply synthesis window
        signal = signal[...,-output_window_length:] * window # drop samples, [-1,T,output_window_length]

        # overlap-add 
        multiple = output_window_length // hop_length
        pad_size = math.ceil(n_frames*1.0/multiple)*multiple - n_frames
        if pad_size > 0:
            signal = F.pad(signal, (0,0,0,pad_size))
        signal = signal.view([n_batch,-1,multiple,output_window_length]) # [n_batch,T/multiple,multiple,output_window_length]
        signal = signal.unbind(dim=-2) # list of [n_batch,T/multiple,output_window_length]
        ret = []
        for ii,tmpsig in enumerate(signal):
            tmpsig = tmpsig.reshape([n_batch,-1]) # [n_batch,T/multiple*output_window_length]
            tmpsig = F.pad(tmpsig, (ii*hop_length,(multiple-1-ii)*hop_length))
            ret.append(tmpsig)
        ret = torch.sum(torch.stack(ret,dim=-1),dim=-1) # [n_batch,-1]

        overlap_len = output_window_length - hop_length
        ret = ret[:,overlap_len:-overlap_len].view(other_dims+[-1]) # because we use center=True in torch.stft

    elif last_dim == 4:
        pass
    else:
        raise

    return ret

    
def synthwin(awin, hop_length, swin=None):
    """
    returns normalized synthesis window for perfect reconstruction
    """
    window_length = awin.shape[0]
    padded_window_length = int(np.ceil(window_length*1.0/hop_length))*hop_length

    awin = F.pad(awin, (0,padded_window_length-window_length))
    multiple = padded_window_length // hop_length
    if swin is None:
        swin = awin
    twin = awin * swin
    twin = torch.sum(twin.view([multiple,hop_length]),dim=0)
    twin = torch.cat([twin]*multiple)
    swin = swin / twin
    #swin = swin[:window_length]
    return swin


def complex_multiply(t1, t2):
    a, b = t1
    c, d = t2
    # (a+b*i)*(c+d*i) = ac-bd + (ad+bc)*i
    return a*c-b*d, a*d+b*c


def complex_divide(t1, t2, eps=1e-8):
    a, b = t1
    c, d = t2
    # (a+b*i)/(c+d*i) = (a+b*i)*(c-d*i)/(c*c+d*d)
    g = c**2+d**2
    g = g + (g.detach()==0.0).to(g.dtype)*eps
    e, f = complex_multiply((a,b),(c/g,-d/g))
    return e, f


if __name__ == "__main__":

    """
    from audio_utils import do_stft as do_stft_scipy
    from audio_utils import sqrt_hann as sqrt_hann_scipy

    window_length, hop_length = 512, 128

    # test analysis and synthesis window
    window_scipy = sqrt_hann_scipy(window_length).astype(np.float32)
    window_torch = sqrt_hann(window_length).numpy().astype(np.float32)
    print(np.sum(np.abs(window_scipy-window_torch))/np.sum(np.abs(window_scipy)))

    #np.random.seed(1)
    signal = np.random.randn(window_length+300*hop_length+2).astype(np.float32)
    #signal = np.random.randn(window_length+30*hop_length).astype(np.float32)
    signal = np.stack([signal,signal],axis=0)
    

    # test stft and istft
    signal_ri = do_stft(torch.tensor(signal), window_length, hop_length, window_type='sqrt_hann', fft_size=None)
    signal_ri = signal_ri.numpy() # [..., T, F, 2]
    signal_cs = signal_ri[...,0] + 1j*signal_ri[...,1] # [..., T, F]
    print(signal_cs.shape)

    signal_cs_scipy = do_stft_scipy(signal, 16000, window_length, hop_length, window_type='sqrt_hann', fft_size=None)
    print(signal_cs_scipy.shape)

    print(np.sum(np.abs(signal_cs[...,1:-1,:] - signal_cs_scipy))/np.sum(np.abs(signal_cs)))

    from audio_utils import do_istft as do_istft_scipy
    resyn_signal_cs_scipy = do_istft_scipy(signal_cs_scipy, 16000, window_length, hop_length, window_type='sqrt_hann', fft_size=None)
    print(signal.shape)
    print(resyn_signal_cs_scipy.shape)
    resyn_signal_cs_scipy = resyn_signal_cs_scipy[...,:signal.shape[-1]]

    resyn_signal_cs = do_istft(torch.tensor(signal_ri), window_length, hop_length, window_type='sqrt_hann', fft_size=None)
    print(resyn_signal_cs.shape)
    resyn_signal_cs = resyn_signal_cs[...,:signal.shape[-1]].numpy()
    print(resyn_signal_cs.shape)
    
    print(np.sum(np.abs(resyn_signal_cs-signal))/np.sum(np.abs(signal)))
    print(np.sum(np.abs(resyn_signal_cs_scipy-signal))/np.sum(np.abs(signal)))
    """


    """
    # test conv
    filter = np.random.randn(300).astype(np.float32)
    filter = np.stack([filter,filter],axis=0)
    def conv(sig, filter, mode='full'):
        assert mode == 'full'
        sig_len = sig.shape[1]
        return np.stack([scipy.signal.convolve(sig[cc,:],filter[cc,:],mode=mode)[:sig_len] for cc in range(filter.shape[0])],axis=0)

    filtered_signal = conv(signal, filter)
    print(filtered_signal.shape)
    filtered_signal_new = conv_in_time_using_fft(torch.tensor(signal), torch.tensor(filter), torch.tensor(filter)).numpy()
    print(filtered_signal_new.shape)
    print(np.sum(np.abs(filtered_signal-filtered_signal_new))/np.sum(np.abs(filtered_signal)))
    """

    window_length, hop_length = 512, 128
    #window_length, hop_length = 400, 160
    output_window_length = window_length

    np.random.seed(1)
    signal = np.random.randn(window_length+2000*hop_length+2).astype(np.float32)
    #signal = np.random.randn(window_length+30*hop_length).astype(np.float32)
    signal = np.stack([signal,signal],axis=0)
    
    # test stft and istft
    signal_ri = do_stft(torch.tensor(signal), window_length, hop_length, window_type='sqrt_hann', fft_size=None)
    #signal_ri = do_stft(torch.tensor(signal), window_length, hop_length, window_type='sqrt_hann', fft_size=480)
    #signal_ri = do_stft(torch.tensor(signal), window_length, hop_length, window_type='sqrt_hann', fft_size=512)

    print(signal_ri.shape)

    resyn_signal = do_istft(torch.tensor(signal_ri), window_length, output_window_length, hop_length, window_type='sqrt_hann', fft_size=None)
    #resyn_signal = do_istft(torch.tensor(signal_ri), window_length, hop_length, window_type='sqrt_hann', fft_size=480)
    #resyn_signal = do_istft(torch.tensor(signal_ri), window_length, hop_length, window_type='sqrt_hann', fft_size=512)
    print(resyn_signal.shape)
    resyn_signal = resyn_signal[...,:signal.shape[-1]].numpy()
    print(resyn_signal.shape)
    
    print(np.sum(np.abs(resyn_signal-signal))/np.sum(np.abs(signal)))
