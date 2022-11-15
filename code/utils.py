import torch
import numpy as np
import torch.nn as nn
import soundfile
import soundfile
from audiolazy import *
from argparse import Namespace
import math

def fir_high_pass(samples, fs, fH, N, outputType):
    # Referece: https://fiiir.com

    fH = fH / fs

    # Compute sinc filter.
    h = np.sinc(2 * fH * (np.arange(N) - (N - 1) / 2.))
    # Apply window.
    h *= np.hamming(N)
    # Normalize to get unity gain.
    h /= np.sum(h)
    # Create a high-pass filter from the low-pass filter through spectral inversion.
    h = -h
    h[int((N - 1) / 2)] += 1
    # Applying the filter to a signal s can be as simple as writing
    s = np.convolve(samples, h).astype(outputType)
    return s

def normalize_audio(min, max, audio):
    return (max - min) * ((audio - torch.min(audio)) / (torch.max(audio) - torch.min(audio))) + min

def get_uniform_distribution(num_bins):
    t = torch.empty(num_bins)
    return nn.init.uniform_(t, a=-0.8, b=0.8)

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    
    return smoothed

def entropy_to_bitrate(entropy, sr, window_size, overlap, bottleneck_dim):
    return (sr / (window_size-overlap)) * entropy * np.prod(bottleneck_dim)

def bitrate_to_entropy(bitrate, sr, window_size, overlap, bottleneck_dim):
    return bitrate * (window_size - overlap) / (sr * np.prod(bottleneck_dim))

def get_entropy(p):
    return -torch.sum(torch.mul(p,torch.log(p)))

def soundfile_info(path):
    info = {}
    sfi = soundfile.info(path)
    info['samplerate'] = sfi.samplerate
    info['samples'] = int(sfi.duration * sfi.samplerate)
    info['duration'] = sfi.duration
    return Namespace(**info)

def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    freqs = np.linspace(
        0, float(rate) / 2, n_fft // 2 + 1,
        endpoint=True
    )

    return np.max(np.where(freqs <= bandwidth)[0]) + 1

def vec2mat(x, framesize, overlap, window=[]): 
    assert len(x.shape)==1
    hopsize=framesize-overlap
    nFr=x.shape[0]//hopsize
    if x.shape[0] >= nFr*hopsize:
        x=np.concatenate((x, np.zeros(nFr*hopsize+framesize-x.shape[0])))
        nFr=nFr+1
    X=np.zeros((framesize, nFr))    
    for i in range(nFr):
        X[:,i]=x[i*hopsize:i*hopsize+framesize].copy()
    if len(window) > 0:
        X[:overlap,:]*=window[:overlap]
        X[-overlap:,:]*=window[-overlap:]
    return X

def mat2vec(X, framesize, overlap, window=[]): 
    hopsize=framesize-overlap
    nFr=X.shape[1]
    x=np.zeros((nFr-1)*hopsize+framesize)
    if len(window) > 0:
        X[:overlap,:]*=window[:overlap]
        X[-overlap:,:]*=window[-overlap:]
    for i in range(nFr):
        x[i*hopsize:i*hopsize+framesize]+=X[:,i].copy()
    return x


def prepare_audio(audio, H=16384, in_chan=1, overlap=32):
    
    audio_len = audio.size()[-1]
    hop_size  = H - int(overlap)
    num_frames = math.ceil(audio_len / hop_size)

    prep_audio = torch.zeros(num_frames, in_chan, H)
    timestamps = torch.zeros(num_frames,2)

    end = 0
    for i in range(num_frames):

        start = (i * hop_size)
        end   = start + H

        timestamps[i,0] = start
        timestamps[i,1] = end

        if audio_len > end:
            prep_audio[i,:,:] = torch.clone(audio[:,:,start:end])
        else:
            last = H - (end - audio_len)
            prep_audio[i,:,:last] = torch.clone(audio[:,:,start:start+last])
            prep_audio[i,:,last:] = 0
    
    return prep_audio, timestamps

def overlap_add(audio, timestamps, H=16384, in_chan=1, overlap=32, device='cpu'):
    
    audio_ = torch.clone(audio).to(device)

    num_frames = audio_.size()[0]
    target_len = num_frames * (H - overlap) + overlap
    y = torch.zeros(in_chan, target_len, device=device)

    hann = torch.hann_window(overlap*2, periodic=True, device=device)

    for i in range(num_frames):

        start = int(timestamps[i,0].item())
        end   = int(timestamps[i,1].item())

        chunk = torch.clone(audio_[i,:,:]).to(device)
        for j in range(in_chan):
            chunk[j,:overlap]  = chunk[j,:overlap] * hann[:overlap]
            chunk[j,-overlap:] = chunk[j,-overlap:] * hann[overlap:]

        y[:,start:end] = y[:,start:end] + chunk

    return y

def LPC_synthesis(LPCcoef, residual):
    analysis_filt = ZFilter(list(LPCcoef), [1])
#     print(analysis_filt)
    synth_filt = 1 / analysis_filt
#     print(synth_filt)
    unstable=True
    for i in range(10000): # check thte synth filter stability for 10000 times
        if not lsf_stable(synth_filt):
            LPCcoef*=.95
            analysis_filt = ZFilter(list(LPCcoef), [1])
            synth_filt = 1 / analysis_filt
        else:
            unstable=False
            break
        
    yh=synth_filt(residual)
    yh=np.array(list(yh))
    return yh, unstable

def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=44100, scaler=None):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    a = fftnoise(f)
    if scaler:
        x, y = scaler
        range = max(a) - min(a)
        a = (a - min(a)) / range
        range2 = y - x
        a = (a * range2) + x

    return a

def get_mean_param(params):
    """Return the parameter used to show reconstructions or generations.
    For example, the mean for Normal, or probs for Bernoulli.
    For Bernoulli, skip first parameter, as that's (scalar) temperature
    """
    return params[1] if params[0].dim() == 0 else params[0]

class Constants(object):
    eta = 1e-6
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88                # largest cuda v s.t. exp(v) < inf
    logfloorc = -104             # smallest cuda v s.t. exp(v) > 0
    invsqrt2pi = 1. / math.sqrt(2 * math.pi)
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count