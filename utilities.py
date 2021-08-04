import numpy as np
import scipy.signal as sgn
from scipy.fft import fft, fftfreq
from scipy.signal import stft, istft
from scipy.interpolate import interp1d


import torch
from torch.utils.data import Dataset
import os
import h5py

class WaveformDataset(Dataset):
    def __init__(self, X_train, Y_train):
        self.X_train = np.moveaxis(X_train, 1, -1)
        self.Y_train = np.moveaxis(Y_train, 1, -1)

    def __len__(self):
        return self.X_train.shape[0]

    def __getitem__(self, idx):
        X_waveform = self.X_train[idx]
        Y_waveform = self.X_train[idx]
        return X_waveform, Y_waveform

class WaveformDataset_h5(Dataset):
    def __init__(self, annotations_file):
        self.hdf5_file = h5py.File(annotations_file, 'r')

    def __len__(self):
        return self.hdf5_file['X_train'].shape[0]

    def __getitem__(self, idx):
        X_waveform = self.hdf5_file['X_train'][idx]
        Y_waveform = self.hdf5_file['Y_train'][idx]
        return X_waveform, Y_waveform

def downsample_series(time, series, f_downsampe):
    """Down sample the time series given a lower sampling frequency f_downsample,
    time_new, series_downsample, dt_new = downsample_series(time, series, f_downsampe)

    The time series has been lowpass filtered (f_filter=f_downsample/2) first,
    and then downsampled through interpolation.
    """
    dt = time[1] - time[0]
    # lowpass filter
    b, a = sgn.butter(4, f_downsampe / 2 * 2 * dt)
    series_filt = sgn.filtfilt(b, a, series, axis=0)
    # downsample through interpolation
    dt_new = 1 / f_downsampe
    #time_new = np.arange(time[0], time[-1] + dt_new, dt_new)
    time_new = np.arange(time[0], time[-1], dt_new)
    #series_downsample = np.interp(time_new, time, series_filt)
    interp_f = interp1d(time, series_filt, axis=0, bounds_error=False, fill_value=0.)
    series_downsample = interp_f(time_new)

    # plt.figure()
    # plt.plot(time_noise, noise_BH1, '-k')
    # plt.plot(time, series_filt, '-r')
    # plt.plot(time_new, series_downsample, '-b')

    return time_new, series_downsample, dt_new


def waveform_fft(waveform, dt):
    """ return the Fourier spectrum of the waveform
    freq, sp = waveform_fft(waveform, dt)
    """
    sp = fft(waveform)
    freq = fftfreq(waveform.size, d=dt)

    sp_positive = abs(sp[freq > 0])
    freq_positive = freq[freq > 0]
    return freq_positive, sp_positive


def waveform_stft(waveform, dt, twin=60, toverlap=20, win_type='hann', complex=False):
    """ return the spectrogram of the waveform
    freq, time, sxx = waveform_stft(waveform, dt, twin=60, toverlap=20, win_type='hann')
    """
    fs = 1 / dt
    # apply the thresholding method in the STFT to separate the noise and signals
    f, t, sxx = stft(waveform, fs, nperseg=int(twin / dt),
                     noverlap=int(toverlap / dt), window=win_type)

    if complex:
        return f, t, sxx
    else:
        sxx_abs = abs(sxx)
        return f, t, sxx_abs


def waveform_inverse_stft(Sxx, dt, twin=60, toverlap=20, win_type='hann'):
    fs = 1 / dt
    time, waveform = istft(Sxx, fs=fs, nperseg=int(twin / dt), noverlap=int(toverlap / dt), window=win_type)
    return time, waveform
