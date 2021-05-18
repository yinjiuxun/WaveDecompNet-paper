import numpy as np
import scipy.signal as sgn
from scipy.fft import fft, fftfreq


def downsample_series(time, series, f_downsampe):
    """Down sample the time series given a lower sampling frequency f_downsample,
    time_new, series_downsample, dt_new = downsample_series(time, series, f_downsampe)

    The time series has been lowpass filtered (f_filter=f_downsample/2) first,
    and then downsampled through interpolation.
    """
    dt = time[1] - time[0]
    # lowpass filter
    b, a = sgn.butter(4, f_downsampe / 2 * 2 * dt)
    series_filt = sgn.filtfilt(b, a, series)
    # downsample through interpolation
    dt_new = 1 / f_downsampe
    time_new = np.arange(time[0], time[-1] + dt_new, dt_new)
    series_downsample = np.interp(time_new, time, series_filt)

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
