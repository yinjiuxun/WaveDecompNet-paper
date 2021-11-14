import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import butter, filtfilt, fftconvolve
import h5py

from utilities import mkdir

# %%
working_dir = os.getcwd()
# waveforms
network_station1 = "IU.POHA"  # HV.HSSD "HV.WRM" "IU.POHA" "HV.HAT"
waveform_dir = working_dir + '/continuous_waveforms'
model_dataset_dir = "Model_and_datasets_1D_all"
# model_dataset_dir = "Model_and_datasets_1D_STEAD2"
# model_dataset_dir = "Model_and_datasets_1D_STEAD_plus_POHA"
bottleneck_name = "LSTM"

waveform_output_dir1 = waveform_dir + '/' + model_dataset_dir + '/' + network_station1


with h5py.File(waveform_output_dir1 + '/' + bottleneck_name + '_processed_waveforms.hdf5', 'r') as f:
    waveform_time1 = f['waveform_time'][:]
    waveform_original1 = f['waveform_original'][:]
    waveform_recovered1 = f['waveform_recovered'][:]
    noise_recovered1 = f['noise_recovered'][:]

dt = waveform_time1[1] - waveform_time1[0]


# reshape the original waveform and the recovered noise
waveform_original1 = np.reshape(waveform_original1[:, np.newaxis, :], (-1, 600, 3))
noise_recovered1 = np.reshape(noise_recovered1[:, np.newaxis, :], (-1, 600, 3))
waveform_recovered1 = np.reshape(waveform_recovered1[:, np.newaxis, :], (-1, 600, 3))
#noise_recovered1 = waveform_original1 - waveform_recovered1 - noise_recovered1

# Try low- and high-pass filtering
fc = 1.0
[aa, bb] = butter(2, fc * 2 * dt, 'low')
noise_filtered1 = filtfilt(aa, bb, waveform_original1, axis=1)

[aa, bb] = butter(2, fc * 2 * dt, 'high')
earthquake_filtered1 = filtfilt(aa, bb, waveform_original1, axis=1)
earthquake_filtered1 = waveform_original1 - noise_filtered1 # different way

# Check the spectra of different components
from scipy.fft import fft, fftfreq
from scipy.signal import fftconvolve

output_figures = waveform_output_dir1 + '/waveforms_compare'
mkdir(output_figures)
def get_mean_spectrum(dt, data):
    freq = fftfreq(data.shape[1], dt)
    spectrum = np.mean(abs(fft(data, axis=1)),axis=0)
    i_freq = freq > 0
    return freq[i_freq], spectrum[i_freq, :]

residual_left1 = waveform_original1 - noise_recovered1 - waveform_recovered1

titles = ['E', 'N', 'Z']
segment_time = np.arange(0, 60, 0.1)
step = 100
for i_tr in range(0, waveform_recovered1.shape[0], step):
    print('=' * 12 + str(i_tr) + '=' * 12)
    _, raw_spectrum = get_mean_spectrum(dt, waveform_original1[i_tr:i_tr+step, :, :])
    _, noise_spectrum = get_mean_spectrum(dt, noise_recovered1[i_tr:i_tr+step, :, :])
    _, noise_filtered_spectrum = get_mean_spectrum(dt, noise_filtered1)

    earthquake_waveform = waveform_recovered1[i_tr + int(step / 2), :, :]
    _, earthquake_spectrum = get_mean_spectrum(dt, earthquake_waveform[np.newaxis, :, :])

    filter_earthquake_waveform = earthquake_filtered1[i_tr + int(step / 2), :, :]
    _, earthquake_filtered_spectrum = get_mean_spectrum(dt, filter_earthquake_waveform[np.newaxis, :, :])

    freq, residual_spectrum = get_mean_spectrum(dt, residual_left1[i_tr:i_tr+step, :, :])
    plt.close('all')
    fig, ax = plt.subplots(3, 2, figsize=(12, 8))
    for i_chan in range(3):
        ax[i_chan, 0].plot(segment_time, waveform_original1[i_tr + int(step/2), :, i_chan], label='raw')
        ax[i_chan, 0].plot(segment_time, noise_recovered1[i_tr + int(step/2), :, i_chan], label='noise')
        #ax[i_chan, 0].plot(segment_time, noise_filtered1[i_tr + int(step / 2), :, i_chan], label='filtered')

        ax[i_chan, 0].plot(segment_time, residual_left1[i_tr + int(step/2), :, i_chan], label='residual')

        ax[i_chan, 0].plot(segment_time, waveform_recovered1[i_tr + int(step / 2), :, i_chan], label='earthquake')
        #ax[i_chan, 0].plot(segment_time, earthquake_filtered1[i_tr + int(step / 2), :, i_chan], label='filtered earthquake')

        ax[i_chan, 0].set_xscale('linear')
        ax[i_chan, 0].set_title(titles[i_chan])
        ax[i_chan, 0].legend()
        if i_chan == 2:
            ax[i_chan, 0].set_xlabel('Time (s)')

        ax[i_chan, 1].loglog(1/freq, raw_spectrum[:, i_chan], label='raw', linewidth=2)
        ax[i_chan, 1].loglog(1/freq, noise_spectrum[:, i_chan], label='noise')
        # ax[i_chan, 1].loglog(1 / freq, noise_filtered_spectrum[:, i_chan], label='filtered')

        ax[i_chan, 1].loglog(1/freq, residual_spectrum[:, i_chan], label='residual')

        ax[i_chan, 1].loglog(1 / freq, earthquake_spectrum[:, i_chan], label='earthquake')
        #ax[i_chan, 1].loglog(1 / freq, earthquake_filtered_spectrum[:, i_chan], label='filtered earthquake')

        ax[i_chan, 1].set_ylim(1e2, 1e8)
        ax[i_chan, 1].set_xlim(0.05, 500)
        ax[i_chan, 1].set_title(titles[i_chan])
        ax[i_chan, 1].legend()
        if i_chan == 2:
            ax[i_chan, 1].set_xlabel('Period (s)')
    plt.savefig(output_figures + '/' + str(i_tr) + '_waveform_spectrum.png')



