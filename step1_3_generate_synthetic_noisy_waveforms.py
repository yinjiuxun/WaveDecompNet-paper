#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the synthetic waveform data with randomized noise that is extracted from real waveforms.
Created on Mon Apr 19 12:38:02 2021

@author: Yin9xun
"""
# %%
import os
import numpy as np
import scipy
import sys
import glob
from matplotlib import pyplot as plt

# import the HDF5 format module
import h5py

# functions for STFT (spectrogram)
from scipy import signal as sgn

from utilities import downsample_series

# load the module to generate random waveforms
sys.path.append('./pyrocko_synthetics')
# 3-component synthetic seismograms from Pyrocko
from synthetic_waveforms import pyrocko_synthesis, random_pyrocko_synthetics
# random Ricker wavelet
from synthetic_waveforms import random_ricker


def randomization_noise(noise, rng=np.random.default_rng(None)):
    """function to produce randomized noise by shiftint the phase
    randomization_noise(noise, rng=np.random.default_rng(None))
    return randomized noise
    """

    s = scipy.fft.fft(noise)
    phase_angle_shift = (rng.random(len(s)) - 0.5) * 2 * np.pi
    # make sure the inverse transform is real
    phase_angle_shift[0] = 0
    phase_angle_shift[int(len(s) / 2 + 1):(len(s) + 1)] = -1 * \
                                                          np.flip(phase_angle_shift[1:int(len(s) / 2 + 1)])

    phase_shift = np.exp(np.ones(s.shape) * phase_angle_shift * 1j)

    # Here apply the phase shift in the entire domain
    # s_shifted = np.abs(s) * phase_shift

    # Instead, try only shift frequency below 10Hz
    freq = scipy.fft.fftfreq(len(s), dt)
    ii_freq = abs(freq) <= 10
    s_shifted = s.copy()
    s_shifted[ii_freq] = np.abs(s[ii_freq]) * phase_shift[ii_freq]

    noise_random = np.real(scipy.fft.ifft(s_shifted))
    return noise_random


def produce_randomized_noise(noise, num_of_random,
                             rng=np.random.default_rng(None)):
    """function to produce num_of_random randomized noise
    produce_randomized_noise(noise, num_of_random,
                             rng=np.random.default_rng(None))
    return np array with shape of num_of_random x npt_noise
    """

    noise_array = []
    # randomly produce num_of_random pieces of random noise
    for i in range(num_of_random):
        noise_random = randomization_noise(noise, rng=rng)
        noise_array.append(noise_random)

    noise_array = np.array(noise_array)
    return noise_array


def plot_and_compare_randomized_noise(noise):
    plt.figure()
    plt.plot(time_noise, noise, '-r')
    noise_random = randomization_noise(noise)
    plt.plot(time_noise, noise_random, '-b')

    # % Compare the STFT distribution of noise and randomized noise, looks better.
    # the randomization of noise in the Fourier domain seems to be able to wipe out
    # the signal residual while keep the general STFT structure of noise

    # TODO: refactor this part to generate randomized noise
    twin = 100
    toverlap = 50
    win_type = 'hann'

    # compare the STFT before and after randomization
    f, t, Sxx = sgn.stft(noise, fs, nperseg=int(twin / dt),
                         noverlap=int(toverlap / dt), window=win_type)
    vmax = np.amax(abs(Sxx))

    plt.figure()
    plt.subplot(221)
    plt.pcolormesh(t, f, np.abs(Sxx), shading='auto', vmax=vmax / 1.2)
    plt.title('STFT of origianl noise')
    plt.ylim(0, 20)

    plt.subplot(223)
    plt.pcolormesh(t, f, np.abs(Sxx), shading='auto', vmax=vmax / 1.2)
    plt.title('STFT of origianl noise')
    plt.ylim(0, 1)
    plt.show()

    # apply the thresholding method in the STFT to separate the noise and signals
    f, t, Sxx = sgn.stft(noise_random, fs, nperseg=int(twin / dt),
                         noverlap=int(toverlap / dt), window=win_type)

    plt.subplot(222)
    plt.pcolormesh(t, f, np.abs(Sxx), shading='auto', vmax=vmax / 1.2)
    plt.title('STFT of randomized noise')
    plt.ylim(0, 20)
    plt.show()

    plt.subplot(224)
    plt.pcolormesh(t, f, np.abs(Sxx), shading='auto', vmax=vmax / 1.2)
    plt.title('STFT of randomized noise')
    plt.ylim(0, 1)
    plt.show()




# %%
# initialize X_train and Y_train
X_train = []
Y_train = []

# load the data
hdf5_files = np.array(glob.glob('./waveforms/noise/*.hdf5'))
#hdf5_files = hdf5_files[np.random.randint(0, len(hdf5_files), 10, dtype="int")]

N_random = 50  # randomized noise for each noise data
# store the randomly produced source information
lat, lon, depth, strike, dip, rake, duration = [], [], [], [], [], [], []

for i_hdf5, hdf5_file in enumerate(hdf5_files):

    with h5py.File(hdf5_file, 'r') as f:
        time_noise = f['noise_time'][:]
        dt = time_noise[1] - time_noise[0]
        noise_BH1 = f['noise']['BH1'][:]
        noise_BH2 = f['noise']['BH2'][:]
        noise_BHZ = f['noise']['BHZ'][:]

    print(hdf5_file + f"--- {(i_hdf5 + 1)/len(hdf5_files) * 100:.2f}% --")

    # %% downsample the noise piece by specific the frequency
    f_downsample = 1.0  # Hz
    _, noise_BH1, _ = downsample_series(time=time_noise, series=noise_BH1, f_downsampe=f_downsample)
    _, noise_BH2, _ = downsample_series(time=time_noise, series=noise_BH2, f_downsampe=f_downsample)
    time_noise, noise_BHZ, dt = downsample_series(time=time_noise, series=noise_BHZ, f_downsampe=f_downsample)

    # sampling rate
    fs = 1 / dt

    # produce N_random pieces of randomized noise
    rng1 = np.random.default_rng(seed=1)
    rng2 = np.random.default_rng(seed=2)
    rngz = np.random.default_rng(seed=3)
    noise_BH1_random = produce_randomized_noise(noise_BH1, N_random, rng=rng1)
    noise_BH2_random = produce_randomized_noise(noise_BH2, N_random, rng=rng2)
    noise_BHZ_random = produce_randomized_noise(noise_BHZ, N_random, rng=rngz)

    # % % #TODO: prepare the synthetic seismic signals
    # duration and number of points in the synthetic waveforms
    synthetic_length = 600
    npt_synthetic = int(synthetic_length / dt)
    N_segments = int(noise_BHZ_random.shape[1] / npt_synthetic)
    syn_time = np.arange(0, npt_synthetic) * dt

    # loop over each piece of randomized noise
    for i_noise in range(N_random):
        # current noise piece
        noise_BH1_now = noise_BH1_random[i_noise, :]
        noise_BH2_now = noise_BH2_random[i_noise, :]
        noise_BHZ_now = noise_BHZ_random[i_noise, :]

        # combine the noise into one np array
        noise_array = np.array([noise_BH1_now, noise_BH2_now, noise_BHZ_now])

        # get the std of current noises (from all three components)
        noise_std = np.std(noise_array)

        # loop over each segment
        for i_seg in range(N_segments):
            # produce the randomized synthetic signal
            # syn_seismic = random_ricker(synthetic_length, dt)  # Ricker wavelet

            # produce the randomized synthetic seismograms from pyrocko
            time_synthetics, synthetic_waveforms, src_info, channels = random_pyrocko_synthetics(
                store_superdirs=['./pyrocko_synthetics'], store_id='ak135_2000km_1Hz', max_amplitude=20)

            # save the source parameters
            lat.append(src_info["lat"])
            lon.append(src_info["lon"])
            depth.append(src_info["depth"])
            strike.append(src_info["strike"])
            dip.append(src_info["dip"])
            rake.append(src_info["rake"])
            duration.append(src_info["duration"])

            # interpolate to the same time axis
            f_interp = scipy.interpolate.interp1d(time_synthetics, synthetic_waveforms, kind='linear',
                                                  bounds_error=False, fill_value=0)
            synthetic_waveforms = f_interp(syn_time)

            synthetic_waveforms = noise_std * synthetic_waveforms
            syn_signal = noise_array[:, i_seg * npt_synthetic:(i_seg + 1) * npt_synthetic] + synthetic_waveforms

            # append the randomized results
            X_train.append(syn_signal)
            Y_train.append(synthetic_waveforms)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# make the directory to store the synthetic data
output_dir = './synthetic_noisy_waveforms'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# %% save the randomly produced source parameters
with h5py.File(output_dir + '/source_parameters_pyrocko_ENZ.hdf5', 'w') as f:
    f.create_dataset('lat', data=lat)
    f.create_dataset('lon', data=lon)
    f.create_dataset('depth', data=depth)
    f.create_dataset('strike', data=strike)
    f.create_dataset('dip', data=dip)
    f.create_dataset('rake', data=rake)
    f.create_dataset('duration', data=duration)


# %% save the prepared data
with h5py.File(output_dir + '/training_waveforms_pyrocko_ENZ.hdf5', 'w') as f:
    f.create_dataset('time', data=syn_time)
    f.create_dataset('X_train', data=X_train)
    f.create_dataset('Y_train', data=Y_train)

# Not in use now
# %% Visualization
# visualize the distribution of the source parameters
with h5py.File(output_dir + '/training_waveforms_pyrocko_ENZ.hdf5', "r") as f:
    lat = f["lat"][:]
    lon = f["lon"][:]
    depth = f["depth"][:]
    strike = f["strike"][:]
    dip = f["dip"][:]
    rake = f["rake"][:]
    duration = f["duration"][:]

plt.figure(30, figsize=(14, 10))
source_parameter_list = [depth/1e3, duration, strike, dip, rake]
source_parameter_name = ["depth (km) ", "duration (s)", "strike", "dip", "rake"]
plt.subplot(2, 3, 1)
plt.scatter(lon, lat, s=0.5, c=depth, alpha=0.2)
plt.xlabel('Lon')
plt.ylabel('Lat')
plt.title("Spatial distribution")

for i_src, src_parameter in enumerate(source_parameter_list):
    plt.subplot(2,3,i_src + 2)
    plt.hist(src_parameter, bins=100)
    plt.title(source_parameter_name[i_src])

plt.savefig('./Figures/Pyrocko_source_parameters2.png')
# visualize the synthetic signals
i = np.random.randint(0, X_train.shape[0])
plt.figure(10)
plt.subplot(211)
plt.plot(syn_time, np.squeeze(Y_train[i, :]).T)
plt.subplot(212)
plt.plot(syn_time, np.squeeze(X_train[i, :]).T)
plt.show()

# visualize the randomly produced noise
hdf5_temp = hdf5_files[np.random.randint(0, len(hdf5_files))]
with h5py.File(hdf5_temp, 'r') as f:
    time_noise = f['noise_time'][:]
    dt = time_noise[1] - time_noise[0]
    fs = 1 / dt
    noise_BH1 = f['noise']['BH1'][:]
    noise_BH2 = f['noise']['BH2'][:]
    noise_BHZ = f['noise']['BHZ'][:]

plot_and_compare_randomized_noise(noise_BHZ)


# #%% Some sort of dispersive signal
# time = np.arange(0,4800)*dt+10
# syn_seismic = np.sin(8*np.pi*10**(time/25)/20)/(time+2)**2

# f, t, Sxx = sgn.spectrogram(syn_seismic, fs, nperseg=128, noverlap=64)

# plt.close('all')
# plt.figure()
# plt.subplot(211)
# plt.plot(time, syn_seismic)
# plt.subplot(212)
# plt.pcolormesh(t,f,Sxx, shading='auto')
