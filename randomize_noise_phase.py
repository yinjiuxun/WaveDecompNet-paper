#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

# import Ricker
import ricker

# %% Randomize the phase in the Fourier domain


def randomization_noise(noise, rng=np.random.default_rng(None)):
    '''function to produce randomized noise by shiftint the phase
    randomization_noise(noise, rng=np.random.default_rng(None))
    return randomized noise
    '''

    s = scipy.fft.fft(noise)
    phase_angle_shift = (rng.random(len(s))-0.5)*2*np.pi
    # make sure the inverse transform is real
    phase_angle_shift[0] = 0
    phase_angle_shift[int(len(s)/2+1):(len(s)+1)] = -1 * \
        np.flip(phase_angle_shift[1:int(len(s)/2+1)])

    phase_shift = np.exp(np.ones(s.shape)*phase_angle_shift*1j)

    # Here apply the phase shift in the entire domain
    # s_shifted = np.abs(s) * phase_shift

    # Instead, try only shift frequency below 10Hz
    freq = scipy.fft.fftfreq(len(s), dt)
    II_freq = abs(freq) <= 10
    s_shifted = s.copy()
    s_shifted[II_freq] = np.abs(s[II_freq]) * phase_shift[II_freq]

    noise_random = np.real(scipy.fft.ifft(s_shifted))
    return noise_random


def produce_randomized_noise(noise, num_of_random,
                             rng=np.random.default_rng(None)):
    '''function to produce num_of_random randomized noise
    produce_randomized_noise(noise, num_of_random,
                             rng=np.random.default_rng(None))
    return np array with shape of num_of_random x npt_noise
    '''

    noise_array = []
    # randomly produce num_of_random pieces of random noise
    for i in range(num_of_random):
        noise_random = randomization_noise(noise, rng=rng)
        noise_array.append(noise_random)

    noise_array = np.array(noise_array)
    return noise_array


def plot_and_compare_randomized_noise():
    plt.close('all')
    plt.figure()
    plt.plot(time_noise, noise_BH1, '-r')
    plt.plot(time_noise, noise_BH1_random, '-b')

    # % Compare the STFT distribution of noise and randomized noise, looks better.
    # the randomization of noise in the Fourier domain seems to be able to wipe out
    # the signal residual while keep the general STFT structure of noise

    # TODO: refactor this part to generate randomized noise
    twin = 100
    toverlap = 50
    win_type = 'hann'

    # apply the thresholding method in the STFT to separate the noise and signals
    f, t, Sxx = sgn.stft(noise_BH1, fs, nperseg=int(twin / dt),
                         noverlap=int(toverlap / dt), window=win_type)
    vmax = np.amax(abs(Sxx))

    plt.figure()
    plt.subplot(221)
    plt.pcolormesh(t, f, np.abs(Sxx), shading='auto', vmax=vmax/1.2)
    plt.title('STFT of origianl noise')
    plt.ylim(0, 20)

    plt.subplot(223)
    plt.pcolormesh(t, f, np.abs(Sxx), shading='auto', vmax=vmax/1.2)
    plt.title('STFT of origianl noise')
    plt.ylim(0, 1)
    plt.show()

    # apply the thresholding method in the STFT to separate the noise and signals
    f, t, Sxx = sgn.stft(noise_BH1_random, fs, nperseg=int(twin / dt),
                         noverlap=int(toverlap / dt), window=win_type)

    plt.subplot(222)
    plt.pcolormesh(t, f, np.abs(Sxx), shading='auto', vmax=vmax/1.2)
    plt.title('STFT of randomized noise')
    plt.ylim(0, 20)
    plt.show()

    plt.subplot(224)
    plt.pcolormesh(t, f, np.abs(Sxx), shading='auto', vmax=vmax/1.2)
    plt.title('STFT of randomized noise')
    plt.ylim(0, 1)
    plt.show()


# %%
# initialize X_train and Y_train
X_train = []
Y_train = []

# load the data
hdf5_files = glob.glob('./waveforms/noise/*.hdf5')
#hdf5_files = [hdf5_files[np.random.randint(0, len(hdf5_files))]]

N_random = 10  # randomized noise for each noise data
for ifile, hdf5_file in enumerate(hdf5_files):#[0:1]):

    with h5py.File(hdf5_file, 'r') as f:
        time_noise = f['noise_time'][:]
        dt = time_noise[1]-time_noise[0]
        fs = 1/dt
        noise_BH1 = f['noise']['BH1'][:]
        noise_BH2 = f['noise']['BH2'][:]
        noise_BHZ = f['noise']['BHZ'][:]
        print(hdf5_file)

    rng = np.random.default_rng(seed=1)
    noise_BH1_random = randomization_noise(noise_BH1, rng=rng)

    # plot_and_compare_randomized_noise()
    noise_BHZ_random = produce_randomized_noise(noise_BHZ, N_random, rng=rng)


# % % #TODO: prepare the synthetic seismic signals
# use ricker wavelet first? Only look at BHZ component first
# the input synthetic waveforms can be other types

    # duration and number of points in the Ricker wavelet
    ricker_len = 60
    npt_ricker = int(ricker_len/dt)
    N_segments = int(noise_BHZ_random.shape[1]/npt_ricker)
    syn_time = np.arange(0, npt_ricker) * dt

    for i_seg in range(N_segments):

        # random parameters for the synthetics to set
        peak_loc = np.random.random() * ricker_len
        fc = np.random.random() * 5
        amp = np.random.random() * 10

        syn_seismic = ricker.ricker(
            f=fc, len=ricker_len, dt=dt, peak_loc=peak_loc)
        syn_seismic = amp * np.std(noise_BHZ) * syn_seismic
        
        

        syn_signal = noise_BHZ[i_seg * npt_ricker:(i_seg + 1) * npt_ricker]  \
            + syn_seismic
        
        X_train.append(syn_signal)
        Y_train.append(syn_seismic)
        
X_train = np.array(X_train)
Y_train = np.array(Y_train)

#%% save the prepared data
with h5py.File('training_datasets.hdf5', 'w') as f:    
    f.create_dataset('time', data=syn_time)
    f.create_dataset('X_train', data=X_train)
    f.create_dataset('Y_train', data=X_train)


#%%
# plt.figure()
# plt.subplot(211)
# plt.plot(syn_time, syn_seismic * amp * np.std(noise_BHZ))
# plt.subplot(212)
# plt.plot(syn_time, syn_signal)


# %% #TODO: cut the noise pieces into several short segments


# %% #TODO: randomly combine the synthetic signal with noise, reshape to the
# proper dimension: nsample * nt * 1


# Not in use now
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
