#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 12:38:02 2021

@author: Yin9xun
"""
# %%
import os
import numpy as np
from numpy import random as rd
import scipy
import sys
import glob
from matplotlib import pyplot as plt

# import the ASDF format module
import asdf

# functions for STFT (spectrogram)
from scipy import signal as sgn

# %% Try read the asdf data
# ff = asdf.open('./waveforms/events_data_processed/IU.XMAS.M7.3.20190714-091050.asdf')
# ff.tree
# time_wave = ff.tree['waveform_time']
# wave_1 = ff.tree['waveforms']['BH1']
# wave_2 = ff.tree['waveforms_denoised']['BH1']

asdf_files = temp = glob.glob('./waveforms/noise/*.asdf')
asdf_files = temp = glob.glob(
    './waveforms/noise/IU.XMAS.M6.0.20190118-131832.asdf')

for ifile, asdf_file in enumerate(asdf_files[0:1]):
    ff = asdf.open(asdf_file)
    ff.tree
    time_noise = ff.tree['noise_time']
    dt = time_noise[1]-time_noise[0]
    fs = 1/dt
    noise_BH1 = ff.tree['noise']['BH1']
    noise_BH2 = ff.tree['noise']['BH2']
    noise_BHZ = ff.tree['noise']['BHZ']
    print(asdf_file)


# %% Randomize the phase in the Fourier domain
s = scipy.fft.fft(noise_BH1)
phase_angle_shift = (rd.rand(len(s))-0.5)*2*np.pi
# make sure the inverse transform is real
phase_angle_shift[0] = 0
phase_angle_shift[int(len(s)/2+1):(len(s)+1)] = -1 * \
    np.flip(phase_angle_shift[1:int(len(s)/2+1)])


phase_shift = np.exp(np.ones(s.shape)*phase_angle_shift*1j)
s_shifted = np.abs(s) * phase_shift

noise_BH1_random = np.real(scipy.fft.ifft(s_shifted))

plt.figure()
plt.plot(time_noise, noise_BH1, '-r')
plt.plot(time_noise, noise_BH1_random, '-b')

# %% Compare the STFT distribution of noise and randomized noise, looks better.
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
plt.subplot(121)
plt.pcolormesh(t, f, np.abs(Sxx), shading='auto', vmax=vmax/1.2)
plt.title('STFT of origianl noise')
plt.ylim(0, 2)

# apply the thresholding method in the STFT to separate the noise and signals
f, t, Sxx = sgn.stft(noise_BH1_random, fs, nperseg=int(twin / dt),
                     noverlap=int(toverlap / dt), window=win_type)

plt.subplot(122)
plt.pcolormesh(t, f, np.abs(Sxx), shading='auto', vmax=vmax/1.2)
plt.title('STFT of randomized noise')
plt.ylim(0, 2)
plt.show()

# %% Randomize the phase in the STFT domain

# twin=100
# toverlap=50
# win_type='hann'

# # apply the thresholding method in the STFT to separate the noise and signals
# f, t, Sxx = sgn.stft(noise_BH1, fs, nperseg=int(twin / dt),
#                          noverlap=int(toverlap / dt), window=win_type)
# vmax = np.amax(abs(Sxx))

# plt.figure()
# plt.pcolormesh(t, f, np.abs(Sxx), shading='auto', vmax=vmax/1.2)
# plt.show()

# # # check the distribution of STFT phase
# # phase_Sxx = np.angle(Sxx.flatten())
# # plt.figure()
# # plt.hist(phase_Sxx)
# # plt.show()
# # # checked: normal distribution between -pi to pi

# # produce the random phase shift
# phase_angle_shift = (rd.rand(Sxx.shape[0], Sxx.shape[1])-0.5)*2*np.pi
# phase_shift = np.exp(np.ones(Sxx.shape)*phase_angle_shift*1j)

# Sxx_shifted = np.abs(Sxx) * phase_shift


# time_temp, noise_BH1_random = sgn.istft(Sxx_shifted, fs, nperseg=int(twin / dt),
#                                          noverlap=int(toverlap / dt), window=win_type)

# # interpolate the denoised waveform to the same time axis as the original waveforms
# noise_BH1_random = np.interp(time_noise, time_temp, noise_BH1_random, left=0, right=0)

# plt.figure()
# plt.plot(time_noise, noise_BH1, '-r')
# plt.plot(time_noise, noise_BH1_random, '-b')
# plt.show()
