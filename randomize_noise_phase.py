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
import sys
from matplotlib import pyplot as plt
import cmath

# import the ASDF format module
import asdf

# functions for STFT (spectrogram)
from scipy import signal as sgn

#%% Try read the asdf data
# ff = asdf.open('./waveforms/events_data_processed/IU.XMAS.M7.3.20190714-091050.asdf')
# ff.tree
# time_wave = ff.tree['waveform_time']
# wave_1 = ff.tree['waveforms']['BH1']
# wave_2 = ff.tree['waveforms_denoised']['BH1']


ff = asdf.open('./waveforms/noise/IU.XMAS.M6.0.20190411-081821.asdf')
ff.tree
time_noise = ff.tree['noise_time']
dt = time_noise[1]-time_noise[0]
fs = 1/dt
noise_BH1 = ff.tree['noise']['BH1']
noise_BH2 = ff.tree['noise']['BH2']
noise_BHZ = ff.tree['noise']['BHZ']

#%%
twin=100
toverlap=50
win_type='hann'

# apply the thresholding method in the STFT to separate the noise and signals
f, t, Sxx = sgn.stft(noise_BH1, fs, nperseg=int(twin / dt),
                         noverlap=int(toverlap / dt), window=win_type)

# # check the distribution of STFT phase
# phase_Sxx = np.angle(Sxx.flatten())
# plt.figure()
# plt.hist(phase_Sxx)
# plt.show()
# # checked: normal distribution between -pi to pi

# produce the random phase shift
phase_angle_shift = (rd.rand(Sxx.shape[0], Sxx.shape[1])-0.5)*2*np.pi
phase_shift = np.exp(np.ones(Sxx.shape)*phase_angle_shift*1j)

Sxx_shifted = np.abs(Sxx) * phase_shift


time_temp, noise_BH1_random = sgn.istft(Sxx_shifted, fs, nperseg=int(twin / dt),
                                         noverlap=int(toverlap / dt), window=win_type)

# interpolate the denoised waveform to the same time axis as the original waveforms
noise_BH1_random = np.interp(time_noise, time_temp, noise_BH1_random, left=0, right=0)

plt.figure()
plt.plot(time_noise, noise_BH1, '-r')
plt.plot(time_noise, noise_BH1_random, '-b')
plt.show()