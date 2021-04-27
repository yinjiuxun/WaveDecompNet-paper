#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:08:45 2021

@author: Yin9xun
"""
#%%
import os
import numpy as np
import datetime
from matplotlib import pyplot as plt

# import the Obspy modules that we will use in this exercise
import obspy

# functions for fft
from scipy.fft import fft, fftfreq, fftshift

# functions for STFT (spectrogram)
from scipy import signal as sgn

#%%
# soft thresholding
def soft_threshold(y, gamma):
    II1 = np.abs(y) > gamma
    II2 = np.abs(y) <= gamma
    x = y.copy()
    x[II1] = x[II1] * (np.abs(x[II1]) - gamma) / np.abs(x[II1])
    x[II2] = 0
    return x

# hard thresholding
def hard_threshold(y, gamma):
    x = y.copy()
    x[np.abs(x)<gamma] = 0
    return x

# scale the coef to denoise
def scale_to_denoise(Sxx, gammaN, Mmax):
    II1 = np.abs(Sxx) > Mmax
    II2 = (np.abs(Sxx) > gammaN) & (np.abs(Sxx) <= Mmax)
    II3 = np.abs(Sxx) < gammaN
    
    Sxx_processed = Sxx.copy()
    #Twxo_processed[II1] = Twxo[II1] * (np.abs(Twxo[II1]) - Mmax) / np.abs(Twxo[II1])
    Sxx_processed[II1] = Sxx[II1]
    Sxx_processed[II2] = Sxx[II2] * (np.abs(Sxx[II2]) - gammaN) / np.abs(Sxx[II2])
    Sxx_processed[II3] = 0
    
    return Sxx_processed

#%% read the waveforms
working_dir = '/Users/Yin9xun/Work/island_stations'
os.chdir(working_dir)
tr = obspy.read(working_dir + '/waveforms/events_data/*.mseed')
#%%
plt.close('all')

i_tr = np.random.randint(0,len(tr))
print(i_tr)
#i_tr = 192
st0 = tr[i_tr]
st = st0.copy()
#st.decimate(factor=5, strict_length=False)
data = st.data
time = st.times()
dt = st.stats.delta
fs = 1/dt

# noise is 3500 s before P arrival, signal is 3600 s after P arrival
noise = data[time <3600]
signal = data[(time >=3600) & (time <7200)]

plt.figure()
plt.plot(time, data)
plt.plot(time[time <3600], noise)
plt.plot(time[(time >=3600) & (time <7200)], signal)
plt.show()


#%% list of different parts of the waveforms
wave_list = [signal, data, noise]
wave_list = [signal, data, data]
time_list = [time[(time >=3600) & (time <7200)], time, time[time <3600]]
time_list = [time[(time >=3600) & (time <7200)], time, time]
title_list = ['Signal segment', 'Entire data', 'Noise segment']

#%% Spectrogram (original)
fig, ax = plt.subplots(2, 2, figsize=(14,8))
ax = ax.flatten()

ax[0].plot(time, data,'-k')
ax[0].plot(time[time <3600], noise,'-g')
ax[0].plot(time[(time >=3600) & (time <7200)], signal,'-b')
ax[0].set_title('Waveform')

for i_ax in range(len(wave_list)):
    
    time_series = wave_list[i_ax]
    time_series_time = time_list[i_ax]
    
    f, t, Sxx = sgn.stft(time_series, fs, nperseg=int(100/dt), 
                                noverlap=int(90/dt), window='hann')
    if i_ax == 0:
        vmax = np.max(np.abs(Sxx.flatten()))
        vmin = np.min(np.abs(Sxx.flatten()))
    
    sb = ax[i_ax + 1].pcolormesh(t + time_series_time[0], f, np.abs(Sxx), shading='gouraud', vmax=vmax/1.4, vmin=vmin)
    
    if i_ax == 1:
        I_positive = np.abs(Sxx) > 1e-16
        gammaN_all = np.sqrt(2*np.log(len(time_series))) * np.median(np.abs(np.abs(Sxx[I_positive]) - np.median(np.abs(Sxx[I_positive])))) * 1.4826
             
    if i_ax != 1:
        fig.colorbar(sb, ax=ax[i_ax + 1])
        
    if i_ax == 2:
        I_positive = np.abs(Sxx) > 1e-16
        #gammaN = np.sqrt(2*np.log(len(time_series))) * np.median(np.abs(np.abs(Sxx[I_positive]) - np.median(np.abs(Sxx[I_positive])))) * 1.4826
        #gammaN = np.sqrt(2*np.log(len(time_series))) * np.std(np.abs(Sxx[I_positive]).flatten()) #* 1.4826
        gammaN = np.sqrt(2*np.log(len(Sxx[I_positive]))) * np.std(np.abs(Sxx[I_positive]).flatten())

        #Mmax = np.mean(np.max(abs(Sxx)))
        Mmax = np.max(abs(Sxx))/1
        #gammaN = Mmax/1.5
        
    ax[i_ax + 1].set_ylabel('Frequency [Hz]')
    ax[i_ax + 1].set_xlabel('Time [sec]')
    ax[i_ax + 1].set_yscale('log')
    ax[i_ax + 1].set_title(title_list[i_ax])
    ax[i_ax + 1].set_ylim(0.01,st.stats.sampling_rate/2)

plt.show()


#%% Spectrogram (thresholod denoising)
fig, ax = plt.subplots(2, 2, figsize=(14,8))
ax = ax.flatten()
ax[0].plot(time, data, '-k')
wave_list_denoised = []

for i_ax in range(len(wave_list)):
    
    time_series = wave_list[i_ax]
    time_series_time = time_list[i_ax]
    
    f, t, Sxx0 = sgn.stft(time_series, fs, nperseg=int(100/dt), 
                                noverlap=int(90/dt), window='hann')
    
    Sxx = soft_threshold(Sxx0, gammaN)
    #Sxx = scale_to_denoise(Sxx0, gammaN, Mmax)
    time_temp, time_series_denoise = sgn.istft(Sxx, fs, nperseg=int(100/dt), 
                                noverlap=int(90/dt), window='hann')
    
    # interpolate the denoised waveform to the same time axis as the original waveforms
    time_series_denoise = np.interp(time_series_time, time_temp, time_series_denoise, left=0, right=0)
    
    wave_list_denoised.append(time_series_denoise)
    
    ax[0].plot(time_series_time, time_series_denoise)
    
    if i_ax == 0:
        vmax = np.max(np.abs(Sxx.flatten()))
        vmin = np.min(np.abs(Sxx.flatten()))
    
    sb = ax[i_ax + 1].pcolormesh(t + time_series_time[0], f, np.abs(Sxx), shading='gouraud', vmax=vmax/1.4, vmin=vmin)
    
    if i_ax != 1:
        fig.colorbar(sb, ax=ax[i_ax + 1])
        
    ax[i_ax + 1].set_ylabel('Frequency [Hz]')
    ax[i_ax + 1].set_xlabel('Time [sec]')
    ax[i_ax + 1].set_yscale('log')
    ax[i_ax + 1].set_title(title_list[i_ax])
    ax[i_ax + 1].set_ylim(0.01,st.stats.sampling_rate/2)

ax[0].set_title('Waveform')
plt.show()

#%%
noise_series = wave_list[1] - wave_list_denoised[1]
plt.figure()
plt.plot(time_list[1], noise_series)
plt.plot(time_list[1], wave_list[1], '-k', alpha=0.4)
plt.plot(time_list[1], wave_list_denoised[1], '-r', alpha=0.4)
plt.show()

#%% Randomization in STFT domain
# Randomize the phase in the STFT domain

twin=100
toverlap=50
win_type='hann'

# apply the thresholding method in the STFT to separate the noise and signals
f, t, Sxx = sgn.stft(noise_BH1, fs, nperseg=int(twin / dt),
                          noverlap=int(toverlap / dt), window=win_type)
vmax = np.amax(abs(Sxx))

plt.figure()
plt.pcolormesh(t, f, np.abs(Sxx), shading='auto', vmax=vmax/1.2)
plt.show()

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