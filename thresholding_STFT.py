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
tr = obspy.read(working_dir + '/waveforms/clear/*.mseed')


st0 = tr[43]
st = st0.copy()
st.decimate(factor=5, strict_length=False)
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
time_list = [time[(time >=3600) & (time <7200)], time, time[time <3600]]
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
    
    if i_ax != 1:
        fig.colorbar(sb, ax=ax[i_ax + 1])
        
    if i_ax == 2:
        gammaN = np.sqrt(2*np.log(len(time_series))) * np.mean(np.abs(np.abs(Sxx) - np.mean(np.abs(Sxx)))) * 1.4826
        Mmax = np.mean(np.max(abs(Sxx), axis=1))
        Mmax = np.max(abs(Sxx))/1
        gammaN = Mmax/2
        
    ax[i_ax + 1].set_ylabel('Frequency [Hz]')
    ax[i_ax + 1].set_xlabel('Time [sec]')
    ax[i_ax + 1].set_yscale('log')
    ax[i_ax + 1].set_title(title_list[i_ax])
    ax[i_ax + 1].set_ylim(0.01,4)

plt.show()


#%% Spectrogram (thresholod denoising)
fig, ax = plt.subplots(2, 2, figsize=(14,8))
ax = ax.flatten()
ax[0].plot(time, data, '-k')

for i_ax in range(len(wave_list)):
    
    time_series = wave_list[i_ax]
    time_series_time = time_list[i_ax]
    
    f, t, Sxx0 = sgn.stft(time_series, fs, nperseg=int(100/dt), 
                                noverlap=int(90/dt), window='hann')
    
    #Sxx = soft_threshold(Sxx0, Mmax)
    Sxx = scale_to_denoise(Sxx0, gammaN, Mmax)
    time_temp, time_series_denoise = sgn.istft(Sxx, fs, nperseg=int(100/dt), 
                                noverlap=int(90/dt), window='hann')
    
    ax[0].plot(time_temp + time_series_time[0], time_series_denoise)
    
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
    ax[i_ax + 1].set_ylim(0.01,4)

ax[0].set_title('Waveform')
plt.show()