#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:41:08 2021

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
# functions for CWT and SSCWT
from ssqueezepy import cwt, icwt, ssq_cwt, ssq_stft, issq_cwt, ssqueeze


#%% read the waveforms
working_dir = '/Users/Yin9xun/Work/island_stations/waveforms/clear'
tr = obspy.read(working_dir + '/*.mseed')


st0 = tr[0]
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

#%% Spectrogram
fig, ax = plt.subplots(2, 2, figsize=(12,10))
ax = ax.flatten()

ax[0].plot(time, data)
ax[0].plot(time[time <3600], noise)
ax[0].plot(time[(time >=3600) & (time <7200)], signal)

for i_ax in range(len(wave_list)):
    
    time_series = wave_list[i_ax]
    time_series_time = time_list[i_ax]
    
    f, t, Sxx = sgn.spectrogram(time_series, fs, mode='magnitude', nperseg=int(100/dt), 
                                noverlap=int(90/dt), window='hann')
    if i_ax == 0:
        vmax = np.max(Sxx.flatten())
        vmin = np.min(Sxx.flatten())
    
    sb = ax[i_ax + 1].pcolormesh(t + time_series_time[0], f, Sxx, shading='gouraud', vmax=vmax, vmin=vmin)
    
    if i_ax != 1:
        fig.colorbar(sb, ax=ax[i_ax + 1])
        
    ax[i_ax + 1].set_ylabel('Frequency [Hz]')
    ax[i_ax + 1].set_xlabel('Time [sec]')
    ax[i_ax + 1].set_yscale('log')
    ax[i_ax + 1].set_ylim(0.01,5)


plt.show()

#%% SSCWT
time_series = wave_list[1]
time_series_time = time_list[1]
Twxo, Wxo, ssq_freqs,scale, *_ = ssq_cwt(time_series, fs=fs, t=time_series_time, nv=64)

#%%
plt.figure()
plt.subplot(2,1,1)
vmax = np.max(abs(Wxo.flatten()))
vmin = np.min(abs(Wxo.flatten()))
plt.pcolormesh(time_series_time, np.flip(ssq_freqs), abs(Wxo), shading='auto', vmax=vmax, vmin=vmin)
plt.yscale('log')
plt.ylim(0.01,5)

plt.subplot(2,1,2)
vmax = np.max(abs(Twxo.flatten()))
vmin = np.min(abs(Twxo.flatten()))
plt.pcolormesh(time_series_time, ssq_freqs, abs(Twxo), shading='auto', vmax=vmax/5, vmin=vmin)
plt.yscale('log')
plt.ylim(0.01,5)
plt.show()

#%% CWT + SSCWT all together (WARNING: MAY TAKE TOO LONG)
fig, ax = plt.subplots(2, 2, figsize=(12,10))
ax = ax.flatten()

ax[0].plot(time, data)
ax[0].plot(time[time <3600], noise)
ax[0].plot(time[(time >=3600) & (time <7200)], signal)

for i_ax in range(len(wave_list)):
    
    time_series = wave_list[i_ax]
    time_series_time = time_list[i_ax]
    
    Twxo, Wxo, ssq_freqs,scale, *_ = ssq_cwt(time_series, fs=fs, t=time_series_time)
    
    if i_ax == 0:
        vmax = np.max(abs(Wxo.flatten()))
        vmin = np.min(abs(Wxo.flatten()))
    
    sb = ax[i_ax + 1].pcolormesh(time_series_time, scale, abs(Wxo), shading='gouraud', vmax=vmax, vmin=vmin)
    
    if i_ax != 1:
        fig.colorbar(sb, ax=ax[i_ax + 1])
        
    ax[i_ax + 1].set_ylabel('Frequency [Hz]')
    ax[i_ax + 1].set_xlabel('Time [sec]')
    ax[i_ax + 1].set_yscale('log')


plt.show()

