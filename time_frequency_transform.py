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
from scipy import signal
# functions for CWT and SSCWT
from ssqueezepy import cwt, icwt, ssq_cwt, ssq_stft, issq_cwt, ssqueeze


#%% read the waveforms
working_dir = '/Users/Yin9xun/Work/island_stations/waveforms'
tr = obspy.read(working_dir + '/clear/*.mseed')

#%% Spectrogram
f, t, Sxx = signal.spectrogram(x, fs)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


#%% CWT


#%% SSCWT
