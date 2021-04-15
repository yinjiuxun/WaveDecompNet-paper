#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:37:54 2021

@author: Yin9xun
"""
#%%
import os
import numpy as np
import sys
import datetime
from matplotlib import pyplot as plt

# import the Obspy modules that we will use in this exercise
import obspy

# functions for STFT (spectrogram)
from scipy import signal as sgn

# import the modules with different threshold functions
sys.path.append('/Users/Yin9xun/Work/island_stations/SeisDenoise')
import threshold_functions as threshold

#%% loop over catalog to read waveforms
working_dir = '/Users/Yin9xun/Work/island_stations/waveforms'
#Read catalog first
catalog = obspy.read_events(working_dir + '/events_by_distance.xml')
print(catalog)

#%% make output directory
figure_output_dir = working_dir + '/events_data_figures'
if not os.path.exists(figure_output_dir):
    os.makedirs(figure_output_dir)
    
#%%
i_event0 = np.random.randint(0,193)
plt.close('all')
for i_event in range(len(catalog)):
    event = catalog[i_event]
    #% % extract the event information
    event_time = event.origins[0].time
    event_lon = event.origins[0].longitude
    event_lat = event.origins[0].latitude
    event_dep = event.origins[0].depth/1e3
    event_mag = event.magnitudes[0].mag
    
    # read 3-component event wavefroms 
    event_name = "IU.XMAS" + ".M" + str(event_mag) + "." + event_time.strftime("%Y%m%d-%H%M%S")
    fname = "/events_data/" + event_name + '.mseed'
    
    try:
        tr = obspy.read(working_dir + fname)
    except:
        print("Issue with " + "event " + event_time.strftime("%Y%m%d-%H%M%S"))
        continue
    
    
    for i_chan in range(len(tr)): # loop over each channel
        channel = tr[i_chan].stats.channel
        st0 = tr[i_chan]
        st = st0.copy()
        #st.decimate(factor=5, strict_length=False)
        data = st.data
        time = st.times()
        dt = st.stats.delta
        fs = 1/dt
        
        # parameters about the STFT
        twin = 100
        toverlap = 50
        win_type = 'hann'
        
        f, t, Sxx = sgn.stft(data, fs, nperseg=int(twin/dt), 
                                noverlap=int(toverlap/dt), window=win_type)        
        # apply STFT and calculate the soft threshold value gammaN
        I_positive = np.abs(Sxx) > 1e-16
        #gammaN = np.sqrt(2*np.log(len(time_series))) * np.std(np.abs(Sxx[I_positive]).flatten()) #* 1.4826
        gammaN = np.sqrt(2*np.log(len(Sxx[I_positive]))) * np.std(np.abs(Sxx[I_positive]).flatten())
        
        # thresholded the TF domain
        Sxx_thresholded = threshold.soft_threshold(Sxx, gammaN)
        
        time_temp, data_denoised = sgn.istft(Sxx_thresholded, fs, nperseg=int(twin/dt), 
                                noverlap=int(toverlap/dt), window=win_type)
    
        # interpolate the denoised waveform to the same time axis as the original waveforms
        data_denoised = np.interp(time, time_temp, data_denoised, left=0, right=0)
        
        plt.figure(i_event, figsize=(12,12))
        plt.subplot(len(tr),2,i_chan*2+1)
        plt.plot(time, data, '-b', alpha=1)
        
        plt.plot(time, data_denoised, '-r', linewidth=0.3, alpha=0.8, zorder = 5)
        if i_chan == 3:
            plt.xlabel('Time (s)')
            
        plt.title(event_name + '.' + channel)
        plt.subplot(len(tr),2,i_chan*2+2)
        plt.plot(time, data-data_denoised, '-k', alpha=0.9, linewidth=0.3, zorder = 3)
        if i_chan == 3:
            plt.xlabel('Time (s)')
            
        plt.ylim(-2e-5,2e-5)
        plt.title('Noise in ' + channel)
        
        #plt.show()
        plt.savefig(figure_output_dir + '/'+ event_name + '.png')
        
#%% TODO: save the separated data and noise waveforms
