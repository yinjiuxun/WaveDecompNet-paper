#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:21:46 2021

@author: Yin9xun
"""
#%%
import os
import numpy as np
import datetime
from matplotlib import pyplot as plt

# import the Obspy modules that we will use in this exercise
import obspy
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

# functions for fft
from scipy.fft import fft, fftfreq, fftshift

from scipy import signal as sgn

#%%
working_dir = '/Users/Yin9xun/Work/island_stations/waveforms'


#%% Read catalog first
catalog = obspy.read_events(working_dir + '/events_by_distance.xml')
print(catalog)

#%% Check STA LTA
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta
from obspy.signal.trigger import plot_trigger

for i_event in [10]:#range(0,len(catalog),20):
    event = catalog[i_event]
    #% % extract the event information
    event_time = event.origins[0].time
    event_lon = event.origins[0].longitude
    event_lat = event.origins[0].latitude
    event_dep = event.origins[0].depth/1e3
    event_mag = event.magnitudes[0].mag
    
    # read event wavefroms
    
    event_name = "IU.XMAS" + ".M" + str(event_mag) + "." + event_time.strftime("%Y%m%d-%H%M%S")
    fname = "/events_data/" + event_name + '.mseed'
    
    try:
        tr = obspy.read(working_dir + fname)
    except:
        print("Issue with " + "event " + event_time.strftime("%Y%m%d-%H%M%S"))
        continue

# look at the waveforms 0: BH1, 1: BH2, 2: BHZ
    st = tr[0]
    st = st.filter('bandpass', freqmin=0.01,freqmax=0.05)
    data = st.data
    time = st.times()
    dt = st.stats.delta
    fs = 1/dt
    cft = recursive_sta_lta(data, int(5 * fs), int(20 * fs))
    plot_trigger(st, cft, 2, 0.5)