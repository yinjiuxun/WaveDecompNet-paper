#!/usr/bin/env python
# coding: utf-8

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


#%% function to download the seismograms give the network, station, startime and endtime of the waveform
def download_stations(network, stations, channels, path, location='10'):

    
    for station in stations:
        for channel in channels:
            fname = path + "/" + network + "." + station + "." + channel 
            inventory = client.get_stations(network=network, station=station, channel=channel, 
                                  location=location,level="response")
            inventory.write(fname + ".xml", format="STATIONXML") 

# download waveform given network          
def download_waveforms(network, stations, channels, starttime, endtime, path, location='10'):
    
    for station in stations:
        for channel in channels:
            fname = path + "/" + network + "." + station + "." + channel + "." + starttime.strftime('%Y%m%d')
            tr = client.get_waveforms(network=network, station=station, channel=channel, 
                                      location=location, starttime = starttime - 1800, endtime=endtime + 1800,
                                      attach_response=True)
            tr.detrend("spline", order=3, dspline=1000)
            tr.remove_response(output="VEL")
            
            # here to deal with the taperring at both end, only keep the central 1-hour long data
            newtr = tr.slice(starttime, endtime) # probably fix later
            
            newtr.write(fname + ".mseed", format='MSEED')
            

#%%
working_dir = '/Users/Yin9xun/Work/island_stations'
if not os.path.exists(working_dir):
    os.makedirs(working_dir)
os.chdir(working_dir)

print(os.getcwd())

#%%
#@title Step 2.1 Specify the stations and time { display-mode: "both" }
# Specify the network and stations we want.
network = 'IU' # A specific seismic network to which the stations belong 
stations = np.array(['XMAS']) # Names of the stations
channels = np.array(['BHE','BHN','BHZ']) # Channels
# Specify begining time of the waveforms
year = 2011 
month = 3 
day = 11 
hour = 0 
minute = 0 
second = 0 
seconds_per_day = 24 * 60 * 60



station_dir = working_dir + '/stations'
if not os.path.exists(station_dir):
    os.makedirs(station_dir)
waveform_dir = working_dir + '/waveforms'
if not os.path.exists(waveform_dir):
    os.makedirs(waveform_dir)


#%%
client = Client('IRIS', debug=True)

download_stations(network, stations, channels, station_dir)

# Set the time of waveform we want (1 hour long)
starttime=obspy.UTCDateTime(year, month, day, hour, minute, second)
endtime=obspy.UTCDateTime(year, month, day+1, hour, minute, second)
download_waveforms(network, stations, channels, starttime, endtime, waveform_dir)


# In[ ]:




