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

#%%
client = Client('IRIS', debug=True)

#%% function to download the seismograms give the network, station, startime and endtime of the waveform
def download_station(network, station, channel, location, path):

    fname = path + "/" + network + "." + station + "." + location + "." + channel 
    inventory = client.get_stations(network=network, station=station, channel=channel, 
                          location=location,level="response")
    inventory.write(fname + ".xml", format="STATIONXML") 
    return inventory

# download waveform given network          
def download_waveforms(network, station, channel, location, starttime, endtime, path, fname):
    
    tr = client.get_waveforms(network=network, station=station, channel=channel, 
                              location=location, starttime = starttime - 1800, endtime=endtime + 1800,
                              attach_response=True)
    tr.detrend("spline", order=3, dspline=1000)
    tr.remove_response(output="VEL")
    
    # here to deal with the taperring at both end, only keep the central 1-hour long data
    newtr = tr.slice(starttime, endtime) # probably fix later
    
    newtr.write(path + "/" + fname + ".mseed", format='MSEED')
    return newtr
            

#%%
working_dir = '/Users/Yin9xun/Work/island_stations'
if not os.path.exists(working_dir):
    os.makedirs(working_dir)
os.chdir(working_dir)

print(os.getcwd())


#%% Specify the network and stations we want.
networks = np.array(['IU']) # A specific seismic network to which the stations belong 
stations = np.array(['XMAS']) # Names of the stations
channels = np.array(['BHE','BHN','BHZ']) # Channels
location = '10'


station_dir = working_dir + '/stations'
if not os.path.exists(station_dir):
    os.makedirs(station_dir)
    
for network in networks:
    for station in stations:
        for channel in channels:
            inventory = download_station(network, station, channel, location, station_dir)    
            
#%% Get the location of station
sta_lon = inventory[0][0].longitude
sta_lat = inventory[0][0].latitude

#%% load the USGS earthquake catalog (TODO: this can be modified as directly downloading using Obspy with conditions)
catalog = obspy.read_events(working_dir + '/USGS_catalog_Tohoku_1year_M6+.xml')
print(catalog)
print(catalog[0])

#%%
event_mag_pre = 99 # this variable is used to record the magnitude of previous event
# for events with the same magnitude, only download one
for i_event in range(len(catalog)):
    event = catalog[i_event]
    print(event)
    print(event.origins[0])
    #% % extract the event information
    event_time = event.origins[0].time
    event_lon = event.origins[0].longitude
    event_lat = event.origins[0].latitude
    event_dep = event.origins[0].depth/1e3
    event_mag = event.magnitudes[0].mag
    
    # # if the current event has the same magnitude as the previous one, skip
    # # this step is not necessary for actual application
    # if event_mag == event_mag_pre:
    #     continue
    # else:
    #     event_mag_pre = event_mag
    
    #% % estimate the distance and the P arrival time from the event to the station
    distance_to_source = locations2degrees(sta_lat, sta_lon, event_lat, event_lon)
    model = TauPyModel(model='iasp91')
    
    arrivals = model.get_ray_paths(event_dep, distance_to_source, phase_list=['P'])
    P_arrival = arrivals[0].time
    
    #% % determine the time window of the waveform based on the P arrival, will download 1 hour before and 2 hours after P
    starttime = event_time + P_arrival - 1 * 3600
    endtime = event_time + P_arrival + 2 * 3600
    
    
    waveform_dir = working_dir + '/waveforms'
    if not os.path.exists(waveform_dir):
        os.makedirs(waveform_dir)
    

    fname = network + "." + station + ".M" + str(event_mag) + "." + event_time.strftime("%Y%m%d-%H%M%S")
    try:
        tr = download_waveforms(network, station, 'BH*', location, starttime, endtime, waveform_dir, fname)
    except:
        print("Issue with " + "event " + event_time.strftime("%Y%m%d-%H%M%S"))


#%%
tr.plot()


