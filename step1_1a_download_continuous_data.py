#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 21:23:35 2021

@author: Yin9xun
"""
# %%
import os
import numpy as np
import datetime
from matplotlib import pyplot as plt

# import the Obspy modules that we will use in this exercise
import obspy
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

from utilities import mkdir


# %%
def download_station(network, station, channel, location, path):
    fname = path + "/" + network + "." + station + "." + location + "." + channel
    inventory = client.get_stations(network=network, station=station, channel=channel,
                                    location=location, level="response")
    inventory.write(fname + ".xml", format="STATIONXML")
    return inventory


# download waveform given network
def download_waveforms(network, station, channel, location, starttime, endtime, path, fname):
    tr = client.get_waveforms(network=network, station=station, channel=channel,
                              location=location, starttime=starttime - 1800, endtime=endtime + 1800,
                              attach_response=True)
    # tr.detrend("spline", order=3, dspline=1000)
    # tr.remove_response(output="VEL")

    # here to deal with the taperring at both end, only keep the central 1-hour long data
    newtr = tr.slice(starttime, endtime)  # probably fix later

    newtr.write(path + "/" + fname + ".mseed", format='MSEED')
    return newtr


# %%
client = Client('IRIS', debug=True)

# %%
working_dir = os.getcwd()

# %% Specify the network and stations we want.
networks = np.array(['IU'])  # A specific seismic network to which the stations belong
stations = np.array(['POHA'])  # Names of the stations
channels = np.array(['BH1', 'BH2', 'BHZ'])  # Channels
location = '00' #'10'

waveform_dir = working_dir + '/continuous_waveforms'
mkdir(waveform_dir)

station_dir = waveform_dir + '/stations'
mkdir(station_dir)

for network in networks:
    for station in stations:
        for channel in channels:
            inventory = download_station(network, station, channel, location, station_dir)

        # %%
sta_lat = inventory[0][0].latitude
sta_lon = inventory[0][0].longitude

# %% Search for the events within a given range
t1 = obspy.UTCDateTime("2021-08-01")
t2 = obspy.UTCDateTime("2021-09-01")

catalog = client.get_events(starttime=t1, endtime=t2, minmagnitude=2.5,
                            latitude=sta_lat, longitude=sta_lon, minradius=5, maxradius=90,
                            maxdepth=100)

catalog.write(waveform_dir + "/events_by_distance.xml", format="QUAKEML")

# % % determine the time window of the waveform based on the P arrival, will download 1 hour before and 2 hours
# after P
starttime = t1 - 1 * 3600
endtime = t2 + 1 * 3600

fname = network + "." + station + "." + location + "." + starttime.strftime("%Y%m%d") + "-" + endtime.strftime("%Y%m%d")
try:
    tr = download_waveforms(network, station, 'BH*', location, starttime, endtime, waveform_dir, fname)
except:
    print("Issue downloading data")

