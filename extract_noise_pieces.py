#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:37:54 2021

@author: Yin9xun
"""
# %%
import threshold_functions as threshold
import os
import numpy as np
import sys
import datetime
from matplotlib import pyplot as plt

# import the ASDF format module
import asdf

# import the HDF5 module
import h5py

# import the Obspy modules that we will use in this exercise
import obspy

# functions for STFT (spectrogram)
from scipy import signal as sgn

# import the modules with different threshold functions
sys.path.append('./')


# %% define the refactoring functions

def stft_thresholding_denoise(twin=100, toverlap=50, win_type='hann', threshold_type='soft'):
    # apply the thresholding method in the STFT to separate the noise and signals
    f, t, Sxx = sgn.stft(data, fs, nperseg=int(twin / dt),
                         noverlap=int(toverlap / dt), window=win_type)

    # apply STFT and calculate the soft threshold value gammaN
    Sxx0 = Sxx.copy()
    # use the waveform before P arrival to get the noise statistics
    Sxx0 = Sxx0[:, t < 3500]
    I_positive = np.abs(Sxx0) > 1e-16
    gammaN = np.sqrt(
        2 * np.log(len(Sxx0[I_positive]))) * np.std(np.abs(Sxx0[I_positive]).flatten()) / 0.6745

    # thresholded the TF domain
    if threshold_type == 'soft':
        Sxx_thresholded = threshold.soft_threshold(Sxx, gammaN)
    elif threshold_type == 'hard':
        Sxx_thresholded = threshold.hard_threshold(Sxx, gammaN)

    time_temp, data_denoised = sgn.istft(Sxx_thresholded, fs, nperseg=int(twin / dt),
                                         noverlap=int(toverlap / dt), window=win_type)

    # interpolate the denoised waveform to the same time axis as the original waveforms
    data_denoised = np.interp(
        time, time_temp, data_denoised, left=0, right=0)

    return data_denoised


def plot_waveforms():
    # plot waveforms and save the figures
    plt.figure(i_event, figsize=(12, 12))
    plt.subplot(len(tr), 2, i_chan * 2 + 1)
    plt.plot(time, data, '-b', alpha=1)

    plt.plot(time, data_denoised, '-r', linewidth=0.3, alpha=0.8, zorder=5)
    if i_chan == 2:
        plt.xlabel('Time (s)')

    plt.title(event_name + '.' + channel)
    plt.subplot(len(tr), 2, i_chan * 2 + 2)
    plt.plot(time, data - data_denoised, '-k',
             alpha=0.9, linewidth=0.3, zorder=3)
    if i_chan == 2:
        plt.xlabel('Time (s)')

    plt.ylim(-4e4, 4e4)
    plt.title('Noise in ' + channel)


def save_event_waveforms(output_name):
    with h5py.File(output_name, 'w') as f:
        f.attrs.create('event_name', event_name)
        f.attrs.create('event_mag', event_mag)
        f.attrs.create(
            'event_time', event_time.datetime.strftime('%Y%m%d-%H:%M:%S'))
        f.attrs.create('event_lon', event_lon)
        f.attrs.create('event_lat', event_lat)
        f.attrs.create('event_dep', event_dep)
        f.create_dataset('time', data=time)

        # original waveforms
        grp = f.create_group('waveforms')
        grp.create_dataset("BH1", data=waveform_dict['BH1'])
        grp.create_dataset("BH2", data=waveform_dict['BH2'])
        grp.create_dataset("BHZ", data=waveform_dict['BHZ'])
        # denoised waveforms
        grp = f.create_group('waveforms_denoised')
        grp.create_dataset("BH1", data=waveform_denoised_dict['BH1'])
        grp.create_dataset("BH2", data=waveform_denoised_dict['BH2'])
        grp.create_dataset("BHZ", data=waveform_denoised_dict['BHZ'])


def save_noise(output_name):
    with h5py.File(output_name, 'w') as f:
        f.create_dataset('noise_time', data=time)

        # storing the noise in each components
        grp = f.create_group('noise')
        grp.create_dataset("BH1", data=noise_dict['BH1'])
        grp.create_dataset("BH2", data=noise_dict['BH2'])
        grp.create_dataset("BHZ", data=noise_dict['BHZ'])


# %% loop over catalog to read waveforms
working_dir = './waveforms'
# Read catalog first
catalog = obspy.read_events(working_dir + '/events_by_distance.xml')
print(catalog)

# %% make output directory
figure_output_dir = working_dir + '/events_data_figures'
if not os.path.exists(figure_output_dir):
    os.makedirs(figure_output_dir)

processed_waveform_output_dir = working_dir + '/events_data_processed'
if not os.path.exists(processed_waveform_output_dir):
    os.makedirs(processed_waveform_output_dir)

noise_output_dir = working_dir + '/noise'
if not os.path.exists(noise_output_dir):
    os.makedirs(noise_output_dir)
# %%
plt.close('all')
for i_event in range(len(catalog)):
    event = catalog[i_event]
    # extract the event information
    event_time = event.origins[0].time
    event_lon = event.origins[0].longitude
    event_lat = event.origins[0].latitude
    event_dep = event.origins[0].depth / 1e3
    event_mag = event.magnitudes[0].mag

    # read 3-component event wavefroms
    event_name = "IU.XMAS" + ".M" + \
                 str(event_mag) + "." + event_time.strftime("%Y%m%d-%H%M%S")
    fname = "/events_data/" + event_name + '.mseed'

    try:
        tr0 = obspy.read(working_dir + fname)
    except:
        print("Issue with " + "event " + event_time.strftime("%Y%m%d-%H%M%S"))
        continue

    # detrend the signal (need to think more about this step.)
    tr = tr0.copy()
    dt = tr[0].stats.delta
    tr.detrend("spline", order=3, dspline=int(600 / dt))

    # initialize the dictionary to store waveforms
    waveform_dict = {}
    waveform_denoised_dict = {}
    noise_dict = {}

    # loop over each channel
    for i_chan in range(len(tr)):
        channel = tr[i_chan].stats.channel

        # original data without any processing
        data0 = tr0[i_chan].data

        # detrended data to be denoised
        st = tr[i_chan]
        data = st.data

        # other information about the waveforms
        dt = st.stats.delta
        fs = 1 / dt
        time = st.times()

        # parameters about the STFT
        twin = 100
        toverlap = 50
        win_type = 'hann'

        # STFT thresholding denoise
        data_denoised = stft_thresholding_denoise(twin=twin,
                                                  toverlap=toverlap,
                                                  win_type=win_type,
                                                  threshold_type='soft')
        # get the waveforms and noise at each channel
        waveform_dict[channel] = data0
        waveform_denoised_dict[channel] = data_denoised
        noise_dict[channel] = data - data_denoised

        # plot the waveforms
        plot_waveforms()

    # save the waveform figures
    plt.savefig(figure_output_dir + '/' + event_name + '.png')
    plt.close('all')

    # save the waveforms in ASDF format
    save_event_waveforms(processed_waveform_output_dir +
                         '/' + event_name + '.hdf5')

    # save the noise only
    save_noise(noise_output_dir + '/' + event_name + '.hdf5')
