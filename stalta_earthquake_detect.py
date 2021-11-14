# %%
import os
import numpy as np
import datetime
from matplotlib import pyplot as plt
import h5py

import obspy
from utilities import mkdir
import matplotlib

matplotlib.rcParams.update({'font.size': 12})

working_dir = os.getcwd()

# waveforms
waveform_dir = working_dir + '/continuous_waveforms'

network_station = 'IU.POHA'
second_per_day = 24 * 3600
tr_earthquake = obspy.read(waveform_dir + '/' + network_station + '.00.20210731-20210901_separated_earthquake.mseed')
tr_raw = obspy.read(waveform_dir + '/' + network_station + '.00.20210731-20210901_original_earthquake.mseed')

scaling_factor_waveform = 1e5
for i in range(3):
    tr_raw[i].data = (tr_raw[i].data / scaling_factor_waveform).astype('float32')
    tr_earthquake[i].data = (tr_earthquake[i].data / scaling_factor_waveform).astype('float32')

f_sample = tr_raw[0].stats.sampling_rate
waveform_time = np.arange(tr_raw[0].stats.npts) * tr_raw[0].stats.delta

st1 = tr_raw.copy()  # raw seismic data
st2 = tr_earthquake.copy()  # separated earthquake data

from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta, carl_sta_trig, coincidence_trigger, plot_trigger

ENZ_color = ['r', 'b', 'g']

# Parameters of the STA/LTA
short_term = 3
long_term = 60
trigger_on = 3.5
trigger_off = 2

# cft = classic_sta_lta(st[0].data, int(short_term * f_sample), int(long_term * f_sample))
# cft = carl_sta_trig(st[0].data, int(short_term * 10), int(long_term * 10), 0.8, 0.8)
cft1_E = recursive_sta_lta(st1[0].data, int(short_term * f_sample), int(long_term * f_sample))
cft1_N = recursive_sta_lta(st1[1].data, int(short_term * f_sample), int(long_term * f_sample))
cft1_Z = recursive_sta_lta(st1[2].data, int(short_term * f_sample), int(long_term * f_sample))

cft2_E = recursive_sta_lta(st2[0].data, int(short_term * f_sample), int(long_term * f_sample))
cft2_N = recursive_sta_lta(st2[1].data, int(short_term * f_sample), int(long_term * f_sample))
cft2_Z = recursive_sta_lta(st2[2].data, int(short_term * f_sample), int(long_term * f_sample))

# trig = coincidence_trigger("classicstalta", trigger_on, trigger_off, st, 3, sta=short_term, lta=long_term)
trig1 = coincidence_trigger("recstalta", trigger_on, trigger_off, st1, 3, sta=short_term, lta=long_term)
trig2 = coincidence_trigger("recstalta", trigger_on, trigger_off, st2, 3, sta=short_term, lta=long_term)

print('Trigger events: Raw Data: ' + str(len(trig1)) + ' vs Separated Data: ' + str(len(trig2)))

detect_time1 = np.array([trig1[i]['time'] - st1[0].stats.starttime for i in range(len(trig1))])
detect_duration1 = np.array([trig1[i]['duration'] for i in range(len(trig1))])

detect_time2 = np.array([trig2[i]['time'] - st2[0].stats.starttime for i in range(len(trig2))])
detect_duration2 = np.array([trig2[i]['duration'] for i in range(len(trig2))])

# plot original data and STA/LTA results
def plot_long_waveform_for_visualization():
    fig, ax = plt.subplots(4, 2, sharex=True, figsize=(14, 12))

    # Original data
    ax[0, 0].plot(waveform_time / second_per_day, cft1_E, color=ENZ_color[0])
    ax[0, 0].plot(waveform_time / second_per_day, cft1_N, color=ENZ_color[1])
    ax[0, 0].plot(waveform_time / second_per_day, cft1_Z, color=ENZ_color[2])

    ax[0, 0].plot(waveform_time[[0, -1]] / second_per_day, np.ones((2, 1)) * trigger_on, color='k')
    ax[0, 0].plot(waveform_time[[0, -1]] / second_per_day, np.ones((2, 1)) * trigger_off, '--k')
    ax[0, 0].set_title('Original waveform')
    ax[0, 0].set_ylabel('STA/LTA ratio')

    ax[1, 0].plot(waveform_time / second_per_day, tr_raw[0].data, color=ENZ_color[0])
    ax[1, 0].plot(detect_time1 / second_per_day, np.ones(detect_time1.shape) * np.mean(tr_raw[0].data), 'kx')
    axi = ax[1, 0]
    ax[1, 0].set_ylabel('E component')

    ax[2, 0].plot(waveform_time / second_per_day, tr_raw[1].data, color=ENZ_color[1])
    ax[2, 0].plot(detect_time1 / second_per_day, np.ones(detect_time1.shape) * np.mean(tr_raw[1].data), 'kx')
    ax[2, 0].sharey(axi)
    ax[2, 0].set_ylabel('N component')

    ax[3, 0].plot(waveform_time / second_per_day, tr_raw[2].data, color=ENZ_color[2])
    ax[3, 0].plot(detect_time1 / second_per_day, np.ones(detect_time1.shape) * np.mean(tr_raw[2].data), 'kx')
    ax[3, 0].sharey(axi)
    ax[3, 0].set_ylabel('Z component')

    # Separated data
    ax[0, 1].plot(waveform_time / second_per_day, cft2_E, color=ENZ_color[0])
    ax[0, 1].plot(waveform_time / second_per_day, cft2_N, color=ENZ_color[1])
    ax[0, 1].plot(waveform_time / second_per_day, cft2_Z, color=ENZ_color[2])

    ax[0, 1].plot(waveform_time[[0, -1]] / second_per_day, np.ones((2, 1)) * trigger_on, color='k')
    ax[0, 1].plot(waveform_time[[0, -1]] / second_per_day, np.ones((2, 1)) * trigger_off, '--k')
    ax[0, 1].sharey(ax[0, 0])
    ax[0, 1].set_ylim(-0.5, 15)
    ax[0, 1].set_title('Separated waveform')

    ax[1, 1].plot(waveform_time / second_per_day, tr_earthquake[0].data, color=ENZ_color[0])
    ax[1, 1].plot(detect_time2 / second_per_day, np.ones(detect_time2.shape) * np.mean(tr_earthquake[0].data), 'kx')
    ax[1, 1].sharey(axi)

    ax[2, 1].plot(waveform_time / second_per_day, tr_earthquake[1].data, color=ENZ_color[1])
    ax[2, 1].plot(detect_time2 / second_per_day, np.ones(detect_time2.shape) * np.mean(tr_earthquake[1].data), 'kx')
    ax[2, 1].sharey(axi)

    ax[3, 1].plot(waveform_time / second_per_day, tr_earthquake[2].data, color=ENZ_color[2])
    ax[3, 1].plot(detect_time2 / second_per_day, np.ones(detect_time2.shape) * np.mean(tr_earthquake[2].data), 'kx')
    ax[3, 1].sharey(axi)

# This part is used to visually find the time window range for zoom in
plt.close('all')
plot_long_waveform_for_visualization()

# plot the histogram for detected earthquakes
plt.figure()
plt.hist(detect_time2 / second_per_day, bins=31)
plt.hist(detect_time1 / second_per_day, bins=31)

# Loop version to plot better Zoom-in figure
def plot_zoom_in_waveform(time_range):
    waveform_time_in_day = waveform_time / second_per_day
    detect_time_in_day1 = detect_time1 / second_per_day
    detect_time_in_day2 = detect_time2 / second_per_day

    index_range = (waveform_time_in_day >= time_range[0]) & (waveform_time_in_day <= time_range[1])

    # time in day showing in waveform
    waveform_time_in_day = waveform_time_in_day[index_range]
    detect_time_in_day1 = detect_time_in_day1[
        (detect_time_in_day1 >= time_range[0]) & (detect_time_in_day1 <= time_range[1])]
    detect_time_in_day2 = detect_time_in_day2[
        (detect_time_in_day2 >= time_range[0]) & (detect_time_in_day2 <= time_range[1])]
    detect_time_list = [detect_time_in_day1, detect_time_in_day2]

    # change to datetime format
    start_time = tr_earthquake[0].stats.starttime.timestamp
    time_in_datetime = np.array([datetime.datetime.fromtimestamp(
        waveform_time_point * second_per_day + start_time).strftime("%m-%d-%I:%M:%S")
                                 for waveform_time_point in waveform_time_in_day])

    # waveform data in the corresponding range
    tr_raw_data = [tr_raw[j].data[index_range] for j in range(3)]
    tr_earthquake_data = [tr_earthquake[j].data[index_range] for j in range(3)]

    trace_list = [tr_raw_data, tr_earthquake_data]
    cft_list = [[cft1_E[index_range], cft1_N[index_range], cft1_Z[index_range]],
                [cft2_E[index_range], cft2_N[index_range], cft2_Z[index_range]]]

    component_list = ['E', 'N', 'Z']
    name_list = ['Original waveform', 'Separated waveform']
    plt.close('all')
    fig, ax = plt.subplots(4, 2, sharex=True, figsize=(14, 12))
    axi = ax[1, 0]

    for i_chan in range(3):
        for i_tr, tr_waveform in enumerate(trace_list):
            ax[0, i_tr].plot(waveform_time_in_day, cft_list[i_tr][i_chan],
                             color=ENZ_color[i_chan], label=component_list[i_chan])

            if i_chan == 2:
                ax[0, i_tr].plot(waveform_time_in_day[[0, -1]], np.ones((2, 1)) * trigger_on,
                                 color='k', label='Trigger ON')
                ax[0, i_tr].plot(waveform_time_in_day[[0, -1]], np.ones((2, 1)) * trigger_off,
                                 '--k', label='Trigger OFF')
                ax[0, 1].sharey(ax[0, 0])

            ax[0, i_tr].set_title(name_list[i_tr])
            ax[0, i_tr].set_ylabel('STA/LTA ratio')

    for i_tr, tr_waveform in enumerate(trace_list):
        detect_time_in_day = detect_time_list[i_tr]
        for i_chan in range(3):
            ax[i_chan + 1, i_tr].plot(waveform_time_in_day, tr_waveform[i_chan], color=ENZ_color[i_chan])
            ax[i_chan + 1, i_tr].plot(detect_time_in_day,
                                      np.ones(detect_time_in_day.shape) * np.mean(tr_waveform[i_chan]), 'kx')
            ax[i_chan + 1, i_tr].sharey(axi)

            ax[i_chan + 1, 0].set_ylabel(component_list[i_chan] + ' component')

            ax[i_chan + 1, i_tr].set_xticks(waveform_time_in_day[[0, int(len(waveform_time_in_day) / 2), -1]])
            ax[i_chan + 1, i_tr].set_xticklabels(time_in_datetime[[0, int(len(waveform_time_in_day) / 2), -1]])

    for i_tr in range(2):
        ax[0, i_tr].legend()


# output some zoom_in_figure
output_dir = waveform_dir + '/STALTA'
mkdir(output_dir)
mkdir(output_dir + '/' + network_station)

waveform_time = np.arange(tr_raw[0].stats.npts) * tr_raw[0].stats.delta

time_range = [11.825, 11.84]
time_range = [8.3615, 8.3645]
time_range = [21.0675, 21.07]

time_range_list = [[8.3615, 8.3645], [11.825, 11.84], [21.0675, 21.07]]
for time_range in time_range_list:
    plot_zoom_in_waveform(time_range)
    file_name = network_station + '_t_' + str(time_range[0]) + '.pdf'
    plt.savefig(output_dir + '/' + network_station + '/' + file_name)





