# %%
import os
import numpy as np
import datetime
from matplotlib import pyplot as plt
import h5py

import obspy

import matplotlib

matplotlib.rcParams.update({'font.size': 12})

working_dir = os.getcwd()

# waveforms
waveform_dir = working_dir + '/continuous_waveforms'

second_per_day = 24 * 3600
tr_earthquake = obspy.read(waveform_dir + '/' + 'IU.POHA.00.20210731-20210901_separated_earthquake.mseed')
tr_raw = obspy.read(waveform_dir + '/' + 'IU.POHA.00.20210731-20210901_original_earthquake.mseed')

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

plt.close('all')
# plot original data and STA/LTA results
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
ax[2, 1].plot(detect_time2 / second_per_day, np.ones(detect_time2.shape) * np.mean(tr_earthquake[0].data), 'kx')
ax[2, 1].sharey(axi)

ax[3, 1].plot(waveform_time / second_per_day, tr_earthquake[2].data, color=ENZ_color[2])
ax[3, 1].plot(detect_time2 / second_per_day, np.ones(detect_time2.shape) * np.mean(tr_earthquake[0].data), 'kx')
ax[3, 1].sharey(axi)

ax[3, 1].set_xlim(2.547, 2.548)
ax[3, 1].set_ylim(-5, 5)

ax[3, 1].set_xlim(8.3615, 8.3645)
ax[3, 1].set_ylim(-0.2, 0.2)

ax[3, 1].set_xlim(11.825, 11.84)
ax[3, 1].set_ylim(-0.5, 0.5)

ax[3, 1].set_xlim(0, 31)
ax[3, 1].set_ylim(-1, 1)


# Loop version, TODO: not show the entire waveform, only show part!
trace_list = [tr_raw, tr_earthquake]
cft_list = [[cft1_E, cft1_N, cft1_Z], [cft2_E, cft2_N, cft2_Z]]
component_list = ['E', 'N', 'Z']
name_list = ['Original waveform', 'Separated waveform']
plt.close('all')
fig, ax = plt.subplots(4, 2, sharex=True, figsize=(14, 12))
axi = ax[1, 0]

for i_chan in range(3):
    for i_tr, tr_waveform in enumerate(trace_list):
        ax[0, i_tr].plot(waveform_time / second_per_day, cft_list[i_tr][i_chan], color=ENZ_color[i_chan])
        ax[0, i_tr].plot(waveform_time[[0, -1]] / second_per_day, np.ones((2, 1)) * trigger_on, color='k')
        ax[0, i_tr].plot(waveform_time[[0, -1]] / second_per_day, np.ones((2, 1)) * trigger_off, '--k')

        ax[0, i_tr].set_title(name_list[i_tr])
        ax[0, i_tr].set_ylabel('STA/LTA ratio')

for i_tr, tr_waveform in enumerate(trace_list):
    for i_chan in range(3):
        ax[i_chan + 1, i_tr].plot(waveform_time / second_per_day, tr_waveform[i_chan].data, color=ENZ_color[i_chan])
        ax[i_chan + 1, i_tr].plot(detect_time1 / second_per_day, \
                                  np.ones(detect_time1.shape) * np.mean(tr_waveform[i_chan].data), 'kx')
        ax[i_chan + 1, i_tr].sharey(axi)

        ax[i_chan + 1, 0].set_ylabel(component_list[i_chan] + ' component')







plt.figure()

plt.hist(detect_time2 / second_per_day, bins=31)
plt.hist(detect_time1 / second_per_day, bins=31)
