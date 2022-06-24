# %%
import os
import numpy as np
import datetime
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import h5py

import obspy
from utilities import mkdir
import matplotlib

matplotlib.rcParams.update({'font.size': 12})
#%%
working_dir = os.getcwd()

model_and_datasets = 'Model_and_datasets_1D_all_snr_40_unshuffled_equal_epoch100'
bottleneck_name = 'LSTM'
network_station = 'IU.POHA'

# waveforms
waveform_dir = '/kuafu/yinjx/WaveDecompNet_dataset/continuous_waveforms/' \
               + model_and_datasets + '/' + network_station + '/' + bottleneck_name

second_per_day = 24 * 3600
tr_earthquake = obspy.read(waveform_dir + '/' + network_station + '.00.20210731-20210901_separated_earthquake.mseed')
tr_raw = obspy.read(waveform_dir + '/' + network_station + '.00.20210731-20210901_original_earthquake.mseed')

scaling_factor_waveform = 1e5
for i in range(3):
    tr_raw[i].data = (tr_raw[i].data / scaling_factor_waveform).astype('float32')
    tr_earthquake[i].data = (tr_earthquake[i].data / scaling_factor_waveform).astype('float32')

f_sample = tr_raw[0].stats.sampling_rate
waveform_time = np.arange(tr_raw[0].stats.npts) * tr_raw[0].stats.delta

highpass_f = None
if highpass_f:
    tr_raw.filter("highpass", freq=highpass_f) # apply a highpass filter 
st1 = tr_raw.copy()  # raw seismic data
st2 = tr_earthquake.copy()  # separated earthquake data

from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta, carl_sta_trig, coincidence_trigger, plot_trigger

ENZ_color = ['r', 'b', 'orange']

# Parameters of the STA/LTA
threshold_coincidence = 1
short_term = 2
long_term = 60
trigger_on = 6
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
trig1 = coincidence_trigger("recstalta", trigger_on, trigger_off, st1, threshold_coincidence, sta=short_term, lta=long_term)
trig2 = coincidence_trigger("recstalta", trigger_on, trigger_off, st2, threshold_coincidence, sta=short_term, lta=long_term)

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

# # This part is used to visually find the time window range for zoom in
# plt.close('all')
# plot_long_waveform_for_visualization()

# load the precalculated event arrival for zoom-in waveforms
with h5py.File(waveform_dir + '/' + bottleneck_name + '_processed_waveforms.hdf5', 'r') as f:
    event_arrival_P = f["event_arrival_P_local"][:]
    event_arrival_S = f["event_arrival_S_local"][:]

# Loop version to plot better Zoom-in figure
# Loop version to plot better Zoom-in figure
def plot_zoom_in_waveform(time_range, strict_xlim=False, time_inset=None):
    waveform_time_in_day = waveform_time / second_per_day
    window_time_in_day = waveform_time_in_day[::600] # this is the time window boundary
    P_arrival_in_day = event_arrival_P / second_per_day
    S_arrival_in_day = event_arrival_S / second_per_day

    detect_time_in_day1 = detect_time1 / second_per_day
    detect_time_in_day2 = detect_time2 / second_per_day

    index_range = (waveform_time_in_day >= time_range[0]) & (waveform_time_in_day <= time_range[1])
    index_window = (window_time_in_day >= time_range[0]) & (window_time_in_day <= time_range[1])
    index_P = (P_arrival_in_day >= time_range[0]) & (P_arrival_in_day <= time_range[1])
    index_S = (S_arrival_in_day >= time_range[0]) & (S_arrival_in_day <= time_range[1])

    # time in day showing in waveform
    waveform_time_in_day = waveform_time_in_day[index_range]
    detect_time_in_day1 = detect_time_in_day1[
        (detect_time_in_day1 >= time_range[0]) & (detect_time_in_day1 <= time_range[1])]
    detect_time_in_day2 = detect_time_in_day2[
        (detect_time_in_day2 >= time_range[0]) & (detect_time_in_day2 <= time_range[1])]
    detect_time_list = [detect_time_in_day1, detect_time_in_day2]

    # change to datetime format
    start_time = tr_earthquake[0].stats.starttime.timestamp
    time_in_datetime = np.array([obspy.UTCDateTime(
        waveform_time_point * second_per_day + start_time).strftime('%m-%d-%H:%M:%S')
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
            ax[0, i_tr].annotate(f'({str(chr(i_tr+97))}) ', xy=(-0.1, 1.1), xycoords=ax[0, i_tr].transAxes, fontsize=15)

    y_lim = 0.2
    for i_tr, tr_waveform in enumerate(trace_list):
        detect_time_in_day = detect_time_list[i_tr]
        for i_chan in range(3):
            ax[i_chan + 1, i_tr].plot(waveform_time_in_day, tr_waveform[i_chan], color=ENZ_color[i_chan])
            ax[i_chan + 1, i_tr].plot(detect_time_in_day,
                                      np.ones(detect_time_in_day.shape) * np.mean(tr_waveform[i_chan]),
                                      'kx', markeredgewidth=1.5, markersize=10, zorder=10)
            ax[i_chan + 1, i_tr].plot(P_arrival_in_day[index_P],
                                      np.ones(P_arrival_in_day[index_P].shape) * np.mean(tr_waveform[i_chan]),
                                      'ko', markeredgewidth=1.5, markersize=3, markerfacecolor='g')
            ax[i_chan + 1, i_tr].plot(S_arrival_in_day[index_S],
                                      np.ones(S_arrival_in_day[index_S].shape) * np.mean(tr_waveform[i_chan]),
                                      'ko', markeredgewidth=1.5, markersize=6, markerfacecolor='g')

            # label the time window
            for window_time in window_time_in_day[index_window]:
                ax[i_chan + 1, i_tr].axvline(x=window_time, color='gray', linewidth=0.5)

            # add inset
            if time_inset is not None:
                rect = patches.Rectangle((time_inset[0][0], time_inset[1][0]),
                                         time_inset[0][1] - time_inset[0][0],
                                         time_inset[1][1] - time_inset[1][0],
                                         linewidth=1, edgecolor='k', facecolor='none', zorder=12)

                # Add the patch to the Axes
                ax[i_chan + 1, i_tr].add_patch(rect)

                axins = ax[i_chan + 1, i_tr].inset_axes([0.45, 0.65, 0.59, 0.39])
                axins.plot(waveform_time_in_day, tr_waveform[i_chan], color=ENZ_color[i_chan])
                axins.plot(detect_time_in_day,
                                          np.ones(detect_time_in_day.shape) * np.mean(tr_waveform[i_chan]),
                                          'kx', markeredgewidth=1.5, markersize=10, zorder=10)
                axins.plot(P_arrival_in_day[index_P],
                                      np.ones(P_arrival_in_day[index_P].shape) * np.mean(tr_waveform[i_chan]),
                                      'ko', markeredgewidth=1.5, markersize=3, markerfacecolor='g')
                axins.plot(S_arrival_in_day[index_S],
                                      np.ones(S_arrival_in_day[index_S].shape) * np.mean(tr_waveform[i_chan]),
                                      'ko', markeredgewidth=1.5, markersize=6, markerfacecolor='g')
                axins.set_xticks([])
                axins.set_yticks([])
                t_interval = waveform_time_in_day[-1] - waveform_time_in_day[0]
                axins.set_xlim(time_inset[0])
                axins.set_ylim(time_inset[1])
                if (i_tr == 0) and (i_chan == 0):
                    axins0 = axins
                else:
                    axins.sharey(axins0)

            ax[i_chan + 1, i_tr].sharey(axi)

            ax[i_chan + 1, 0].set_ylabel(component_list[i_chan] + ' component')

            ax[i_chan + 1, 1].tick_params(axis='y', labelcolor='w')
            ax[i_chan + 1, i_tr].set_xticks(waveform_time_in_day[[0, int(len(waveform_time_in_day) / 2), -1]])
            ax[i_chan + 1, i_tr].set_xticklabels(time_in_datetime[[0, int(len(waveform_time_in_day) / 2), -1]])
            if strict_xlim:
                ax[i_chan + 1, i_tr].set_xlim(time_range)

            if y_lim < 1.1 * np.max(abs(tr_waveform[i_chan])):
                y_lim = 1.1 * np.max(abs(tr_waveform[i_chan]))
            ax[i_chan + 1, i_tr].set_ylim((-y_lim, y_lim))

    for i_tr in range(2):
        ax[0, i_tr].legend(loc=1)

# output some zoom_in_figure
if highpass_f:
    output_dir = waveform_dir + f'/STALTA_highpass_{highpass_f}Hz'
else:
    output_dir = waveform_dir + f'/STALTA'
mkdir(output_dir)

waveform_time = np.arange(tr_raw[0].stats.npts) * tr_raw[0].stats.delta

time_range_list = [[0.3003, 0.3014], [8.362, 8.3638], [11.825, 11.84], [21.0685, 21.0695],
                   [10.765, 10.7665], [9.1395, 9.1405],
                   [27.0864,  27.0874], [28.4652, 28.4665], [13.5445,  13.5473],
                   [9.9512,  9.9525], [28.6048,  28.6060]]


for time_range in time_range_list:
    plot_zoom_in_waveform(np.array(time_range))
    file_name = network_station + '_t_' + str(time_range[0]) + '_coincidence_' + str(threshold_coincidence) + '.pdf'
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.savefig(output_dir + '/' + file_name, bbox_inches='tight')

# plot Zooming figure with insets
for time_range in time_range_list:
    t_interval = time_range[1] - time_range[0]

    time_inset = [[time_range[0] + t_interval/6.2, time_range[1]-t_interval/1.7], [-0.1, 0.1]]
    plot_zoom_in_waveform(np.array(time_range), time_inset=time_inset)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    file_name = network_station + '_t_' + str(time_range[0]) + '_coincidence_' + str(threshold_coincidence) + '_insets.pdf'
    plt.savefig(output_dir + '/' + file_name, bbox_inches='tight')


# plot the histogram for detected earthquakes
# Get the event time from catalog
event_catalog_global = '/kuafu/yinjx/WaveDecompNet_dataset/continuous_waveforms/catalog.20210731-20210901.xml'
event_catalog_local = '/kuafu/yinjx/WaveDecompNet_dataset/continuous_waveforms/catalog_local.20210731-20210901.xml'

time0 = obspy.UTCDateTime('20210731')

def get_catalog_events(event_catalog_local, time0, M):
    events_local = obspy.read_events(event_catalog_local)
    events_local = events_local.filter(f'magnitude >= {M}')

    event_local_magnitude_list = [event.magnitudes[0].mag for event in events_local]
    event_local_time_in_day_list = [(event.origins[0].time - time0)/24/3600  for event in events_local]

    catalog_event_local_counts, catalog_event_local_day = np.histogram(event_local_time_in_day_list, bins=30, range=(1, 31))
    catalog_event_local_day = catalog_event_local_day[:-1] + np.diff(catalog_event_local_day)/2
    return catalog_event_local_counts,catalog_event_local_day

catalog_event_local_counts1, catalog_event_local_day = get_catalog_events(event_catalog_local, time0, 1)
catalog_event_local_counts2, catalog_event_local_day = get_catalog_events(event_catalog_local, time0, 2)

plt.figure(figsize=(8,4))
plt.hist(detect_time2 / second_per_day, bins=31, label='STA/LTA of decomp. waveform')
plt.hist(detect_time1 / second_per_day, bins=31, label='STA/LTA of raw waveform')
plt.plot(catalog_event_local_day, catalog_event_local_counts2, '-o', color='k', label='M2+ local events in catalog')
plt.legend()
#plt.ylim(0, 200)
plt.xlabel('Time (day)')
plt.ylabel('Number of STA/LTA triggers')
plt.title('Trigger events: Raw Data: ' + str(len(trig1)) + ' vs Separated Data: ' + str(len(trig2)))

plt.savefig(output_dir + '/' + 'stalta_triggers_coincidence_' + str(threshold_coincidence) + '.pdf',
            bbox_inches='tight')

# %%
