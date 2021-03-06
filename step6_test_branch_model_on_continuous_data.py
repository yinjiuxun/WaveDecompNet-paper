# %%
import os
import numpy as np
import datetime
from matplotlib import pyplot as plt
import h5py
import time as timing

# import the Obspy modules that we will use in this exercise
import obspy
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

from utilities import downsample_series, mkdir
from torch_tools import WaveformDataset, try_gpu
import torch
from torch.utils.data import DataLoader

import matplotlib

matplotlib.rcParams.update({'font.size': 18})

time_start_time = timing.time()  # time when code starts

# %%
working_dir = '/Users/yinjiuxun/Work/WaveDecompNet'

# waveforms
waveform_dir = '/kuafu/yinjx/WaveDecompNet_dataset/continuous_waveforms'
network_station = "IU.POHA" # "HV.HSSD" "IU.POHA" "HV.WRM" "HV.HAT" "HV.AIND" "HV.DEVL"
waveform_mseed = waveform_dir + '/' + 'IU.POHA.00.20210731-20210901.mseed'
# waveform_mseed = waveform_dir + '/HV_data_20210731-20210901/' + network_station + '.*.20210731-20210901.mseed'

tr = obspy.read(waveform_mseed)
tr.merge(fill_value=0)  # in case that there are segmented traces
# tr.filter('highpass', freq=0.1)
# f1=plt.figure(1, figsize=(8, 12))
# tr[0].plot(type='dayplot', interval=24*60, fig=f1, show_y_UTC_label=False, color=['k', 'r', 'b', 'g'])
# plt.savefig(waveform_dir + '/one_month_data_' + network_station + '.png')

# t1 = obspy.UTCDateTime("2021-07-19T12:07:00")
# t2 = obspy.UTCDateTime("2021-07-19T12:10:00")
# t1 = obspy.UTCDateTime("2021-08-03T11:58:00")
# t2 = obspy.UTCDateTime("2021-08-03T12:03:00")
# tr.plot(starttime=t1, endtime=t2)

time_load_trace_time = timing.time() - time_start_time  # time spent on loading data

npts0 = tr[0].stats.npts  # number of samples
dt0 = tr[0].stats.delta  # dt

# Reformat the waveform data into array
waveform0 = np.zeros((npts0, 3))
for i in range(3):
    waveform0[:, i] = tr[i].data

time0 = np.arange(0, npts0) * dt0

# Downsample the waveform data
f_downsample = 10
time, waveform, dt = downsample_series(time0, waveform0, f_downsample)

# del time0, waveform0, tr

# Reformat the data into the format required by the model (batch, channel, samples)
data_mean = np.mean(waveform, axis=0)
data_std = np.std(waveform, axis=0)
waveform_normalized = (waveform - data_mean) / (data_std + 1e-12)
waveform_normalized = np.reshape(waveform_normalized[:, np.newaxis, :], (-1, 600, 3))

# # Reformat the data into the format required by the model (batch, channel, samples)
# # For individual batch
# waveform = np.reshape(waveform[:, np.newaxis, :], (-1, 600, 3))
#
# #Normalize the waveform first!
# data_mean = np.mean(waveform, axis=1, keepdims=True)
# data_std = np.std(waveform, axis=1, keepdims=True)
# waveform_normalized = (waveform - data_mean) / (data_std + 1e-12)

# Predict the separated waveforms
waveform_data = WaveformDataset(waveform_normalized, waveform_normalized)

# time spent on downsample and normalization data
time_process_trace_time = timing.time() - time_start_time - time_load_trace_time

# %% Need to specify model_name first
bottleneck_name = "LSTM" # LSTM, attention
#model_dataset_dir = "Model_and_datasets_1D_STEAD_plus_POHA"
#model_dataset_dir = "Model_and_datasets_1D_STEAD2"
# model_dataset_dir = "Model_and_datasets_1D_all_snr_40"
# model_dataset_dir = "Model_and_datasets_1D_all_snr_40_unshuffled"
model_dataset_dir = "Model_and_datasets_1D_all_snr_40_unshuffled_equal_epoch100"
# model_dataset_dir = "Model_and_datasets_1D_synthetic"
model_name = "Branch_Encoder_Decoder_" + bottleneck_name

model_dir = model_dataset_dir + f'/{model_name}'

# %% load model
model = torch.load(model_dir + '/' + f'{model_name}_Model.pth', map_location='cpu')#try_gpu())

batch_size = 256
test_iter = DataLoader(waveform_data, batch_size=batch_size, shuffle=False)

# Test on real data
all_output1 = np.zeros(waveform_normalized.shape)  # signal
all_output2 = np.zeros(waveform_normalized.shape)  # noise
# all_output = np.zeros(waveform.shape)
model.eval()
for i, (X, _) in enumerate(test_iter):
    print('+' * 12 + f'batch {i}' + '+' * 12)
    output1, output2 = model(X)

    # output1 corresponds to earthquake signal
    output1 = output1.detach().numpy()
    output1 = np.moveaxis(output1, 1, -1)
    all_output1[(i * batch_size): ((i + 1) * batch_size), :, :] = output1

    # output2 corresponds to ambient noise
    output2 = output2.detach().numpy()
    output2 = np.moveaxis(output2, 1, -1)
    all_output2[(i * batch_size): ((i + 1) * batch_size), :, :] = output2

# Check the waveform
waveform_recovered = all_output1 * data_std + data_mean
waveform_recovered = np.reshape(waveform_recovered, (-1, 3))

noise_recovered = all_output2 * data_std + data_mean
noise_recovered = np.reshape(noise_recovered, (-1, 3))

waveform_original = np.reshape(waveform, (-1, 3))
waveform_time = np.arange(waveform_original.shape[0]) / f_downsample

time_decompose_waveform = timing.time() - time_start_time - time_load_trace_time - time_process_trace_time

print('Time spent on decomposing seismograms: ')
print(f'Load one-month mseed data: {time_load_trace_time:.3f} sec\n' +
      f'Process data (downsample, normalization, reshape): {time_process_trace_time:.3f} sec\n' +
      f'Decompose into earthquake and noise: {time_decompose_waveform:.3f} sec\n')

## event catalog
event_catalog = waveform_dir + '/' + 'catalog.20210731-20210901.xml'
local_event_catalog = waveform_dir + '/' + 'catalog_local.20210731-20210901.xml'

# station information
station = obspy.read_inventory(waveform_dir + '/stations/IU.POHA.00.BH1.xml')
# station = obspy.read_inventory(waveform_dir + '/stations/HV.HAT.*.HHE.xml')
sta_lat = station[0][0].latitude
sta_lon = station[0][0].longitude

# read the global catalog
events0 = obspy.read_events(event_catalog)
# this is to show the large earthquake occur
events = events0.filter("time > 2021-08-11T12:00:00", "time < 2021-08-17T12:00:00", "magnitude >= 5")
#events.plot(projection='local', water_fill_color='#A7D7EF', continent_fill_color='#F4F2DD')
# estimate the arrival time of each earthquake to the station
t0 = tr[0].stats.starttime
event_arrival_P = np.zeros(len(events))
event_arrival_S = np.zeros(len(events))
event_time_P = []
epi_distance = np.zeros(len(events))
event_magnitude = np.array([event.magnitudes[0].mag for event in events])
for i_event in range(len(events)):
    event = events[i_event]
    # print(event)
    # print(event.origins[0])
    # % % extract the event information
    event_time = event.origins[0].time
    event_lon = event.origins[0].longitude
    event_lat = event.origins[0].latitude
    event_dep = event.origins[0].depth / 1e3

    # % % estimate the distance and the P arrival time from the event to the station
    try:
        distance_to_source = locations2degrees(sta_lat, sta_lon, event_lat, event_lon)
        epi_distance[i_event] = distance_to_source
        model = TauPyModel(model='iasp91')

        arrivals = model.get_ray_paths(event_dep, distance_to_source, phase_list=['P'])
        P_arrival = arrivals[0].time
        arrivals = model.get_ray_paths(event_dep, distance_to_source, phase_list=['S'])
        S_arrival = arrivals[0].time
        # the relative arrival time on the waveform when the event signal arrives
        event_info = {"time": event_time + P_arrival, "text": []}  # str(event.magnitudes[0].mag)
        event_time_P.append(event_info)
        event_arrival_P[i_event] = event_time - t0 + P_arrival
        event_arrival_S[i_event] = event_time - t0 + S_arrival
    except:
        event_arrival_P[i_event] = np.nan
        event_arrival_S[i_event] = np.nan

# read the local catalog (< 5 degree)
events0 = obspy.read_events(local_event_catalog)
# this is to show the large earthquake occur
events = events0.filter("magnitude >= 2.5")
#events.plot(projection='local', water_fill_color='#A7D7EF', continent_fill_color='#F4F2DD')
# estimate the arrival time of each earthquake to the station
t0 = tr[0].stats.starttime
event_arrival_P_local = np.zeros(len(events))
event_arrival_S_local = np.zeros(len(events))
event_time_P = []
epi_distance = np.zeros(len(events))
event_magnitude = np.array([event.magnitudes[0].mag for event in events])
for i_event in range(len(events)):
    event = events[i_event]
    # print(event)
    # print(event.origins[0])
    # % % extract the event information
    event_time = event.origins[0].time
    event_lon = event.origins[0].longitude
    event_lat = event.origins[0].latitude
    event_dep = event.origins[0].depth / 1e3

    # % % estimate the distance and the P arrival time from the event to the station
    try:
        distance_to_source = locations2degrees(sta_lat, sta_lon, event_lat, event_lon)
        epi_distance[i_event] = distance_to_source
        model = TauPyModel(model='iasp91')

        arrivals = model.get_ray_paths(event_dep, distance_to_source, phase_list=['P'])
        P_arrival = arrivals[0].time
        arrivals = model.get_ray_paths(event_dep, distance_to_source, phase_list=['s'])
        S_arrival = arrivals[0].time
        # the relative arrival time on the waveform when the event signal arrives
        event_info = {"time": event_time + P_arrival, "text": []}  # str(event.magnitudes[0].mag)
        event_time_P.append(event_info)
        event_arrival_P_local[i_event] = event_time - t0 + P_arrival
        event_arrival_S_local[i_event] = event_time - t0 + S_arrival
    except:
        event_arrival_P_local[i_event] = np.nan
        event_arrival_S_local[i_event] = np.nan

# output decomposed waveforms
waveform_output_dir = waveform_dir + '/' + model_dataset_dir
mkdir(waveform_output_dir)
waveform_output_dir = waveform_output_dir + '/' + network_station
mkdir(waveform_output_dir)
waveform_output_dir = waveform_output_dir + '/' + bottleneck_name
mkdir(waveform_output_dir)

with h5py.File(waveform_output_dir + '/' + bottleneck_name + '_processed_waveforms.hdf5', 'w') as f:
    f.create_dataset("waveform_time", data=waveform_time)
    f.create_dataset("waveform_original", data=waveform_original)
    f.create_dataset("waveform_recovered", data=waveform_recovered)
    f.create_dataset("noise_recovered", data=noise_recovered)
    f.create_dataset("event_arrival_P", data=event_arrival_P)  # far field events
    f.create_dataset("event_arrival_S", data=event_arrival_S)  # far field events
    f.create_dataset("event_arrival_P_local", data=event_arrival_P_local)  # local event
    f.create_dataset("event_arrival_S_local", data=event_arrival_S_local)  # local event

np.save(waveform_output_dir + '/M5.5_earthquakes.npy', event_time_P)

# Also output the mseed format of separated waveform for further testing
tr_raw = tr.copy()
tr_earthquake = tr.copy()
tr_noise = tr.copy()
tr_residual = tr.copy()

for i_chan in range(3):
    tr_raw[i_chan].data = waveform_original[:, i_chan]
    tr_raw[i_chan].stats.sampling_rate = f_downsample

    tr_earthquake[i_chan].data = waveform_recovered[:, i_chan]
    tr_earthquake[i_chan].stats.sampling_rate = f_downsample

    tr_noise[i_chan].data = noise_recovered[:, i_chan]
    tr_noise[i_chan].stats.sampling_rate = f_downsample

    tr_residual[i_chan].data = waveform_original[:, i_chan] - waveform_recovered[:, i_chan] - noise_recovered[:, i_chan]
    tr_residual[i_chan].stats.sampling_rate = f_downsample

tr_earthquake.write(waveform_output_dir + '/' + network_station + '.00.20210731-20210901_separated_earthquake.mseed')
tr_raw.write(waveform_output_dir + '/' + network_station + '.00.20210731-20210901_original_earthquake.mseed')

#%%
############################ Make figures ###############################################
# waveforms
waveform_dir = '/kuafu/yinjx/WaveDecompNet_dataset/continuous_waveforms'
network_station = "IU.POHA" # "HV.HSSD" "IU.POHA" "HV.WRM" "HV.HAT" "HV.AIND" "HV.DEVL"

waveform_mseed = waveform_dir + '/' + 'IU.POHA.00.20210731-20210901.mseed'
# waveform_mseed = waveform_dir + '/' + 'IU.POHA.10.20210731-20210901.mseed'
# waveform_mseed = waveform_dir + '/HV_data_20210731-20210901/' + network_station + '.*.20210731-20210901.mseed'
tr = obspy.read(waveform_mseed)
tr.merge(fill_value=0)  # in case that there are segmented traces
# tr.filter('highpass', freq=0.1)
tr.decimate(4)

# Model names and path
bottleneck_name = "LSTM"
#model_dataset_dir = "Model_and_datasets_1D_STEAD_plus_POHA"
#model_dataset_dir = "Model_and_datasets_1D_STEAD2"
model_dataset_dir = "Model_and_datasets_1D_all_snr_40_unshuffled_equal_epoch100"
# model_dataset_dir = "Model_and_datasets_1D_synthetic"
model_name = "Branch_Encoder_Decoder_" + bottleneck_name

# Load the recovered waveforms and then make plots
waveform_output_dir = waveform_dir + '/' + model_dataset_dir + '/' + network_station + '/' + bottleneck_name
with h5py.File(waveform_output_dir + '/' + bottleneck_name + '_processed_waveforms.hdf5', 'r') as f:
    waveform_time = f["waveform_time"][:]
    waveform_original = f["waveform_original"][:]
    waveform_recovered = f["waveform_recovered"][:]
    noise_recovered = f["noise_recovered"][:]
    event_arrival_P = f["event_arrival_P"][:]
    event_arrival_S = f["event_arrival_S"][:]
    event_arrival_P_local = f["event_arrival_P_local"][:]
    event_arrival_S_local = f["event_arrival_S_local"][:]

event_time_P = np.load(waveform_output_dir + '/M5.5_earthquakes.npy', allow_pickle=True)
event_time_P = event_time_P.tolist()

dt = waveform_time[1] - waveform_time[0]

waveform_output_dir = waveform_output_dir  # + '/prefiltered'
mkdir(waveform_output_dir)

# Downsample the waveforms ONLY for plotting purpose
waveform_time = waveform_time[::10]
waveform_original = waveform_original[::10, :]
waveform_recovered = waveform_recovered[::10, :]
noise_recovered = noise_recovered[::10, :]
waveform_residual = waveform_original - waveform_recovered - noise_recovered

# Write the controllable way to plot waveforms
# Scale the waveforms with the same value
waveform_original_S = waveform_original / np.amax(abs(waveform_original))
waveform_recovered_S = waveform_recovered / np.amax(abs(waveform_original))
noise_recovered_S = noise_recovered / np.amax(abs(waveform_original))
waveform_residual_S = waveform_residual / np.amax(abs(waveform_original))

def arange_month_time(waveform_time):
    """A small function to arange the month time into days and hours"""
    waveform_time_in_hours = waveform_time / 3600
    waveform_day = waveform_time_in_hours // 24  # day in the month
    waveform_day = waveform_day.astype('int')
    waveform_hours = waveform_time_in_hours % 24  # time in the day
    return waveform_day, waveform_hours



def plot_month_waveform(ax, waveform_time, waveform, component=0, color='k', title=None,
                        event_arrival_tele=None, event_arrival_local=None, amplitude_scale=10):

    waveform_day, waveform_hours = arange_month_time(waveform_time)

    if event_arrival_tele is not None:
        tele_event_arrival_day, tele_event_arrival_hours = arange_month_time(event_arrival_tele)

    if event_arrival_local is not None:
        local_event_arrival_day, local_event_arrival_hours = arange_month_time(event_arrival_local)

    for ii0 in range(np.ceil(waveform_time[-1] / 24 / 3600).astype('int')):

        i_time = waveform_day == ii0
        ax.plot(waveform_hours[i_time], np.squeeze(waveform[i_time, component] * amplitude_scale) + ii0,
                '-', linewidth=1, color=color)

        if event_arrival_tele is not None:
            i_time_event = np.where(tele_event_arrival_day == ii0)
            ax.plot(tele_event_arrival_hours[i_time_event], ii0*np.ones(i_time_event[0].shape), '*', markerfacecolor='m',
                    linewidth=1, markersize=13, markeredgecolor='k')

        if event_arrival_local is not None:
            i_time_event = np.where(local_event_arrival_day == ii0)
            ax.plot(local_event_arrival_hours[i_time_event], ii0 * np.ones(i_time_event[0].shape), 's', color='green',
                    linewidth=1, markersize=12, markerfacecolor='None')
    ax.grid()
    ax.set_title(title)

component_str = ['E', 'N', 'Z']
amplitude_scale = 14
for component in range(3):
    plt.close('all')
    fig, ax = plt.subplots(2, 2, figsize=(14, 20), sharex=True, sharey=True)
    ax = ax.flatten()

    plot_month_waveform(ax[0], waveform_time, waveform_original_S,
                        event_arrival_tele=event_arrival_P, event_arrival_local=event_arrival_P_local,
                        component=component, color='k', amplitude_scale=amplitude_scale,
                        title='(a) Raw waveform (' + component_str[component] + ')')
    ax[0].set_ylabel('Date')

    plot_month_waveform(ax[1], waveform_time, waveform_recovered_S,
                        event_arrival_tele=event_arrival_P, event_arrival_local=event_arrival_P_local,
                        component=component, color='r', amplitude_scale=amplitude_scale,
                        title='(b) Earthquake waveform (' + component_str[component] + ')')

    plot_month_waveform(ax[2], waveform_time, noise_recovered_S, event_arrival_tele=event_arrival_P, component=component,
                        color='b', amplitude_scale=amplitude_scale,
                        title='(c) Noise waveform (' + component_str[component] + ')')
    ax[2].set_ylabel('Date')
    ax[2].set_xlabel('Time in hour')

    plot_month_waveform(ax[3], waveform_time, waveform_residual_S, event_arrival_tele=event_arrival_P, component=component,
                        color='gray', amplitude_scale=amplitude_scale,
                        title='(d) Residual waveform (' + component_str[component] + ')')
    ax[3].set_xlabel('Time in hour')

    ax[0].invert_yaxis()
    ax[0].set_xlim(0, 24)
    ax[0].set_xticks(np.arange(0, 25, 4))
    ax[0].set_yticks(np.arange(0, 34, 7))
    ax[0].set_yticklabels(['07-31', '08-07', '08-14', '08-21', '08-28'])
    plt.tight_layout()

    plt.savefig(waveform_output_dir + f'/one_month_data_all_{component_str[component]}.pdf')


# Plot zoom-in waveforms
waveform_time_day = waveform_time / 24 / 3600
plt.close('all')
plt.figure(1, figsize=(18, 6))
for ii in range(3):
    plt.plot(waveform_time_day, waveform_original[:, ii] / np.max(abs(waveform_original[:, ii])) * 5 + ii / 2,
             color='gray', alpha=0.8, zorder=-2)
for ii in range(3):
    waveform_recovered_scaled = waveform_recovered[:, ii] / np.max(abs(waveform_original[:, ii])) * 5
    plt.plot(waveform_time_day, waveform_recovered_scaled + ii / 2, color='black', zorder=-1)
    waveform_recovered_scaled[abs(waveform_recovered_scaled - np.mean(waveform_recovered_scaled)) < 5e-4] = np.nan
    plt.plot(waveform_time_day, waveform_recovered_scaled + ii / 2, '-r', linewidth=1.5, zorder=0)
    plt.scatter(event_arrival_P / 24 / 3600, np.ones(len(event_arrival_P)) * ii / 2, 50,
                'b', marker='+', linewidth=1.5, zorder=5)
    plt.scatter(event_arrival_S / 24 / 3600, np.ones(len(event_arrival_S)) * ii / 2, 50,
                'b', marker='x', linewidth=2, zorder=5)
    # plt.scatter(event_arrival_P, np.ones(len(event_arrival_P)) * ii / 2, scaled_magnitude/10,
    #            'b', marker='+', linewidth=2, zorder=5)
    # plt.scatter(event_arrival_S, np.ones(len(event_arrival_S)) * ii / 2, scaled_magnitude/10,
    #            'b', marker='x', linewidth=2, zorder=5)

plt.ylim(-3.3, 4)
plt.xlabel('Time (s)')
plt.yticks([-1, 1], labels='')
time_zoom_in = [(10.8, 10.9), (13.53, 13.6), (12.8, 12.9), (17.45, 17.55)]
for i, xlimit in enumerate(time_zoom_in):
    plt.xlim(xlimit)
    # plt.savefig(waveform_output_dir + '/continueous_separation_IU.POHA_' + bottleneck_name + '_t' + str(i) + '.pdf')
    plt.savefig(waveform_output_dir + '/continueous_separation_' + network_station + '_' + bottleneck_name + '_t' + str(
        i) + '.png')

#
# # Write the processed waveforms to the new trace objects (to use Obspy plot functions)
# tr_recovered = tr.copy()
# tr_noise = tr.copy()
# tr_residual = tr.copy()
#
# for i in range(3):
#     tr_recovered[i].stats.sampling_rate = 1 / dt
#     tr_noise[i].stats.sampling_rate = 1 / dt
#     tr_residual[i].stats.sampling_rate = 1 / dt
#
#     tr_recovered[i].data = waveform_recovered[:, i]
#     tr_noise[i].data = noise_recovered[:, i]
#     tr_residual[i].data = waveform_original[:, i] - waveform_recovered[:, i] - noise_recovered[:, i]
#
# # Visualize the data in one-month
# plt.close('all')
# i_channel = 0
# vertical_scaling = 35000
# # The original data
# f1 = plt.figure(1, figsize=(8, 10))
# tr[i_channel].plot(type='dayplot', interval=24 * 60, vertical_scaling_range=vertical_scaling, tick_format='%m-%d',
#                    fig=f1, show_y_UTC_label=False, color=['k'], title='', x_labels_size=18, events=event_time_P)
# plt.yticks(fontsize=18)
# plt.ylabel("Days")
# plt.xlabel('Time in hours', fontsize=18)
# plt.title('(a) Raw waveform (' + network_station + ')')
# plt.savefig(waveform_output_dir + '/one_month_data_original_BH' + str(i_channel) + '.pdf', bbox_inches='tight')
#
# # The separated earthquake data
# vertical_scaling = 35000
# f2 = plt.figure(2, figsize=(8, 10))
# tr_recovered[i_channel].plot(type='dayplot', interval=24 * 60, vertical_scaling_range=vertical_scaling,
#                              tick_format='%m-%d',
#                              fig=f2, show_y_UTC_label=False, color=['r'], title='', x_labels_size=18,
#                              events=event_time_P)
# plt.yticks(fontsize=18)
# plt.ylabel("Days")
# plt.xlabel('Time in hours', fontsize=18)
# plt.title('(b) Earthquake waveform (' + network_station + ')')
# plt.savefig(waveform_output_dir + '/one_month_data_earthquake_BH' + str(i_channel) + '.pdf', bbox_inches='tight')
#
# # The separated noise data
# vertical_scaling = 35000
# f3 = plt.figure(3, figsize=(8, 10))
# tr_noise[i_channel].plot(type='dayplot', interval=24 * 60, vertical_scaling_range=vertical_scaling, tick_format='%m-%d',
#                          fig=f3, show_y_UTC_label=False, color=['b'], title='', x_labels_size=18, events=event_time_P)
# plt.yticks(fontsize=18)
# plt.ylabel("Days")
# plt.xlabel('Time in hours', fontsize=18)
# plt.title('(c) Noise waveform (' + network_station + ')')
# plt.savefig(waveform_output_dir + '/one_month_data_noise_BH' + str(i_channel) + '.pdf', bbox_inches='tight')
#
# # The residual
# # The separated noise data
# vertical_scaling = 70000
# f4 = plt.figure(4, figsize=(8, 10))
# tr_residual[i_channel].plot(type='dayplot', interval=24 * 60, vertical_scaling_range=vertical_scaling,
#                             tick_format='%m-%d',
#                             fig=f4, show_y_UTC_label=False, color=['gray'], title='', x_labels_size=18,
#                             events=event_time_P)
# plt.yticks(fontsize=18)
# plt.ylabel("Days")
# plt.xlabel('Time in hours', fontsize=18)
# plt.title('(d) Residual waveform (' + network_station + ')')
# plt.savefig(waveform_output_dir + '/one_month_data_residual_BH' + str(i_channel) + '.pdf', bbox_inches='tight')



# # Write the controllable way to plot waveforms
# waveform_time = waveform_time[::10]
# waveform_original = waveform_original[::10, :]
# waveform_recovered = waveform_recovered[::10, :]
# noise_recovered = noise_recovered[::10, :]
# waveform_residual = waveform_original - waveform_recovered - noise_recovered
#
# waveform_time_in_hours = waveform_time / 3600
# waveform_day = waveform_time_in_hours // 24  # day in the month
# waveform_hours = waveform_time_in_hours % 24  # time in the day
#
# waveform_original_S = waveform_original / np.amax(abs(waveform_original))
# waveform_recovered_S = waveform_recovered / np.amax(abs(waveform_original))
# noise_recovered_S = noise_recovered / np.amax(abs(waveform_original))
# waveform_residual_S = waveform_residual / np.amax(abs(waveform_original))
#
# fig, ax = plt.subplots(figsize=(7, 10))
#
# for ii in range(32):
#     ii_wave = np.where(waveform_day.astype('int') == ii)
#     ax.plot(waveform_hours[ii_wave], np.squeeze(waveform_original_S[ii_wave, 0] * 10) + ii, '-k', linewidth=1)
#
# ax.invert_yaxis()
# ax.set_xlim(0, 24)
# ax.set_xticks(np.arange(0, 25, 2))
# ax.set_yticks(np.arange(0, 34, 7))
# ax.set_yticklabels(['07-31', '08-07', '08-14', '08-21', '08-28'])
#
# fig, ax = plt.subplots(figsize=(10, 10))
# for ii in range(32):
#     ii_wave = np.where(waveform_day.astype('int') == ii)
#     ax.plot(waveform_hours[ii_wave], np.squeeze(noise_recovered_S[ii_wave, 0] * 10) + ii, '-b', linewidth=1)
#     ax.plot(waveform_hours[ii_wave], np.squeeze(waveform_residual_S[ii_wave, 0] * 10) + ii, '-', color='gray',
#             linewidth=1)
#
# ax.invert_yaxis()
# ax.set_xlim(0, 24)
# ax.set_xticks(np.arange(0, 25, 2))
# ax.set_yticks(np.arange(0, 34, 7))
# ax.set_yticklabels(['07-31', '08-07', '08-14', '08-21', '08-28'])
#
#
# def plot_month_waveform(ax, waveform_time, waveform, component=0, color='k', title=None, event_arrival=None):
#     waveform_time_in_hours = waveform_time / 3600
#     waveform_day = waveform_time_in_hours // 24  # day in the month
#     waveform_day = waveform_day.astype('int')
#     waveform_hours = waveform_time_in_hours % 24  # time in the day
#
#     if event_arrival is not None:
#         event_arrival_in_hours = event_arrival / 3600
#         event_arrival_day = event_arrival_in_hours // 24  # day in the month
#         event_arrival_day = event_arrival_day.astype('int')
#         event_arrival_hours = event_arrival_in_hours % 24  # time in the day
#
#     for ii0 in range(np.ceil(waveform_time[-1] / 24 / 3600).astype('int')):
#
#         i_time = waveform_day == ii0
#         ax.plot(waveform_hours[i_time], np.squeeze(waveform[i_time, component] * 10) + ii0,
#                 '-', linewidth=1, color=color)
#
#         if event_arrival is not None:
#             i_time_event = np.where(event_arrival_day == ii0)
#             ax.plot(event_arrival_hours[i_time_event], ii0*np.ones(i_time_event[0].shape), '*', markerfacecolor='gold',
#                     linewidth=1, markersize=13, markeredgecolor='k')
#
#     ax.set_title(title)
#
#
# plt.close('all')
# fig, ax = plt.subplots(2, 2, figsize=(14, 20), sharex=True, sharey=True)
# ax = ax.flatten()
#
# plot_month_waveform(ax[0], waveform_time, waveform_original_S, event_arrival=event_arrival_P, color='k', title='(a) Raw waveform')
# ax[0].set_ylabel('Date')
#
# plot_month_waveform(ax[1], waveform_time, waveform_recovered_S, event_arrival=event_arrival_P, color='r', title='(b) Earthquake waveform')
#
# plot_month_waveform(ax[2], waveform_time, noise_recovered_S, event_arrival=event_arrival_P, color='b', title='(c) Noise waveform')
# ax[2].set_ylabel('Date')
# ax[2].set_xlabel('Time in hour')
#
# plot_month_waveform(ax[3], waveform_time, waveform_residual_S, event_arrival=event_arrival_P, color='gray', title='(d) Residual waveform')
# ax[3].set_xlabel('Time in hour')
#
# ax[0].invert_yaxis()
# ax[0].set_xlim(0, 24)
# ax[0].set_xticks(np.arange(0, 25, 2))
# ax[0].set_yticks(np.arange(0, 34, 7))
# ax[0].set_yticklabels(['07-31', '08-07', '08-14', '08-21', '08-28'])
# plt.tight_layout()
#
# plt.savefig(working_dir + '/continuous_waveforms_all.pdf')

# %%
