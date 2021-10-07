# %%
import os
import numpy as np
import datetime
from matplotlib import pyplot as plt

# import the Obspy modules that we will use in this exercise
import obspy
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

from utilities import downsample_series, mkdir
from torch_tools import WaveformDataset, try_gpu
import torch
from torch.utils.data import DataLoader

# %%
working_dir = os.getcwd()

# waveforms
waveform_dir = working_dir + '/continuous_waveforms'
#waveform_mseed = waveform_dir + '/' + 'IU.POHA.00.20210630-20210801.mseed'
waveform_mseed = waveform_dir + '/' + 'IU.POHA.00.20210731-20210901.mseed'
#waveform_mseed = waveform_dir + '/' + 'IU.POHA.10.20210731-20210901.mseed'

tr = obspy.read(waveform_mseed)
tr.merge(fill_value=0) # in case that there are segmented traces
#tr.filter('highpass', freq=1)

t1 = obspy.UTCDateTime("2021-07-19T12:07:00")
t2 = obspy.UTCDateTime("2021-07-19T12:10:00")
# t1 = obspy.UTCDateTime("2021-08-03T11:58:00")
# t2 = obspy.UTCDateTime("2021-08-03T12:03:00")
# tr.plot(starttime=t1, endtime=t2)

npts0 = tr[0].stats.npts # number of samples
dt0 = tr[0].stats.delta # dt

# event catalog
event_catalog = waveform_dir + '/' + 'catalog.20210731-20210901.xml'

# station information
station = obspy.read_inventory(waveform_dir + '/stations/IU.POHA.00.BH1.xml')
sta_lat = station[0][0].latitude
sta_lon = station[0][0].longitude

# read the catalog
events = obspy.read_events(event_catalog)
# estimate the arrival time of each earthquake to the station
t0 = tr[0].stats.starttime
event_arrival_P = np.zeros(len(events))
event_arrival_S = np.zeros(len(events))
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
        event_arrival_P[i_event] = event_time - t0 + P_arrival
        event_arrival_S[i_event] = event_time - t0 + S_arrival
    except:
        event_arrival_P[i_event] = np.nan
        event_arrival_S[i_event] = np.nan


# Reformat the waveform data into array
waveform0 = np.zeros((npts0, 3))
for i in range(3):
    waveform0[:, i] = tr[i].data

time0 = np.arange(0, npts0) * dt0

# TODO: Downsample the waveform data
f_downsample = 10
time, waveform, dt = downsample_series(time0, waveform0, f_downsample)

del time0, waveform0, tr


data_mean = np.mean(waveform, axis=0)
data_std = np.std(waveform, axis=0)
waveform_normalized = (waveform - data_mean) / (data_std + 1e-12)
waveform_normalized = np.reshape(waveform_normalized[:, np.newaxis, :], (-1, 600, 3))

# # TODO: Reformat the data into the format required by the model (batch, channel, samples)
# waveform = np.reshape(waveform[:, np.newaxis, :], (-1, 600, 3))
#
# # # TODO: Normalize the waveform first!
# data_mean = np.mean(waveform, axis=1, keepdims=True)
# data_std = np.std(waveform, axis=1, keepdims=True)
# waveform_normalized = (waveform - data_mean) / (data_std + 1e-12)

# TODO: Predict the separated waveforms
waveform_data = WaveformDataset(waveform_normalized, waveform_normalized)

# %% Need to specify model_name first
bottleneck_name = "LSTM"
#model_dataset_dir = "Model_and_datasets_1D_STEAD_plus_POHA"
model_dataset_dir = "Model_and_datasets_1D_STEAD2"
#model_dataset_dir = "Model_and_datasets_1D_synthetic"
model_name = "Autoencoder_Conv1D_" + bottleneck_name

model_dir = model_dataset_dir + f'/{model_name}'

# %% load model
model = torch.load(model_dir + '/' + f'{model_name}_Model.pth', map_location=try_gpu())

batch_size = 256
test_iter = DataLoader(waveform_data, batch_size=batch_size, shuffle=False)


# Test on real data
all_output = np.zeros(waveform_normalized.shape)
#all_output = np.zeros(waveform.shape)
model.eval()
for i, (X, _) in enumerate(test_iter):
    print('+' * 12 + f'batch {i}' + '+' * 12)
    output = model(X)
    output = output.detach().numpy()

    output = np.moveaxis(output, 1, -1)
    all_output[(i*batch_size) : ((i + 1)* batch_size), :, :] = output


# Check the waveform
waveform_recovered = all_output * data_std + data_mean
waveform_recovered = np.reshape(waveform_recovered, (-1, 3))

waveform_original = np.reshape(waveform, (-1, 3))
waveform_time = np.arange(waveform_original.shape[0]) / f_downsample

# scaled event magnitude (this is just a test)
scaled_magnitude = 10**event_magnitude/epi_distance/10

import h5py
waveform_output_dir = waveform_dir + '/' + model_dataset_dir
mkdir(waveform_output_dir)
with h5py.File(waveform_output_dir + '/' + bottleneck_name + '_processed_waveforms.hdf5', 'w') as f:
    f.create_dataset("waveform_time", data=waveform_time)
    f.create_dataset("waveform_original", data=waveform_original)
    f.create_dataset("waveform_recovered", data=waveform_recovered)

plt.close(1)
plt.figure(1, figsize=(18, 6))
for ii in range(3):
    plt.plot(waveform_time, waveform_original[:, ii]/np.max(abs(waveform_original[:, ii]))*5 + ii/2,
             color='gray', alpha=0.8, zorder=-2)
for ii in range(3):
    waveform_recovered_scaled = waveform_recovered[:, ii]/np.max(abs(waveform_original[:, ii]))*5
    plt.plot(waveform_time, waveform_recovered_scaled + ii/2, color='black', zorder=-1)
    waveform_recovered_scaled[abs(waveform_recovered_scaled - np.mean(waveform_recovered_scaled)) <5e-4] = np.nan
    plt.plot(waveform_time, waveform_recovered_scaled + ii/2, '-r', linewidth=1.5, zorder=0)
    plt.scatter(event_arrival_P, np.ones(len(event_arrival_P)) * ii/2, 50,
                'b', marker='+', linewidth=1.5, zorder=5)
    plt.scatter(event_arrival_S, np.ones(len(event_arrival_S)) * ii / 2, 50,
                'b', marker='x', linewidth=2, zorder=5)
    # plt.scatter(event_arrival_P, np.ones(len(event_arrival_P)) * ii / 2, scaled_magnitude/10,
    #            'b', marker='+', linewidth=2, zorder=5)
    # plt.scatter(event_arrival_S, np.ones(len(event_arrival_S)) * ii / 2, scaled_magnitude/10,
    #            'b', marker='x', linewidth=2, zorder=5)

plt.ylim(-1, 2)
plt.xlim(1018000, 1028000)
plt.xlabel('Time (s)')
plt.yticks([-1, 1], labels='')
plt.savefig(waveform_dir + '/continueous_separation_IU.POHA_' + bottleneck_name + '_STEAD_plus_POHA_t4.pdf')

plt.figure(2)
plt.plot(waveform_original[:, 0])
plt.plot(waveform_original[:, 0] - waveform_recovered[:, 0], alpha=1)



plt.plot(waveform_original[:, 0] / np.max(abs(waveform_original[:, 0])))
plt.plot(waveform_recovered[:, 0] / np.max(abs(waveform_recovered[:, 0])))

all_output0 = np.reshape(all_output, (-1, 3))
plt.plot(all_output0[:, 0])


temp = X
temp_out = model(X)
step = 20
for ii in range(0, 216, step):
    plt.plot(temp[ii, 0, :] + ii/step * 4, '-r')
    plt.plot(temp_out[ii, 0, :].detach().numpy() + ii/step * 4, '-b')

import h5py
# %% load dataset
data_dir = './training_datasets'
data_name = 'training_datasets_STEAD_waveform.hdf5'
#data_name = 'training_datasets_waveform.hdf5'

# %% load dataset
with h5py.File(data_dir + '/' + data_name, 'r') as f:
    time = f['time'][:]
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]