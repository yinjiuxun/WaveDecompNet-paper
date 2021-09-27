# %%
import os
import numpy as np
import datetime
from matplotlib import pyplot as plt

# import the Obspy modules that we will use in this exercise
import obspy

from utilities import downsample_series
from torch_tools import WaveformDataset, try_gpu
import torch
from torch.utils.data import DataLoader

# %%
working_dir = os.getcwd()

waveform_dir = working_dir + '/continuous_waveforms'
waveform_mseed = waveform_dir + '/' + 'IU.POHA.00.20210731-20210901.mseed'

tr = obspy.read(waveform_mseed)

tr.filter('highpass', freq=1)

t1 = obspy.UTCDateTime("2021-08-03T12:07:00")
t2 = obspy.UTCDateTime("2021-08-03T12:10:00")
# t1 = obspy.UTCDateTime("2021-08-03T11:58:00")
# t2 = obspy.UTCDateTime("2021-08-03T12:03:00")
# tr.plot(starttime=t1, endtime=t2)

npts0 = tr[0].stats.npts # number of samples
dt0 = tr[0].stats.delta # dt

# Reformat the waveform data into array
waveform0 = np.zeros((npts0, 3))
for i in range(3):
    waveform0[:, i] = tr[i].data

time0 = np.arange(0, npts0) * dt0

# TODO: Downsample the waveform data
f_downsample = 10
time, waveform, dt = downsample_series(time0, waveform0, f_downsample)

del time0, waveform0, tr


# data_mean = np.mean(waveform, axis=0)
# data_std = np.std(waveform - data_mean, axis=0)
# waveform_normalized = (waveform - data_mean) / (data_std + 1e-12)
# waveform_normalized = np.reshape(waveform_normalized[:, np.newaxis, :], (-1, 600, 3))

# TODO: Reformat the data into the format required by the model (batch, channel, samples)
waveform = np.reshape(waveform[:, np.newaxis, :], (-1, 600, 3))

# # TODO: Normalize the waveform first!
data_mean = np.mean(waveform, axis=1, keepdims=True)
data_std = np.std(waveform - data_mean, axis=1, keepdims=True)
waveform_normalized = (waveform - data_mean) / (data_std + 1e-12)

# TODO: Predict the separated waveforms
waveform_data = WaveformDataset(waveform_normalized, waveform_normalized)

# %% Need to specify model_name first
bottleneck_name = "LSTM"
model_dataset_dir = "Model_and_datasets_1D_STEAD2_test"
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
waveform_recovered = all_output
waveform_recovered = np.reshape(waveform_recovered, (-1, 3))

waveform_original = np.reshape(waveform, (-1, 3))


plt.plot(waveform_original[:, 0])
plt.plot(waveform_recovered[:, 0])

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