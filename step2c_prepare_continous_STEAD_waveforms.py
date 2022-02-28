# %%
import os
import numpy as np
import datetime
from matplotlib import pyplot as plt

# import the Obspy modules that we will use in this exercise
import obspy

from utilities import downsample_series, mkdir
from torch_tools import WaveformDataset, try_gpu
import torch
from torch.utils.data import DataLoader

# # %%
# working_dir = os.getcwd()
#
# waveform_dir = working_dir + '/continuous_waveforms'
# waveform_mseed = waveform_dir + '/' + 'IU.POHA.00.20210731-20210901.mseed'
#
# tr = obspy.read(waveform_mseed)
#
# #tr.filter('highpass', freq=1)
#
# npts0 = tr[0].stats.npts # number of samples
# dt0 = tr[0].stats.delta # dt
#
# # Reformat the waveform data into array
# waveform0 = np.zeros((npts0, 3))
# for i in range(3):
#     waveform0[:, i] = tr[i].data
#
# time0 = np.arange(0, npts0) * dt0
#
# # Downsample the waveform data
# f_downsample = 10
# time, waveform, dt = downsample_series(time0, waveform0, f_downsample)
#
# del time0, waveform0, tr

# Here instead of shuffling noise, direct use noise waveform in the previous months
working_dir = os.getcwd()

waveform_dir = working_dir + '/continuous_waveforms'
waveform_mseed = waveform_dir + '/' + 'IU.POHA.00.20210601-20210801.mseed'
shuffle_phase = False

tr = obspy.read(waveform_mseed)
tr.decimate(4)  # downsample from 40 to 10 Hz
tr.merge(fill_value=np.nan)

# Extract the waveform data
waveform0 = np.zeros((tr[0].stats.npts, 3))

for i in range(3):
    waveform0[:, i] = np.array(tr[i].data)

# Use the median amplitude as the threshold to separate noise and earthquake waveforms
amplitude_threshold = 6
waveform0_amplitude = np.sqrt(np.sum(waveform0 ** 2, axis=1))
amplitude_median = np.nanmedian(abs(waveform0_amplitude))

waveform = waveform0[waveform0_amplitude < (amplitude_threshold * amplitude_median), :]

# keep the right number of samples (integral times of 600) and discard the remainder
n_noise_samples = waveform.shape[0] // 600 * 600
waveform = waveform[:n_noise_samples, :]

# # Used to quickly check the noise waveforms
# fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
# for i, axi in enumerate(ax):
#     axi.plot(waveform0[-18000:-1, i])
#     axi.plot(waveform[-18000:-1, i])

# reformat waveform for random shuffle
waveform = waveform[np.newaxis, :, :]

# Randomly shuffle the waveform data
import scipy


def randomization_noise(noise, rng=np.random.default_rng(None)):
    """function to produce randomized noise by shiftint the phase
    randomization_noise(noise, rng=np.random.default_rng(None))
    return randomized noise
    The input noise has to be an 3D array with (num_batch, num_time, num_channel)
    """

    s = scipy.fft.fft(noise, axis=1)
    phase_angle_shift = (rng.random(s.shape) - 0.5) * 2 * np.pi
    # make sure the inverse transform is real
    phase_angle_shift[:, 0, :] = 0
    phase_angle_shift[:, int(s.shape[1] / 2 + 1):, :] = -1 * \
                                                        np.flip(phase_angle_shift[:, 1:int(s.shape[1] / 2), :])

    phase_shift = np.exp(np.ones(s.shape) * phase_angle_shift * 1j)

    # Here apply the phase shift in the entire domain
    # s_shifted = np.abs(s) * phase_shift

    # Instead, try only shift frequency below 10Hz
    freq = scipy.fft.fftfreq(s.shape[1], dt)
    ii_freq = abs(freq) <= 10
    s_shifted = s.copy()
    s_shifted[:, ii_freq, :] = np.abs(s[:, ii_freq, :]) * phase_shift[:, ii_freq, :]

    noise_random = np.real(scipy.fft.ifft(s_shifted, axis=1))
    return noise_random


if shuffle_phase:
    noise = randomization_noise(waveform)
else:
    noise = waveform

# TODO: Reformat the data into the format required by the model (batch, channel, samples)
waveform = np.reshape(waveform, (-1, 600, 3))
noise = np.reshape(noise, (-1, 600, 3))

# now the noise shuffled from the continuous data has been obtained
# Stack the STEAD waveform
import pandas as pd
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

file_name = "/Users/Yin9xun/Work/STEAD/merged.hdf5"
csv_file = "/Users/Yin9xun/Work/STEAD/merged.csv"

# reading the csv file into a dataframe:
df = pd.read_csv(csv_file)
print(f'total events in csv file: {len(df)}')

# Some work on the snr_db list
snr_db0 = df.snr_db.to_list()
mean_snr_db = np.zeros((len(snr_db0), 1))
for i in range(len(snr_db0)):
    try:
        if np.isnan(snr_db0[i]):
            mean_snr_db[i] = snr_db0[i]
    except TypeError:
        temp = snr_db0[i].replace('[', '')
        temp = temp.replace(']', '')
        temp = temp.split()
        mean_snr_db[i] = np.mean([float(temp[j]) for j in range(len(temp))])
mean_snr_db = mean_snr_db.squeeze()

# filterering the dataframe
df_earthquakes = df[(df.trace_category == 'earthquake_local') & (mean_snr_db >= 40)]
print(f'total number of earthquakes: {len(df_earthquakes)}')

# Set the number of waveforms that will be used during the training
N_events = noise.shape[0]
earthquake_seed = 111
noise_seed = 112
shift_seed = 113
snr_seed = 114

# Prepare the random list to get the events and noise series
from numpy.random import default_rng

rng_earthquake = default_rng(earthquake_seed)
chosen_earthquake_index = rng_earthquake.choice(len(df_earthquakes), N_events)

rng_shift = default_rng(shift_seed)
shift = rng_shift.uniform(-30, 60, N_events)

rng_snr = default_rng(snr_seed)
snr = 10 ** rng_snr.uniform(-1, 2, N_events)

# Choose the earthquakes and noise from STEAD
df_earthquakes = df_earthquakes.iloc[chosen_earthquake_index]
earthquake_list = df_earthquakes['trace_name'].to_list()

# make the output directory
training_dataset_dir = './training_datasets'
mkdir(training_dataset_dir)
model_datasets = training_dataset_dir + '/training_datasets_STEAD_plus_POHA_snr_40_unshuffled.hdf5'

# Loop over each pair
f_downsample = 10
time = np.arange(0, 6000) * 0.01
dtfl = h5py.File(file_name, 'r')

X_train, Y_train = [], []
N_append = 1000  # numbers of waveforms for each chunk update
for i, earthquake in enumerate(earthquake_list):
    if i % N_append == 0:
        print(f'=============={i} / {N_events}=================')

    # this step takes the longest time!
    quake_waveform = np.array(dtfl.get('data/' + str(earthquake)))

    # downsample
    time_new, quake_waveform, dt_new = downsample_series(time, quake_waveform, f_downsample)
    noise_waveform = noise[i, :, :]

    # normalize the amplitude for both
    quake_waveform = quake_waveform - np.mean(quake_waveform, axis=0)
    quake_waveform = quake_waveform / (np.std(quake_waveform, axis=0) + 1e-12)

    noise_waveform = noise_waveform - np.mean(noise_waveform, axis=0)
    noise_waveform = noise_waveform / (np.std(noise_waveform, axis=0) + 1e-12)

    # random shift the signal and scaled with snr
    shift_func = interp1d(time_new + shift[i], quake_waveform, axis=0, kind='nearest', bounds_error=False,
                          fill_value=0.)
    quake_waveform = snr[i] * shift_func(time_new)

    # combine the given snr, stack both signals
    stacked_waveform = quake_waveform + noise_waveform

    # scale again
    scaling_mean = np.mean(stacked_waveform, axis=0)
    stacked_waveform = stacked_waveform - scaling_mean
    scaling_std = np.std(stacked_waveform, axis=0) + 1e-12
    stacked_waveform = stacked_waveform / scaling_std

    quake_waveform = (quake_waveform - scaling_mean) / scaling_std

    # concatenate
    X_train.append(stacked_waveform)
    Y_train.append(quake_waveform)

    if (i + 1) % N_append == 0:
        # Convert to the ndarray
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        # Remove some NaN points in the data
        X_train[np.isnan(X_train)] = 0
        Y_train[np.isnan(Y_train)] = 0

        if i + 1 == N_append:  # create the hdf5 dataset first
            with h5py.File(model_datasets, 'w') as f:
                f.create_dataset('time', data=time_new)
                f.create_dataset('X_train', data=X_train, compression="gzip", chunks=True, maxshape=(None, 600, 3))
                f.create_dataset('Y_train', data=Y_train, compression="gzip", chunks=True, maxshape=(None, 600, 3))
        else:  # update the hdf5 dataset
            with h5py.File(model_datasets, 'a') as f:
                f['X_train'].resize((f['X_train'].shape[0] + N_append), axis=0)
                f['X_train'][-N_append:] = X_train

                f['Y_train'].resize((f['Y_train'].shape[0] + N_append), axis=0)
                f['Y_train'][-N_append:] = Y_train
        X_train, Y_train = [], []

# Check the datasets visually
training_dataset_dir = './training_datasets'
model_datasets = training_dataset_dir + '/training_datasets_STEAD_plus_POHA_snr_40_unshuffled.hdf5'

with h5py.File(model_datasets, 'r') as f:
    time_new = f['time'][:]
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]

i = np.random.randint(0, X_train.shape[0])
plt.close('all')
print(i)
plt.plot(time_new, X_train[i, :, 0])
plt.plot(time_new, Y_train[i, :, 0], alpha=0.7)

# # ============== Following are for testing purpose ==============


# # check the filtering
# dt = 1 / f_downsample
# f_filt = 1
# # highpass filter
# b, a = scipy.signal.butter(4, f_filt * 2 * dt, 'highpass')
# waveform_filt = scipy.signal.filtfilt(b, a, waveform, axis=1)
# noise_filt = scipy.signal.filtfilt(b, a, noise, axis=1)
#
# plt.plot(waveform[12500, :, 0])
# plt.plot(waveform_filt[12500, :, 0])
# plt.plot(noise_filt[12500, :, 0], alpha=0.6)

# Normalization
# data_mean = np.mean(waveform, axis=0)
# data_std = np.std(waveform - data_mean, axis=0)
# waveform_normalized = (waveform - data_mean) / (data_std + 1e-12)
# waveform_normalized = np.reshape(waveform_normalized[:, np.newaxis, :], (-1, 600, 3))
