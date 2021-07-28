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

# filterering the dataframe
df_earthquakes = df[(df.trace_category == 'earthquake_local')]
print(f'total number of earthquakes: {len(df_earthquakes)}')
# choose noise
df_noise = df[(df.trace_category == 'noise')]
print(f'total number of noises: {len(df_noise)}')


# Set the number of waveforms that will be used during the training
N_events = 5000
earthquake_seed = 101
noise_seed = 102
shift_seed = 103
snr_seed = 104

# Prepare the random list to get the events and noise series
from numpy.random import default_rng
rng_earthquake = default_rng(earthquake_seed)
chosen_earthquake_index = rng_earthquake.choice(len(df_earthquakes), N_events)

rng_noises_index = default_rng(noise_seed)
chosen_noise_index = rng_noises_index.choice(len(df_noise), N_events)

rng_shift = default_rng(shift_seed)
shift = rng_shift.uniform(-30, 60, N_events)

rng_snr = default_rng(snr_seed)
snr = 10 ** rng_snr.uniform(-1, 1, N_events)

# Choose the earthquakes and noise from STEAD
df_earthquakes = df_earthquakes.iloc[chosen_earthquake_index]
earthquake_list = df_earthquakes['trace_name'].to_list()

df_noise = df_noise.iloc[chosen_noise_index]
noise_list = df_noise['trace_name'].to_list()

# make the output directory
training_dataset_dir = './training_datasets'
if not os.path.exists(training_dataset_dir):
    os.mkdir(training_dataset_dir)
model_datasets = training_dataset_dir + '/training_datasets_STEAD_waveform_update.hdf5'

# Loop over each pair
from utilities import downsample_series
import time as t
tread1, tread2, tread3, tread4, tread5, tread6 = 0., 0., 0., 0., 0., 0.
f_downsample = 10
time = np.arange(0, 6000) * 0.01
dtfl = h5py.File(file_name, 'r')

X_train, Y_train = [], []
N_append = 1000  # numbers of waveforms for each chunk update
for i, (earthquake, noise) in enumerate(zip(earthquake_list, noise_list)):
    if i % N_append == 0:
        print(f'=============={i} / {N_events}=================')

    # this step takes the longest time!
    quake_waveform = np.array(dtfl.get('data/'+str(earthquake)))
    noise_waveform = np.array(dtfl.get('data/' + str(noise)))

    # downsample
    _, quake_waveform, dt_new = downsample_series(time, quake_waveform, f_downsample)
    time_new, noise_waveform, dt_new = downsample_series(time, noise_waveform, f_downsample)

    # normalize the amplitude for both
    quake_waveform = (quake_waveform - np.mean(quake_waveform, axis=0)) / (np.std(quake_waveform, axis=0) + 1e-12)
    noise_waveform = (noise_waveform - np.mean(noise_waveform, axis=0)) / (np.std(noise_waveform, axis=0) + 1e-12)

    # random shift the signal and scaled with snr
    shift_func = interp1d(time_new + shift[i], quake_waveform, axis=0, kind='nearest', bounds_error=False, fill_value=0.)
    quake_waveform = snr[i] * shift_func(time_new)

    # combine the given snr, stack both signals
    stacked_waveform = quake_waveform + noise_waveform

    # scale again
    scaling_std = np.std(stacked_waveform, axis=0)
    stacked_waveform = stacked_waveform / scaling_std
    quake_waveform = quake_waveform / scaling_std

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
model_datasets = training_dataset_dir + '/training_datasets_STEAD_waveform_update.hdf5'

with h5py.File(model_datasets, 'r') as f:
    time_new = f['time'][:]
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]

i_data = np.random.choice(np.arange(X_train.shape[0]), 1)
plt.close('all')
fig, axi = plt.subplots(3, 1, sharex=True, sharey=True)
for i, axi_i in enumerate(axi):
    axi_i.plot(time_new, X_train[i_data, :, i].flatten(), '-r')
    axi_i.plot(time_new, Y_train[i_data, :, i].flatten(), '-b')
axi[-1].set_xlabel('Time (s)')

