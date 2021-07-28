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
N_events = 50000
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
snr = 10 ** rng_snr.uniform(-1, 2, N_events)

# Choose the earthquakes and noise from STEAD
df_earthquakes = df_earthquakes.iloc[chosen_earthquake_index]
earthquake_list = df_earthquakes['trace_name'].to_list()

df_noise = df_noise.iloc[chosen_noise_index]
noise_list = df_noise['trace_name'].to_list()

# Loop over each pair
from utilities import downsample_series
import time as t
tread1, tread2, tread3, tread4, tread5, tread6 = 0., 0., 0., 0., 0., 0.
f_downsample = 10
time = np.arange(0, 6000) * 0.01
dtfl = h5py.File(file_name, 'r')

X_train, Y_train = [], []
for i, (earthquake, noise) in enumerate(zip(earthquake_list, noise_list)):
    if i % 1000 == 0:
        print(f'=============={i} / {N_events}=================')

    t0 = t.time()
    quake_waveform = np.array(dtfl.get('data/'+str(earthquake)))
    noise_waveform = np.array(dtfl.get('data/' + str(noise)))
    t1 = t.time()
    tread1 = t1 - t0 + tread1

    # downsample
    t0 = t.time()
    _, quake_waveform, dt_new = downsample_series(time, quake_waveform, f_downsample)
    time_new, noise_waveform, dt_new = downsample_series(time, noise_waveform, f_downsample)
    t1 = t.time()
    tread2 = t1 - t0 + tread2

    # normalize the amplitude for both
    t0 = t.time()
    quake_waveform = (quake_waveform - np.mean(quake_waveform, axis=0)) / (np.std(quake_waveform, axis=0) + 1e-12)
    noise_waveform = (noise_waveform - np.mean(noise_waveform, axis=0)) / (np.std(noise_waveform, axis=0) + 1e-12)
    t1 = t.time()
    tread3 = t1 - t0 + tread3

    # random shift the signal and scaled with snr
    t0 = t.time()
    shift_func = interp1d(time_new + shift[i], quake_waveform, axis=0, kind='nearest', bounds_error=False, fill_value=0.)
    quake_waveform = snr[i] * shift_func(time_new)
    t1 = t.time()
    tread4 = t1 - t0 + tread4

    # combine the given snr, stack both signals
    t0 = t.time()
    stacked_waveform = quake_waveform + noise_waveform

    # scale again
    scaling_std = np.std(stacked_waveform, axis=0)
    stacked_waveform = stacked_waveform / scaling_std
    quake_waveform = quake_waveform / scaling_std
    t1 = t.time()
    tread5 = t1 - t0 + tread5

    # concatenate
    t0 = t.time()
    X_train.append(stacked_waveform)
    Y_train.append(quake_waveform)
    t1 = t.time()
    tread6 = t1 - t0 + tread6

# Convert to the ndarray
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# make the output directory
training_dataset_dir = './training_datasets'
if not os.path.exists(training_dataset_dir):
    os.mkdir(training_dataset_dir)

# %% Save the pre-processed datasets
model_datasets = training_dataset_dir + '/training_datasets_STEAD_waveform.hdf5'
if not os.path.exists(model_datasets):
    with h5py.File(model_datasets, 'w') as f:
        f.create_dataset('time', data=time_new)
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('Y_train', data=Y_train)



# TODO: consider the chunk-data update for storing h5 data
# import numpy as np
# import h5py
#
# f = h5py.File('MyDataset.h5', 'a')
# for i in range(10):
#
#   # Data to be appended
#   new_data = np.ones(shape=(100,64,64)) * i
#   new_label = np.ones(shape=(100,1)) * (i+1)
#
#   if i == 0:
#     # Create the dataset at first
#     f.create_dataset('data', data=new_data, compression="gzip", chunks=True, maxshape=(None,64,64))
#     f.create_dataset('label', data=new_label, compression="gzip", chunks=True, maxshape=(None,1))
#   else:
#     # Append new data to it
#     f['data'].resize((f['data'].shape[0] + new_data.shape[0]), axis=0)
#     f['data'][-new_data.shape[0]:] = new_data
#
#     f['label'].resize((f['label'].shape[0] + new_label.shape[0]), axis=0)
#     f['label'][-new_label.shape[0]:] = new_label
#
#   print("I am on iteration {} and 'data' chunk has shape:{}".format(i,f['data'].shape))
#
# f.close()