import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt

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
N_events = 10
earthquake_seed = 101
noise_seed = 102

# Prepare the random list to get the events and noise series
from numpy.random import default_rng
rng_earthquake = default_rng(earthquake_seed)
chosen_earthquake_index = rng_earthquake.choice(len(df_earthquakes), N_events)

rng_noises_index = default_rng(noise_seed)
chosen_noise_index = rng_noises_index.choice(len(df_noise), N_events)

# Choose the earthquakes and noise from STEAD
df_earthquakes = df_earthquakes.iloc[chosen_earthquake_index]
earthquake_list = df_earthquakes['trace_name'].to_list()

df_noise = df_noise.iloc[chosen_noise_index]
noise_list = df_noise['trace_name'].to_list()

# Loop over each pair
from utilities import downsample_series

f_downsample = 10
time =
for i, (earthquake, noise) in enumerate(zip(earthquake_list, noise_list)):
    print(i)
    print(earthquake)
    print(noise)

    # downsample
    time_new, series_downsample, dt_new = downsample_series(time, series, f_downsample)

    # normalize the amplitude for both

    # random shift the signal

    # combine the given snr

    # stack both signals

    # concatenate

