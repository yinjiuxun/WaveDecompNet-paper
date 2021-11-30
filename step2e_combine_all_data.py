# %%
import os
from matplotlib import pyplot as plt
import numpy as np
import h5py


# %% Read the pre-processed datasets
print("#" * 12 + " Loading data " + "#" * 12)
model_datasets = './training_datasets/training_datasets_STEAD_plus_POHA_snr_40.hdf5'
with h5py.File(model_datasets, 'r') as f:
    X_train1 = f['X_train'][:]
    Y_train1 = f['Y_train'][:]


model_datasets = './training_datasets/training_datasets_STEAD_waveform_snr_40.hdf5'
with h5py.File(model_datasets, 'r') as f:
    X_train2 = f['X_train'][:]
    Y_train2 = f['Y_train'][:]
    time_new = f['time'][:]

X_train = np.concatenate((X_train1, X_train2), axis=0)
Y_train = np.concatenate((Y_train1, Y_train2), axis=0)

temp_int = np.random.choice(X_train.shape[0], X_train.shape[0])
X_train = X_train[temp_int, :, :]
Y_train = Y_train[temp_int, :, :]

# write data
training_dataset_dir = './training_datasets'
model_datasets = training_dataset_dir + '/training_datasets_all_snr_40.hdf5'

with h5py.File(model_datasets, 'w') as f:
    f.create_dataset("X_train", data=X_train)
    f.create_dataset("Y_train", data=Y_train)
    f.create_dataset("time_new", data=time_new)

# Load the data, and can check the datasets visually
training_dataset_dir = './training_datasets'
model_datasets = training_dataset_dir + '/training_datasets_all_snr_40.hdf5'

with h5py.File(model_datasets, 'r') as f:
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]
    time_new = f['time_new'][:]