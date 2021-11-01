# %%
import os
from matplotlib import pyplot as plt
import numpy as np
import h5py

# %% Import the data
with h5py.File('./synthetic_noisy_waveforms/synthetic_waveforms_pyrocko_ENZ.hdf5', 'r') as f:
    time = f['time'][:]
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]

# make the output directory
training_dataset_dir = './training_datasets'
if not os.path.exists(training_dataset_dir):
    os.mkdir(training_dataset_dir)

# Applying a more careful scaling for the data:
# For X_train:
# first remove the mean for each component
X_train_mean = np.mean(X_train, axis=2)
X_train = X_train - X_train_mean[:, :, np.newaxis]
# then scale all three components together to get zero-mean and unit variance
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_train_std = np.std(X_train, axis=1)
X_train = X_train / X_train_std[:, np.newaxis]
X_train = np.reshape(X_train, (X_train.shape[0], 3, -1))

# For Y_train:
# directly scale Y_train with the scaling of X_train
Y_train = np.reshape(Y_train, (Y_train.shape[0], -1))
Y_train = Y_train / X_train_std[:, np.newaxis]
Y_train = np.reshape(Y_train, (Y_train.shape[0], 3, -1))


# Adjust the dimension order for the dataset
X_train = np.moveaxis(X_train, 1, -1)
Y_train = np.moveaxis(Y_train, 1, -1)

# %% Check the dataset
# plt.close('all')
# i = np.random.randint(0, X_train.shape[0])
# _, ax = plt.subplots(3,1, sharex=True, sharey=True)
# for i_component, axi in enumerate(ax):
#     axi.plot(time, X_train[i, :, i_component], '-b')
#     axi.plot(time, Y_train[i, :, i_component], '-r')

# %% Save the pre-processed datasets
model_datasets = training_dataset_dir + '/training_datasets_waveform.hdf5'
if not os.path.exists(model_datasets):
    with h5py.File(model_datasets, 'w') as f:
        f.create_dataset('time', data=time)
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('Y_train', data=Y_train)


# %% NOT IN USE NOW: THE sklearn.preprocessing is not so convenient to handle seismic data
# preprocessing scaler, try MinMaxScaler first.
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# %% ML preprocessing
# 1. normalization
# Tried MinMaxScaler
# sc = MinMaxScaler(feature_range=(0, 1))
# X_train = sc.fit_transform(X_train_original.T).T
# Y_train = sc.transform(Y_train_original.T).T

# Scale with StandardScaler
# # Trying StandardScaler based on the 1st components of the data
# sc = StandardScaler()
# sc.fit(np.squeeze(X_train[:, :, 0]).T)
# for i in range(3):
#     X_train[:, :, i] = sc.transform(np.squeeze(X_train[:, :, i]).T).T
#     Y_train[:, :, i] = sc.transform(np.squeeze(Y_train[:, :, i]).T).T