#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:55:02 2021

@author: Yin9xun
"""
# %%
import os
from matplotlib import pyplot as plt
import numpy as np
import h5py

# preprocessing scaler, try MinMaxScaler first.
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# make the output directory
model_dataset_dir = './Model_and_datasets'
if not os.path.exists(model_dataset_dir):
    os.mkdir(model_dataset_dir)

# %% Import the data
with h5py.File('./training_datasets_pyrocko_ENZ.hdf5', 'r') as f:
    time = f['time'][:]
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]

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

# Adjust the dimension order for the dataset
X_train = np.moveaxis(X_train, 1, -1)
Y_train = np.moveaxis(Y_train, 1, -1)

# %% Check the dataset
plt.close('all')
i = np.random.randint(0, X_train.shape[0])
_, ax = plt.subplots(3,1, sharex=True, sharey=True)
for i_component, axi in enumerate(ax):
    axi.plot(time, X_train[i, :, i_component], '-b')
    axi.plot(time, Y_train[i, :, i_component], '-r')

# %% Save the pre-processed datasets
model_datasets = model_dataset_dir + '/processed_synthetic_datasets_ENZ.hdf5'
if not os.path.exists(model_datasets):
    with h5py.File(model_datasets, 'w') as f:
        f.create_dataset('time', data=time)
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('Y_train', data=Y_train)

# 3. split to training (60%), validation (20%) and test (20%)
train_size = 0.6
rand_seed1 = 13
rand_seed2 = 20
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size=0.6, random_state=rand_seed1)
X_validate, X_test, Y_validate, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=rand_seed2)


# %% build the architecture
# %% Model the Data
BATCH_SIZE = 128
EPOCHS = 300

from autoencoder_models import autoencoder_Conv1DTranspose_ENZ
model, model_name = autoencoder_Conv1DTranspose_ENZ(input_shape=X_train.shape[1:])

print(model.summary())
# %% Output the network architecture into a text file
from contextlib import redirect_stdout
with open(f"./Model_and_datasets/{model_name}_Model_summary.txt", "w") as f:
    with redirect_stdout(f):
        model.summary()

# %% Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# %% train the model
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=40)
model_train = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=1,
                        callbacks=[early_stopping_monitor],
                        validation_data=(X_validate, Y_validate))

# %% Show loss evolution
loss = model_train.history['loss']
val_loss = model_train.history['val_loss']
plt.figure()
plt.plot(loss, 'o', label='loss')
plt.plot(val_loss, '-', label='Validation loss')
plt.legend()
plt.title(model_name)
plt.show()
plt.savefig(f"./Figures/{model_name}_Loss_evolution.png")

# %% Save the model
model.save(model_dataset_dir + f'/{model_name}_Model.hdf5')

# add some model information
with h5py.File(model_dataset_dir + f'/{model_name}_Dataset_split.hdf5', 'w') as f:
    f.attrs['model_name'] = model_name
    f.attrs['train_size'] = train_size
    f.attrs['rand_seed1'] = rand_seed1
    f.attrs['rand_seed2'] = rand_seed2



