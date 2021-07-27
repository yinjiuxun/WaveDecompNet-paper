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
from sklearn.model_selection import train_test_split

# make the output directory
model_dataset_dir = './Model_and_datasets'
model_dataset_dir = './Model_and_datasets_spectrogram'
if not os.path.exists(model_dataset_dir):
    os.mkdir(model_dataset_dir)

# %% Read the pre-processed datasets
model_datasets = './training_datasets/training_datasets_spectrogram_mask.hdf5'
with h5py.File(model_datasets, 'r') as f:
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]

# 3. split to training (60%), validation (20%) and test (20%)
train_size = 0.6
rand_seed1 = 13
rand_seed2 = 20
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size=0.6, random_state=rand_seed1)
X_validate, X_test, Y_validate, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=rand_seed2)

# %% build the architecture
# %% Model the Data
BATCH_SIZE = 256
EPOCHS = 600

# %% Specify the model
# ========================= Time series Conv1D models ==================================================================
# from autoencoder_1D_models import autoencoder_Conv1DTranspose_ENZ
# model, model_name = autoencoder_Conv1DTranspose_ENZ(input_shape=X_train.shape[1:])

# from autoencoder_1D_models import autoencoder_Conv1DTranspose_ENZ2
# model, model_name = autoencoder_Conv1DTranspose_ENZ2(input_shape=X_train.shape[1:])

# from autoencoder_1D_models import autoencoder_Conv1DTranspose_ENZ3
# model, model_name = autoencoder_Conv1DTranspose_ENZ3(input_shape=X_train.shape[1:])

# from autoencoder_1D_models import autoencoder_Conv1DTranspose_ENZ5
# model, model_name = autoencoder_Conv1DTranspose_ENZ5(input_shape=X_train.shape[1:])

# from autoencoder_1D_models import autoencoder_Conv1DTranspose_ENZ6
# model, model_name = autoencoder_Conv1DTranspose_ENZ6(input_shape=X_train.shape[1:])


# ========================= Spectrogram Conv2D models ==================================================================
from autoencoder_2D_models import autoencoder_Conv2D_Spectrogram4, autoencoder_Conv2D_Spectrogram5
model, model_name = autoencoder_Conv2D_Spectrogram5(input_shape=X_train.shape[1:])

# make the model output directory
model_dataset_dir = model_dataset_dir + f'/{model_name}'
if not os.path.exists(model_dataset_dir):
    os.mkdir(model_dataset_dir)

# %% Output the network architecture into a text file
model.summary()
from contextlib import redirect_stdout
with open(model_dataset_dir + f"/{model_name}_Model_summary.txt", "w") as f:
    with redirect_stdout(f):
        model.summary()


# %% Train the model
# %% Compile the model
#model.compile(loss='mean_squared_error', optimizer='adam')
#model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')
#model.compile(loss='binary_crossentropy', optimizer='adam')
# Specify an EarlyStopping
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=20)
# Training
model_train = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=1,
                        callbacks=[early_stopping_monitor],
                        validation_data=(X_validate, Y_validate))
# %% Save the model
model.save(model_dataset_dir + f'/{model_name}_Model.hdf5')


# %% Show loss evolution when training is done
loss = model_train.history['loss']
val_loss = model_train.history['val_loss']
plt.figure()
plt.plot(loss, 'o', label='loss')
plt.plot(val_loss, '-', label='Validation loss')
plt.legend()
plt.title(model_name)
plt.show()
# store the model training history
with h5py.File(model_dataset_dir + f'/{model_name}_Training_history.hdf5', 'w') as f:
    for key, item in model_train.history.items():
        f.create_dataset(key, data=item)


# add some model information
with h5py.File(model_dataset_dir + f'/{model_name}_Dataset_split.hdf5', 'w') as f:
    f.attrs['model_name'] = model_name
    f.attrs['train_size'] = train_size
    f.attrs['rand_seed1'] = rand_seed1
    f.attrs['rand_seed2'] = rand_seed2