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

# utility module
from utilities import mkdir

import torch
from torch_tools import WaveformDataset, EarlyStopping, try_gpu, training_loop
from torch.utils.data import DataLoader
from autoencoder_1D_models_torch import Autoencoder_Conv1D, Attention_bottleneck

# make the output directory
model_dataset_dir = './Model_and_datasets_1D_STEAD'
mkdir(model_dataset_dir)

# %% Read the pre-processed datasets
model_datasets = './training_datasets/training_datasets_STEAD_waveform.hdf5'
with h5py.File(model_datasets, 'r') as f:
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]

# 3. split to training (60%), validation (20%) and test (20%)
train_size = 0.6
test_size = 0.5
rand_seed1 = 13
rand_seed2 = 20
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train,
                                                    train_size=train_size, random_state=rand_seed1)
X_validate, X_test, Y_validate, Y_test = train_test_split(X_test, Y_test,
                                                          test_size=test_size, random_state=rand_seed2)

# Convert to the dataset class for Pytorch (here simply load all the data,
# but for the sake of memory, can also use WaveformDataset_h5)
training_data = WaveformDataset(X_train, Y_train)
validate_data = WaveformDataset(X_validate, Y_validate)

# # The encoder-decoder model with self-attention bottleneck
# model_name = "Autoencoder_Conv1D_attention"
# bottleneck = Attention_bottleneck(64, 4, 0.2)  # Add the attention bottleneck
#
# # The encoder-decoder model with LSTM bottleneck
# model_name = "Autoencoder_Cov1D_LSTM"
# bottleneck = torch.nn.LSTM(64, 32, 2, bidirectional=True,
#                            batch_first=True, dtype=torch.float64)

# Linear bottleneck
# model_name = "Autoencoder_Conv1D_Linear"
# bottleneck = torch.nn.Linear(64, 64, dtype=torch.float64)

# Model without specified bottleneck
model_name = "Autoencoder_Conv1D_None"
bottleneck = None

model = Autoencoder_Conv1D(model_name, bottleneck).to(device=try_gpu())

# make the output directory to store the model information
model_dataset_dir = model_dataset_dir + '/' + model_name
mkdir(model_dataset_dir)

batch_size, epochs, lr = 128, 3, 1e-3
patience = 10  # patience of the early stopping

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_iter = DataLoader(training_data, batch_size=batch_size, shuffle=True)
validate_iter = DataLoader(validate_data, batch_size=batch_size, shuffle=True)

model, avg_train_losses, avg_valid_losses = training_loop(train_iter, validate_iter,
                                                          model, loss_fn, optimizer,
                                                          epochs=epochs, patience=patience, device=try_gpu())
print("Training is done!")

# %% Save the model
torch.save(model, model_dataset_dir + f'/{model_name}_Model.pth')

# %% Show loss evolution when training is done
loss = avg_train_losses
val_loss = avg_valid_losses
plt.figure()
plt.plot(loss, 'o', label='loss')
plt.plot(val_loss, '-', label='Validation loss')
plt.legend()
plt.title(model_name)
plt.show()
# store the model training history
with h5py.File(model_dataset_dir + f'/{model_name}_Training_history.hdf5', 'w') as f:
    f.create_dataset("loss", data=loss)
    f.create_dataset("val_loss", data=val_loss)

# add some model information
with h5py.File(model_dataset_dir + f'/{model_name}_Dataset_split.hdf5', 'w') as f:
    f.attrs['model_name'] = model_name
    f.attrs['train_size'] = train_size
    f.attrs['test_size'] = test_size
    f.attrs['rand_seed1'] = rand_seed1
    f.attrs['rand_seed2'] = rand_seed2
