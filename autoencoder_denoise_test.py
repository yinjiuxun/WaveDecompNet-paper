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

# %% Import the data
with h5py.File('./training_datasets.hdf5', 'r') as f:
    time = f['time'][:]
    X_train_original = f['X_train'][:]
    Y_train_original = f['Y_train'][:]

# # %% Check the dataset
# i = np.random.randint(0, X_train.shape[0])
# plt.figure()
# plt.plot(time, Y_train_original[i, :], '-b')
# plt.plot(time, X_train_original[i, :], '-r')
# plt.show()

# %% ML preprocessing
# 1. normalization
# Tried MinMaxScaler
# sc = MinMaxScaler(feature_range=(0, 1))
# X_train = sc.fit_transform(X_train_original.T).T
# Y_train = sc.transform(Y_train_original.T).T

# Trying StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train_original.T).T
Y_train = sc.transform(Y_train_original.T).T

# %% Check the dataset
i = np.random.randint(0, X_train.shape[0])
plt.figure()
plt.plot(time, X_train[i, :], '-b')
plt.plot(time, Y_train[i, :], '-r')
plt.show()

# 2. reshape
X_train = X_train.reshape(-1, X_train.shape[1], 1)
Y_train = Y_train.reshape(-1, Y_train.shape[1], 1)

# 3. split to training (60%), validation (20%) and test (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.4, random_state=13)
X_validate, Y_validate, X_test, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=20)

# %% build the architecture
# %% Model the Data
BATCH_SIZE = 128
EPOCHS = 300

import keras
from keras.models import Sequential
from keras.layers import Conv1D, AveragePooling1D, MaxPooling1D, UpSampling1D
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM, GRU, Bidirectional
import tensorflow as tf

model = Sequential()
model.add(Conv1D(8, 7, padding='same', activation='relu', input_shape=(2400, 1)))
model.add(MaxPooling1D(2))
model.add(Conv1D(16, 7, padding='same', activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(16, 5, padding='same', activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(32, 5, padding='same', activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(rate=0.1))
model.add(Bidirectional(LSTM(units=16, return_sequences=True, dropout=0.1)))
model.add(LSTM(units=16, return_sequences=True))
model.add(UpSampling1D(3))
model.add(Conv1D(32, 3, padding='same', activation='relu'))
model.add(UpSampling1D(2))
model.add(Conv1D(32, 5, padding='same', activation='relu'))
model.add(UpSampling1D(2))
model.add(Conv1D(16, 5, padding='same', activation='relu'))
model.add(UpSampling1D(2))
model.add(Conv1D(16, 7, padding='same', activation='relu'))
model.add(UpSampling1D(2))
model.add(Conv1D(8, 7, padding='same', activation='relu'))
model.add(Conv1D(1, 7, padding='same'))

print(model.summary())
# %% Compile the model
# binary_crossentropy
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics='accuracy')

# %% train the model
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=6)
model_train = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=1,
                        callbacks=[early_stopping_monitor],
                        validation_data=(X_validate, Y_validate))
