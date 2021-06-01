# %%
import os
from matplotlib import pyplot as plt
import numpy as np
import h5py

# make the output directory
model_dataset_dir = './Model_and_datasets_spectrogram'
if not os.path.exists(model_dataset_dir):
    os.mkdir(model_dataset_dir)

# %% Import the data
with h5py.File('./training_datasets_spectrogram_real_imag_standard.hdf5', 'r') as f:
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]

# 3. split to training (60%), validation (20%) and test (20%)
from sklearn.model_selection import train_test_split
train_size = 0.6
rand_seed1 = 13
rand_seed2 = 20
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size=0.6, random_state=rand_seed1)
X_validate, X_test, Y_validate, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=rand_seed2)


from keras.models import Sequential
from keras.layers import Conv1D, AveragePooling1D, MaxPooling1D, UpSampling1D, LeakyReLU, Conv1DTranspose, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, Conv2DTranspose
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM, GRU, Bidirectional
BATCH_SIZE = 128
EPOCHS = 600

input_shape = X_train.shape[1:]

# from autoencoder_models import autoencoder_Conv2D_Spectrogram
# model, model_name = autoencoder_Conv2D_Spectrogram(input_shape=input_shape)

from autoencoder_1D_models import autoencoder_Conv2D_Spectrogram2
model, model_name = autoencoder_Conv2D_Spectrogram2(input_shape=input_shape)

model.summary()

# %% Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# %% train the model
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=20)
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


model.save(model_dataset_dir + f'/{model_name}_Model.hdf5')



# store the model training history (will be refactored)
with h5py.File(model_dataset_dir + f'/{model_name}_Training_history.hdf5', 'w') as f:
    for key, item in model_train.history.items():
        f.create_dataset(key, data=item)

# %% Output the network architecture into a text file
from contextlib import redirect_stdout
with open(model_dataset_dir + f"/{model_name}_Model_summary.txt", "w") as f:
    with redirect_stdout(f):
        model.summary()

# add some model information
with h5py.File(model_dataset_dir + f'/{model_name}_Dataset_split.hdf5', 'w') as f:
    f.attrs['model_name'] = model_name
    f.attrs['train_size'] = train_size
    f.attrs['rand_seed1'] = rand_seed1
    f.attrs['rand_seed2'] = rand_seed2