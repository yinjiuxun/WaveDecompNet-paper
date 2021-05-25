from matplotlib import pyplot as plt
import numpy as np
import h5py
import keras
from sklearn.model_selection import train_test_split

# %% Need to specify model_name first
model_name = 'spectrogram_real_imag'

# %% load dataset
with h5py.File('./training_datasets_spectrogram_real_imag.hdf5', 'r') as f:
    twin = f.attrs['twin']
    toverlap = f.attrs['toverlap']
    win_type = f.attrs['win_type']
    frequency = f['frequency'][:]
    time_win = f['time_win'][:]
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]

# %% load model
model = keras.models.load_model('./Model_and_datasets/' + f'{model_name}_Model.hdf5')

# split the model based on the information provided by the model
train_size = 0.6
rand_seed1 = 13
rand_seed2 = 20

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size=0.6, random_state=rand_seed1)
X_validate, X_test, Y_validate, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=rand_seed2)

# %% model evaluation
test_eval = model.evaluate(X_test, Y_test, verbose=0)

# %% predict the waveforms
Y_predict = model.predict(X_test)

# %% Check the spectrograms
plt.close("all")
i_model = np.random.randint(0, X_test.shape[0])
fig, ax = plt.subplots(6, 3, sharex=True, sharey=True, num=1)
for i in range(6):
    ax[i, 0].pcolormesh(X_test[i_model, :, :, i])
for i in range(6):
    ax[i, 1].pcolormesh(Y_test[i_model, :, :, i])
for i in range(6):
    ax[i, 2].pcolormesh(Y_predict[i_model, :, :, i])


# %% inverse transform to time domain
# %%  to test STFT and inverse STFT
from utilities import waveform_stft, waveform_inverse_stft
dt = 1

fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, num=2)
# %% inverse transform to recover the waveform
i = 0
for i in range(3):
    Sxx_X = (X_test[i_model, :, :, i*2] - 0.5) + (X_test[i_model, :, :, i*2+1] - 0.5) * 1j
    Sxx_Y = (Y_test[i_model, :, :, i*2] - 0.5) + (Y_test[i_model, :, :, i*2+1] - 0.5) * 1j
    Sxx_Y_predict = (Y_predict[i_model, :, :, i*2] - 0.5) + (Y_predict[i_model, :, :, i*2+1] - 0.5) * 1j

    _, X_waveform = waveform_inverse_stft(Sxx_X, dt=dt, twin=twin, toverlap=toverlap)
    _, Y_waveform = waveform_inverse_stft(Sxx_Y, dt=dt, twin=twin, toverlap=toverlap)
    time2, Y_waveform_predict = waveform_inverse_stft(Sxx_Y_predict, dt=dt, twin=twin, toverlap=toverlap)

    ax[i, 0].plot(time2, X_waveform)
    ax[i, 1].plot(time2, Y_waveform)

    ax[i, 2].plot(time2, Y_waveform)
    ax[i, 2].plot(time2, Y_waveform_predict)