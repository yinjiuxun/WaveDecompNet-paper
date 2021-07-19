from matplotlib import pyplot as plt
import numpy as np
import h5py
import keras
from sklearn.model_selection import train_test_split
import os

# %% Need to specify model_name first
model_name = 'spectrogram_mask_softmax'
#model_name = 'spectrogram_mask_skip_connection'
model_dir = './Model_and_datasets_spectrogram' + f'/{model_name}'
data_dir = './training_datasets'

# %% load dataset
with h5py.File(data_dir + '/training_datasets_spectrogram_mask.hdf5', 'r') as f:
    twin = f.attrs['twin']
    toverlap = f.attrs['toverlap']
    win_type = f.attrs['win_type']
    offset = f.attrs['offset']
    dt = f.attrs['dt']
    frequency = f['frequency'][:]
    time_win = f['time_win'][:]
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]

# %% load model
model = keras.models.load_model(model_dir + '/' + f'{model_name}_Model.hdf5')

# split the model based on the information provided by the model
# split the model based on the information provided by the model
with h5py.File(model_dir + '/' + f'/{model_name}_Dataset_split.hdf5', 'r') as f:
    train_size = f.attrs['train_size']
    rand_seed1 = f.attrs['rand_seed1']
    rand_seed2 = f.attrs['rand_seed2']

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size=0.6, random_state=rand_seed1)
X_validate, X_test, Y_validate, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=rand_seed2)

# %% model evaluation
test_eval = model.evaluate(X_test, Y_test, verbose=0)

# Output some figures
figure_dir = model_dir + '/Figures'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

# %% Show loss evolution
with h5py.File(model_dir + '/' + f'{model_name}_Training_history.hdf5', 'r') as f:
    loss = f['loss'][:]
    val_loss = f['val_loss'][:]
plt.figure()
plt.plot(loss, 'o', label='Training loss')
plt.plot(val_loss, '-', label='Validation loss')
plt.plot([1, len(loss)], [test_eval, test_eval], '-', label='Test loss', linewidth=4)
plt.legend()
plt.title(model_name)
plt.show()
plt.savefig(figure_dir + f"/{model_name}_Loss_evolution.png")


# %% predict the waveforms
Y_predict = model.predict(X_test)

# %% Check the spectrograms
plt.close("all")
i_model = np.random.randint(0, X_test.shape[0])
fig, ax = plt.subplots(X_test.shape[3], 3, sharex=True, sharey=True, num=1, figsize=(6, 8))

vmax = None
vmin = None
for i in range(X_test.shape[3]):
    ax[i, 0].pcolormesh(X_test[i_model, :, :, i], vmax=vmax, vmin=vmin)
for i in range(X_test.shape[3]):
    ax[i, 1].pcolormesh(Y_test[i_model, :, :, i], vmax=1/3, vmin=0)
for i in range(X_test.shape[3]):
    ax[i, 2].pcolormesh(Y_predict[i_model, :, :, i], vmax=1/3, vmin=0)

titles = ['E real', 'E imag', 'N real', 'N imag', 'Z real', 'Z imag']
for i in range(X_test.shape[3]):
    ax[i, 0].set_ylabel(titles[i])
titles = ['X', 'Y', 'Y predict']
for i in range(3):
    ax[0, i].set_title(titles[i])
# TODO: Add the processing for masking, check the scaling (currently the scaling is not right)
# %% inverse transform to time domain
# %%  to test STFT and inverse STFT
from utilities import waveform_stft, waveform_inverse_stft
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, num=2, figsize=(8, 6))
# %% inverse transform to recover the waveform
i = 0
for i in range(3):
    Sxx_X = (X_test[i_model, :, :, i*2] - offset) + (X_test[i_model, :, :, i*2+1] - offset) * 1j
    Sxx_Y = (Y_test[i_model, :, :, i*2] - offset) + (Y_test[i_model, :, :, i*2+1] - offset) * 1j
    Sxx_Y = X_test[i_model, :, :, i*2] * Y_test[i_model, :, :, i * 2] + X_test[i_model, :, :, i*2+1] * Y_test[i_model, :, :, i * 2] * 1j
    Sxx_Y_predict = X_test[i_model, :, :, i*2] * Y_predict[i_model, :, :, i*2] + X_test[i_model, :, :, i*2+1] * Y_predict[i_model, :, :, i*2] * 1j
    Sxx_noise_predict = X_test[i_model, :, :, i * 2] * Y_predict[i_model, :, :, i * 2 + 1] \
                        + X_test[i_model, :, :,i * 2 + 1] * Y_predict[i_model, :,:, i * 2 + 1] * 1j

    _, X_waveform = waveform_inverse_stft(Sxx_X, dt=dt, twin=twin, toverlap=toverlap)
    _, Y_waveform = waveform_inverse_stft(Sxx_Y, dt=dt, twin=twin, toverlap=toverlap)
    _, noise_waveform = waveform_inverse_stft(Sxx_noise_predict, dt=dt, twin=twin, toverlap=toverlap)
    time2, Y_waveform_predict = waveform_inverse_stft(Sxx_Y_predict, dt=dt, twin=twin, toverlap=toverlap)

    scaling = 3

    ax[i, 0].plot(time2, X_waveform)
    ax[i, 0].plot(time2, noise_waveform*scaling)
    ax[i, 1].plot(time2, Y_waveform*scaling)

    ax[i, 2].plot(time2, Y_waveform*scaling)
    ax[i, 2].plot(time2, Y_waveform_predict*scaling)

titles = ['E', 'N', 'Z']
for i in range(3):
    ax[i, 0].set_ylabel(titles[i])
titles = ['X', 'Y', 'Y predict']
for i in range(3):
    ax[0, i].set_title(titles[i])


plt.figure(1)
plt.savefig(figure_dir + f'/{model_name}_Prediction_spectrogram_model_{i_model}.png')
plt.figure(2)
plt.savefig(figure_dir + f'/{model_name}_Prediction_waveforms_model_{i_model}.png')

