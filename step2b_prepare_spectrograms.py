# %%
import os
from matplotlib import pyplot as plt
import numpy as np
import h5py


def image_scale_MinMax(img, img_min, img_max, feature_range=(0, 1)):
    """Normalize the image with the given min and max to a specified feature_range"""
    img = (img - img_min) / (img_max - img_min + 1e-14) * (feature_range[1] - feature_range[0]) + feature_range[0]
    return img


def image_scale_Standard(img, mean, std):
    """Normalize the image with the given min and max to a specified feature_range"""
    img = (img - mean) / std
    return img


def extract_real_imag_parts(X_waveform, Y_waveform, normalization):
    """ Extract the normalized real and imaginary part of the spectrograms
    X_real, X_imag, Y_real, Y_imag, f, t, offset = extract_real_imag_parts(X_waveform, Y_waveform, normalization)
    normalization can be "minmax", "standard"
    """
    _, _, Sxx_X = waveform_stft(X_waveform, dt,
                                twin=twin, toverlap=toverlap, win_type=win_type, complex=True)
    f, t, Sxx_Y = waveform_stft(Y_waveform, dt,
                                twin=twin, toverlap=toverlap, win_type=win_type, complex=True)

    # set a better shape
    t = t[:-1]
    Sxx_X = Sxx_X[:, :-1]
    Sxx_Y = Sxx_Y[:, :-1]
    Sxx_noise = Sxx_X - Sxx_Y

    # extract and scale each part
    X_real = np.real(Sxx_X)
    X_imag = np.imag(Sxx_X)
    Y_1 = np.real(Sxx_Y)
    Y_2 = np.imag(Sxx_Y)
    offset = 0.0

    # # MinMax normalization TODO: still looking for better way to avoid specifying offset
    if normalization == 'minmax':
        img_min = np.amin(np.abs(Sxx_X))
        img_max = np.amax(np.abs(Sxx_X))
        X_real = image_scale_MinMax(X_real, img_min, img_max, feature_range=(0.5, 1))
        X_imag = image_scale_MinMax(X_imag, img_min, img_max, feature_range=(0.5, 1))
        Y_1 = image_scale_MinMax(Y_1, img_min, img_max, feature_range=(0.5, 1))
        Y_2 = image_scale_MinMax(Y_2, img_min, img_max, feature_range=(0.5, 1))
        offset = 0.5

    # Standard normalization (zero-mean, unit-variance)
    if normalization == 'standard':
        mean_real, std_real = np.mean(X_real.flatten()), np.std(X_real.flatten())
        mean_imag, std_imag = np.mean(X_imag.flatten()), np.std(X_imag.flatten())
        X_real = image_scale_Standard(X_real, mean_real, std_real)
        X_imag = image_scale_Standard(X_real, mean_imag, std_imag)
        Y_1 = image_scale_Standard(Y_1, mean_real, std_real)
        Y_2 = image_scale_Standard(Y_2, mean_imag, std_imag)
        offset = 0.0

    # Return the signal and noise mask (Zhu et al., 2019)
    if normalization == 'mask':
        mean_real, std_real = np.mean(X_real.flatten()), np.std(X_real.flatten())
        mean_imag, std_imag = np.mean(X_imag.flatten()), np.std(X_imag.flatten())
        X_real = image_scale_Standard(X_real, mean_real, std_real)
        X_imag = image_scale_Standard(X_real, mean_imag, std_imag)
        Y_1 = 1/(1 + abs(Sxx_noise)/(abs(Sxx_Y) + 1e-6)) # signal mask
        Y_2 = 1 - Y_1 # noise mask


    return X_real, X_imag, Y_1, Y_2, f, t, offset


# %%
training_data_dir = "./training_datasets/"
if not os.path.exists(training_data_dir):
    os.mkdir(training_data_dir)

# %% Import the data
with h5py.File('./synthetic_noisy_waveforms/synthetic_waveforms_pyrocko_ENZ.hdf5', 'r') as f:
    time = f['time'][:]
    X_train_original = f['X_train'][:]
    Y_train_original = f['Y_train'][:]

# %% Preprocessing the data to get Min-Max-Scaled spectrograms
from utilities import waveform_stft, waveform_inverse_stft

twin = 30
toverlap = 11  # chose 11 here to make a better shape of the spectrogram
win_type = 'hann'

spectrogram_X, spectrogram_Y = [], []
dt = time[1] - time[0]

for i_tr in range(X_train_original.shape[0]):  # loop over traces
    if i_tr % 2000 == 0:
        print(f'========{i_tr}/' + str(X_train_original.shape[0]) + '==========')

    # Spectrogram X
    spectrogram_X_curr, spectrogram_Y_curr = [], []
    for i_com in range(3):  # loop over components
        # # this is to extract the real and imaginary part of the spectrogram
        normalization = 'mask'
        X1, X2, Y1, Y2, freq, time_win, offset = extract_real_imag_parts(X_train_original[i_tr, i_com, :],
                                                                         Y_train_original[i_tr, i_com, :],
                                                                         normalization=normalization)
        # append spectrogram of each component for X and Y datasets
        spectrogram_X_curr.append(X1)
        spectrogram_X_curr.append(X2)
        spectrogram_Y_curr.append(Y1)
        spectrogram_Y_curr.append(Y2)

    # append spectrogram of each trace
    spectrogram_X.append(spectrogram_X_curr)
    spectrogram_Y.append(spectrogram_Y_curr)


# transform to numpy array
spectrogram_X = np.array(spectrogram_X)
spectrogram_Y = np.array(spectrogram_Y)

# adjust the scaling for the softmax max
if normalization == 'mask':
    spectrogram_Y = spectrogram_Y / 3

# adjust the dimension of the datasets
spectrogram_X = np.moveaxis(spectrogram_X, 1, -1)
spectrogram_Y = np.moveaxis(spectrogram_Y, 1, -1)

# %% save the prepared data
training_dataset_name = training_data_dir + '/training_datasets_spectrogram_' + normalization + '.hdf5'
with h5py.File(training_dataset_name, 'w') as f:
    f.attrs['twin'] = twin
    f.attrs['toverlap'] = toverlap
    f.attrs['win_type'] = win_type
    f.attrs['dt'] = dt
    f.attrs['offset'] = offset
    f.create_dataset('frequency', data=freq)
    f.create_dataset('time_win', data=time_win)
    f.create_dataset('X_train', data=spectrogram_X)
    f.create_dataset('Y_train', data=spectrogram_Y)
