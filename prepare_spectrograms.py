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

    # extract and scale each part
    X_real = np.real(Sxx_X)
    X_imag = np.imag(Sxx_X)
    Y_real = np.real(Sxx_Y)
    Y_imag = np.imag(Sxx_Y)

    # # MinMax normalization TODO: still looking for better way to avoid specifying offset
    if normalization == 'minmax':
        img_min = np.amin(np.abs(Sxx_X))
        img_max = np.amax(np.abs(Sxx_X))
        X_real = image_scale_MinMax(X_real, img_min, img_max, feature_range=(0.5, 1))
        X_imag = image_scale_MinMax(X_imag, img_min, img_max, feature_range=(0.5, 1))
        Y_real = image_scale_MinMax(Y_real, img_min, img_max, feature_range=(0.5, 1))
        Y_imag = image_scale_MinMax(Y_imag, img_min, img_max, feature_range=(0.5, 1))
        offset = 0.5

    # Standard normalization (zero-mean, unit-variance)
    if normalization == 'standard':
        mean_real, std_real = np.mean(X_real.flatten()), np.std(X_real.flatten())
        mean_imag, std_imag = np.mean(X_imag.flatten()), np.std(X_imag.flatten())
        X_real = image_scale_Standard(X_real, mean_real, std_real)
        X_imag = image_scale_Standard(X_real, mean_imag, std_imag)
        Y_real = image_scale_Standard(Y_real, mean_real, std_real)
        Y_imag = image_scale_Standard(Y_imag, mean_imag, std_imag)
        offset = 0.0

    return X_real, X_imag, Y_real, Y_imag, f, t, offset


# %% Import the data
with h5py.File('./training_datasets_pyrocko_ENZ2.hdf5', 'r') as f:
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
        X1, X2, Y1, Y2, freq, time_win, offset = extract_real_imag_parts(X_train_original[i_tr, i_com, :],
                                                                         Y_train_original[i_tr, i_com, :],
                                                                         normalization='minmax')

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

# adjust the dimension of the datasets
spectrogram_X = np.moveaxis(spectrogram_X, 1, -1)
spectrogram_Y = np.moveaxis(spectrogram_Y, 1, -1)

# %% save the prepared data
dataset_file_name = 'training_datasets_spectrogram_real_imag_minmax.hdf5'
with h5py.File(dataset_file_name, 'w') as f:
    f.attrs['twin'] = twin
    f.attrs['toverlap'] = toverlap
    f.attrs['win_type'] = win_type
    f.attrs['dt'] = dt
    f.attrs['offset'] = offset
    f.create_dataset('frequency', data=freq)
    f.create_dataset('time_win', data=time_win)
    f.create_dataset('X_train', data=spectrogram_X)
    f.create_dataset('Y_train', data=spectrogram_Y)

# plt.hist(spectrogram_X.flatten())
# plt.hist(spectrogram_Y.flatten())


# """Below are testing code to visualize and check the spectrogram"""
# # %% Visualize the normalized images
# X1, X2, Y1, Y2, f, t = extract_real_imag_parts(X_train_original[i_tr, i_com, :],
#                                                Y_train_original[i_tr, i_com, :])
# # X1, X2, Y1, Y2, f, t = extract_amplitude_phase_parts(X_train_original[i_tr, i_com, :],
# #                                                      Y_train_original[i_tr, i_com, :])
# plt.figure()
# plt.subplot(231)
# plt.hist(X1.flatten())
# plt.hist(Y1.flatten())
#
# plt.subplot(234)
# plt.hist(X2.flatten())
# plt.hist(Y2.flatten())
#
# plt.subplot(232)
# plt.pcolormesh(t, f, X1, vmin=0, vmax=1)
#
# plt.subplot(233)
# plt.pcolormesh(t, f, Y1, vmin=0, vmax=1)
#
# plt.subplot(235)
# plt.pcolormesh(t, f, X2, vmin=0, vmax=1)
#
# plt.subplot(236)
# plt.pcolormesh(t, f, Y2, vmin=0, vmax=1)
#
#


# # %%  to test STFT and inverse STFT
# twin = 30
# toverlap = 11
# win_type = 'hann'
# i_tr = 100
# i_com = 0
# dt = time[1] - time[0]
#
# X_waveform = X_train_original[i_tr, i_com, :]
# Y_waveform = Y_train_original[i_tr, i_com, :]
#
# _, _, Sxx_X = waveform_stft(X_waveform, dt,
#                             twin=twin, toverlap=toverlap, win_type=win_type, complex=True)
# f, t, Sxx_Y = waveform_stft(Y_waveform, dt,
#                             twin=twin, toverlap=toverlap, win_type=win_type, complex=True)
#
# # %% inverse transform to recover the waveform
# _, X_waveform_rec = waveform_inverse_stft(Sxx_X, dt=dt, twin=twin, toverlap=toverlap)
# time2, Y_waveform_rec = waveform_inverse_stft(Sxx_Y, dt=dt, twin=twin, toverlap=toverlap)
#
# plt.subplot(221)
# plt.pcolormesh(t, f, abs(Sxx_X), vmin=0, vmax=np.amax(abs(Sxx_X)))
# plt.subplot(222)
# plt.pcolormesh(t, f, abs(Sxx_Y), vmin=0, vmax=np.amax(abs(Sxx_X)))
# plt.subplot(223)
# plt.plot(time, X_waveform)
# plt.plot(time2, X_waveform_rec)
#
# plt.subplot(224)
# plt.plot(time, Y_waveform)
# plt.plot(time2, Y_waveform_rec)




# ============== Not IN USE =============
# def extract_amplitude_phase_parts(X_waveform, Y_waveform):
#     """ Extract the normalized amplitude and phase part of the spectrograms"""
#     _, _, Sxx_X = waveform_stft(X_waveform, dt,
#                                 twin=twin, toverlap=toverlap, win_type=win_type, complex=True)
#     f, t, Sxx_Y = waveform_stft(Y_waveform, dt,
#                                 twin=twin, toverlap=toverlap, win_type=win_type, complex=True)
#
#     # set a better shape
#     t = t[:-1]
#     Sxx_X = Sxx_X[:, :-1]
#     Sxx_Y = Sxx_Y[:, :-1]
#
#     img_min_abs = 0
#     img_max_abs = np.amax(np.abs(Sxx_X))
#     img_min_phs = -np.pi
#     img_max_phs = np.pi
#
#     # extract and scale each part
#     X_abs = np.abs(Sxx_X)
#     X_phs = np.angle(Sxx_X)
#     Y_abs = np.abs(Sxx_Y)
#     Y_phs = np.angle(Sxx_Y)
#
#     X_abs = image_scale_MinMax(X_abs, img_min_abs, img_max_abs, feature_range=(0, 0.95))
#     X_phs = image_scale_MinMax(X_phs, img_min_phs, img_max_phs, feature_range=(0, 1))
#     Y_abs = image_scale_MinMax(Y_abs, img_min_abs, img_max_abs, feature_range=(0, 0.95))
#     Y_phs = image_scale_MinMax(Y_phs, img_min_phs, img_max_phs, feature_range=(0, 1))
#
#     return X_abs, X_phs, Y_abs, Y_phs, f, t