# %% import modules
from matplotlib import pyplot as plt
import numpy as np
import h5py
import keras

# %% load model
model = keras.models.load_model('./Model_and_datasets/Synthetic_seismogram_Z_Conv1DTranspose_ENZ.hdf5')
# %% load dataset
with h5py.File('./Model_and_datasets/Synthetic_seismogram_Autoencoder_model_datasets_ENZ.hdf5', 'r') as f:
    time = f['time'][:]
    X_test = f['X_test'][:]
    Y_test = f['Y_test'][:]

# %% predict the waveforms
Y_predict = model.predict(X_test)

components = "ENZ"
# %% Visualize the results, comparing waveforms and their spectra
i_show = np.random.randint(0, X_test.shape[0], 5)
plt.close("all")
fig, ax = plt.subplots(5, 3, figsize=(12, 10), num=1, sharey=True, sharex=True)
for i in range(5):
    for j in range(3):
        ax[i, j].plot(time, X_test[i_show[i], :, j], '-k', label='X test')
        ax[i, j].plot(time, Y_test[i_show[i], :, j], '-b', label='Y test')
        ax[i, j].plot(time, Y_predict[i_show[i], :, j], '-r', label='Y predicted')

        if i == 0:
            ax[i, j].set_title(components[j])
            if j == 0:
                ax[i, j].legend()
        if i == 4:
            ax[i, j].set_xlabel("Time (s)")
plt.show()

# %% Compare the Fourier spectrum
from utilities import waveform_fft
fig, ax = plt.subplots(5, 3, figsize=(12, 10), num=2, sharey=True, sharex=True)
for i in range(5):
    for j in range(3):
        dt = time[1]-time[0]
        _, sp1 = waveform_fft(X_test[i_show[i], :, j], dt)
        _, sp2 = waveform_fft(Y_test[i_show[i], :, j], dt)
        freq, sp3 = waveform_fft(Y_predict[i_show[i], :, j], dt)

        ax[i, j].loglog(freq, sp1, '-k', label='X test')
        ax[i, j].loglog(freq, sp2, '-b', label='Y test')
        ax[i, j].loglog(freq, sp3, '-r', label='Y predicted')

        if i == 0:
            ax[i, j].set_title(components[j])
            if j == 0:
                ax[i, j].legend()
        if i == 4:
            ax[i, j].set_xlabel("Freq (Hz)")
plt.show()

# %%Compare the spectrogram
from utilities import waveform_stft
fig, ax = plt.subplots(5, 3, figsize=(12, 10), num=3, sharey=True, sharex=True)
for i in range(5):
    for j in range(3):
        dt = time[1]-time[0]
        _, _, sxx1 = waveform_stft(X_test[i_show[i], :, j], dt,twin=30, toverlap=5)
        _, _, sxx2 = waveform_stft(Y_test[i_show[i], :, j], dt,twin=30, toverlap=5)
        freq, time_spec, sxx3 = waveform_stft(Y_predict[i_show[i], :, j], dt,twin=30, toverlap=5)

        vmax = np.amax([np.amax(sxx1), np.amax(sxx2), np.amax(sxx3)])
        level = np.arange(0.2,1,0.2) * vmax

        ax[i, j].contourf(time_spec, freq, sxx1, vmax=vmax, alpha=0.5)
        ax[i, j].contour(time_spec, freq, sxx2, levels=level, colors='b', alpha=0.8)
        ax[i, j].contour(time_spec, freq, sxx3, levels=level, colors='r', alpha=0.9)

        if i == 0:
            ax[i, j].set_title(components[j])
        if j == 0:
            ax[i, j].set_ylabel("Freq (Hz)")
        if i == 4:
            ax[i, j].set_xlabel("Time (s)")
plt.show()

plt.figure(1)
plt.savefig('./Figures/Model_prediction_ENZ_waveforms.png')
plt.figure(2)
plt.savefig('./Figures/Model_prediction_ENZ_spectra.png')
plt.figure(3)
plt.savefig('./Figures/Model_prediction_ENZ_spectrogram.png')

# %% Visualize the model
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='./Figures/synthetic_seismograms_model_plot_ENZ.png', show_shapes=True, show_layer_names=True)