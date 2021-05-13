import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import keras

from utilities import downsample_series


def ml_denoise(waveform_dir, file_name, t_shift):
    with h5py.File(waveform_dir + "/" + file_name, "r") as f:
        time = f["time"][:]
        waveform_BH1 = f["waveforms/BH1"][:]
        waveform_BH2 = f["waveforms/BH2"][:]
        waveform_BHZ = f["waveforms/BHZ"][:]

    # down-sample the data to the same sampling rate with the trained model
    f_downsample = 1
    _, waveform_BH1, _ = downsample_series(time=time, series=waveform_BH1, f_downsampe=f_downsample)
    _, waveform_BH2, _ = downsample_series(time=time, series=waveform_BH2, f_downsampe=f_downsample)
    time, waveform_BHZ, dt = downsample_series(time=time, series=waveform_BHZ, f_downsampe=f_downsample)

    waveforms = np.array([waveform_BH1, waveform_BH2, waveform_BHZ])

    # get the 10-minutes long data
    time_new = np.arange(0, 600) + t_shift
    waveforms = waveforms[:, time_new]

    # scale the waveforms with the mean and std
    traces_mean = np.reshape(np.mean(waveforms, axis=1), (3, 1))
    traces_std = np.reshape(np.std(waveforms, axis=1), (3, 1))

    waveforms_new = waveforms.copy()
    waveforms_new = (waveforms_new - traces_mean) / traces_std

    waveforms_new = np.moveaxis(waveforms_new, 0, -1)
    waveforms_new = waveforms_new.reshape(-1, waveforms_new.shape[0], waveforms_new.shape[1])

    # denoise the waveforms
    Y_predict = model.predict(waveforms_new)

    waveforms_predicted = np.squeeze(Y_predict).T * traces_std + traces_mean

    return time_new, waveforms, waveforms_predicted


# %% load the trained model
model = keras.models.load_model('./Model_and_datasets/Synthetic_seismogram_Z_Conv1DTranspose_ENZ.hdf5')

# load the waveforms
waveform_dir = "./waveforms/events_data_processed"
file_name = "IU.XMAS.M6.3.20190519-145650.hdf5"

_, ax = plt.subplots(3, 1, sharex=True, sharey=True, num=1)

for t_shift in np.arange(0,5400,500):
    time_new, waveforms, waveforms_predicted = ml_denoise(waveform_dir=waveform_dir, file_name=file_name, t_shift=t_shift)

    for i, axi in enumerate(ax):
        axi.plot(time_new, waveforms[i, :], '-k', zorder=1)
        axi.plot(time_new, waveforms_predicted[i, :], '-r', zorder=10)
