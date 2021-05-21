import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import keras
import os, glob

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
    waveforms_all = waveforms.copy()

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

    waveforms_new = np.squeeze(waveforms_new).T * traces_std
    waveforms_predicted = np.squeeze(Y_predict).T * traces_std  # + traces_mean

    return time_new, waveforms_new, waveforms_predicted, waveforms_all


# %% Need to specify model_name first
model_name = 'autoencoder_Conv1DTranspose_ENZ'

# %% load model
model = keras.models.load_model('./Model_and_datasets/' + f'/{model_name}_Model.hdf5')

# %% make a directory to hold the realistic waveform results
output_dir = f"./Figures/real_waveforms/{model_name}"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# load the waveforms
waveform_dir = "./waveforms/events_data_processed"
waveform_files = np.array(os.listdir(waveform_dir))
#waveform_files = np.array(glob.glob(waveform_dir + '/*.hdf5'))
for file_name in waveform_files:
    # file_name = "IU.XMAS.M6.3.20190519-145650.hdf5"
    step = 600
    plt.close('all')
    _, ax = plt.subplots(3, 1, sharex=True, sharey=True, num=1, figsize=(5.5, 12))
    for t_shift in np.arange(0, 6900, step):
        time_new, waveforms, waveforms_predicted, _ = ml_denoise(waveform_dir=waveform_dir, file_name=file_name,
                                                                 t_shift=t_shift)
        for i, axi in enumerate(ax):
            axi.plot(time_new, waveforms[i, :], '-k', zorder=1)
            axi.plot(time_new, waveforms_predicted[i, :], '-r', zorder=10)
        if i == 2:
            axi.set_xlabel('Time (s)')

    plt.savefig(output_dir + "/" + file_name[:-5] + '.png')

# TODO: Find out a way to merge the predicted waveforms
# for t_shift in np.arange(600, 6900, step):
#     time_new, _, waveforms_predicted, _ = ml_denoise(waveform_dir=waveform_dir, file_name=file_name, t_shift=t_shift)
#     waveforms_predicted_all[:, (t_shift-600):(t_shift-600+step)] = waveforms_predicted_pre[:, 0:step]
#     waveforms_predicted_all[:, (t_shift-600+step):t_shift] = (waveforms_predicted_pre[:, step:] + waveforms_predicted[:, 0:overlap])/2
#
#     waveforms_predicted_pre = waveforms_predicted.copy()
#
#     if t_shift == 6600:
