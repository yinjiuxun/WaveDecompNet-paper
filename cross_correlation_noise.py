import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import butter, filtfilt, fftconvolve
import h5py

from utilities import mkdir

# %%
working_dir = os.getcwd()
# waveforms
network_station1 = "HV.HAT"  # HV.HSSD "HV.WRM" "IU.POHA" "HV.HAT"
network_station2 = "HV.WRM"  # HV.HSSD "HV.WRM" "IU.POHA" "HV.HAT"
waveform_dir = working_dir + '/continuous_waveforms'
model_dataset_dir = "Model_and_datasets_1D_all"
# model_dataset_dir = "Model_and_datasets_1D_STEAD2"
# model_dataset_dir = "Model_and_datasets_1D_STEAD_plus_POHA"
bottleneck_name = "LSTM"

waveform_output_dir1 = waveform_dir + '/' + model_dataset_dir + '/' + network_station1
waveform_output_dir2 = waveform_dir + '/' + model_dataset_dir + '/' + network_station2

with h5py.File(waveform_output_dir1 + '/' + bottleneck_name + '_processed_waveforms.hdf5', 'r') as f:
    waveform_time1 = f['waveform_time'][:]
    waveform_original1 = f['waveform_original'][:]
    waveform_recovered1 = f['waveform_recovered'][:]
    noise_recovered1 = f['noise_recovered'][:]

with h5py.File(waveform_output_dir2 + '/' + bottleneck_name + '_processed_waveforms.hdf5', 'r') as f:
    waveform_time2 = f['waveform_time'][:]
    waveform_original2 = f['waveform_original'][:]
    waveform_recovered2 = f['waveform_recovered'][:]
    noise_recovered2 = f['noise_recovered'][:]

dt = waveform_time1[1] - waveform_time1[0]
# # A test
# noise_recovered1 = waveform_original1 - waveform_recovered1
# noise_recovered2 = waveform_original2 - waveform_recovered2

# reshape the original waveform and the recovered noise
waveform_original1 = np.reshape(waveform_original1[:, np.newaxis, :], (-1, 600, 3))
noise_recovered1 = np.reshape(noise_recovered1[:, np.newaxis, :], (-1, 600, 3))
waveform_recovered1 = np.reshape(waveform_recovered1[:, np.newaxis, :], (-1, 600, 3))
#noise_recovered1 = waveform_original1 - waveform_recovered1 - noise_recovered1

waveform_original2 = np.reshape(waveform_original2[:, np.newaxis, :], (-1, 600, 3))
noise_recovered2 = np.reshape(noise_recovered2[:, np.newaxis, :], (-1, 600, 3))
waveform_recovered2 = np.reshape(waveform_recovered2[:, np.newaxis, :], (-1, 600, 3))
#noise_recovered2 = waveform_original2 - waveform_recovered2 - noise_recovered2

# test the autocorrelation methodology
batch_size = waveform_original2.shape[0]
data_test1 = waveform_original1[:batch_size, :, :]
data_test2 = waveform_original2[:batch_size, :, :]

noise_test1 = noise_recovered1[:batch_size, :, :]
noise_test2 = noise_recovered2[:batch_size, :, :]

earthquake_test1 = waveform_recovered1[:batch_size, :, :]
earthquake_test2 = waveform_recovered2[:batch_size, :, :]

# zero-pad to 2048 points
pad_size = 600
data_test1 = np.concatenate((np.zeros((batch_size, pad_size - data_test1.shape[1], data_test1.shape[2])), data_test1)
                            , axis=1)
data_test2 = np.concatenate((np.zeros((batch_size, pad_size - data_test2.shape[1], data_test2.shape[2])), data_test2)
                            , axis=1)
noise_test1 = np.concatenate(
    (np.zeros((batch_size, pad_size - noise_test1.shape[1], noise_test1.shape[2])), noise_test1)
    , axis=1)
noise_test2 = np.concatenate(
    (np.zeros((batch_size, pad_size - noise_test2.shape[1], noise_test2.shape[2])), noise_test2)
    , axis=1)

def running_mean_spectrum(X, N):
    """Apply the running mean to smooth the spectrum X, running mean is N-point along axis"""
    return fftconvolve(abs(X), np.ones((X.shape[0], N)) / N, mode='same', axes=1)


def calculate_xcorf(waveform1, waveform2):
    """Calculate cross-correlation functions for single station,
    waveform1 and waveform2 are (nun_windows, num_time_points, )"""
    spectra_data_1 = fft(waveform1, axis=1)
    spectra_data_1[0] = 0
    # spectra_data_1 = spectra_data_1 / abs(spectra_data_1)
    spectra_data_1 = spectra_data_1 / (running_mean_spectrum(spectra_data_1, 32) + 1e-8)
    spectra_data_2 = fft(waveform2, axis=1)
    spectra_data_2[0] = 0
    # spectra_data_2 = spectra_data_2 / abs(spectra_data_2)
    spectra_data_2 = spectra_data_2 / (running_mean_spectrum(spectra_data_2, 32) + 1e-8)
    xcorf = ifft(spectra_data_1 * np.conjugate(spectra_data_2), axis=1)
    return xcorf


num_windows = data_test1.shape[0]
num_channels = data_test1.shape[2]
xcorf_function1 = np.zeros((num_windows, pad_size, num_channels ** 2))  # ordered: 11, 12, 1z, 12, 22, 2z, z1, z2, zz
xcorf_function2 = np.zeros((num_windows, pad_size, num_channels ** 2))  # ordered: 11, 12, 1z, 12, 22, 2z, z1, z2, zz

k = -1
channel_list = ['E', 'N', 'Z']
channel_xcor_list = []
for i in range(3):
    for j in range(3):
        print(channel_list[i] + ' - ' + channel_list[j])
        channel_xcor_list.append(channel_list[i] + ' - ' + channel_list[j])
        k += 1
        xcorf_function1[:, :, k] = np.real(calculate_xcorf(data_test1[:, :, i], data_test2[:, :, j]))
        xcorf_function2[:, :, k] = np.real(calculate_xcorf(noise_test1[:, :, i], noise_test2[:, :, j]))

with h5py.File(waveform_output_dir + '/' + bottleneck_name + '_xcorr_functions.hdf5', 'w') as f:
    f.create_dataset("original_xcor", data=xcorf_function1, compression="gzip")
    f.create_dataset("noise_xcor", data=xcorf_function2, compression="gzip")
    f.create_dataset("waveform_time", data=waveform_time)
    f.create_dataset("channel_xcor_list", data=channel_xcor_list)

# Read results and make plots
with h5py.File(waveform_output_dir + '/' + bottleneck_name + '_xcorr_functions.hdf5', 'r') as f:
    xcorf_function1 = f["original_xcor"][:]  # xcor from original waveforms
    xcorf_function2 = f["noise_xcor"][:]  # xcor from separated noise
    waveform_time = f["waveform_time"][:]
    channel_xcor_list = f["channel_xcor_list"][:]


def average_xcorr_functions(xcorf_funciton, average_hours, time_pts_xcorf, dt, bandpass_filter=None):
    """Function to average the xcorr function"""
    num_windows = xcorf_funciton.shape[0]  # 1 min for each window
    average_acf = np.zeros((int(num_windows / average_windows) + 1, time_pts_xcorf, 9))
    xcorf_day_time = np.arange(int(num_windows / average_windows) + 1) * average_hours / 24
    xcorf_time_lag = np.arange(time_pts_xcorf) * dt - time_pts_xcorf * dt / 2

    for i, j in enumerate(np.arange(0, num_windows, average_windows)):
        average_acf[i, :, :] = np.mean(xcorf_funciton[j:(j + average_windows), :time_pts_xcorf, :], axis=0)


    if bandpass_filter is not None:
        aa, bb = butter(4, bandpass_filter * 2 * dt, 'bandpass')
        average_acf = filtfilt(aa, bb, average_acf, axis=1)

    return xcorf_time_lag, xcorf_day_time, average_acf


# Calculate the correlation coef with the global average
def plot_correlation_coefficent(average_acf1, average_acf2, xcorf_day_time, figure_name):
    line_label = ['original waveform', 'separated noise']
    fig, ax = plt.subplots(9, 1, sharex=True, sharey=True, figsize=(6, 14))
    for index_channel in range(9):

        for j, xcorf_function in enumerate([average_acf1, average_acf2]):
            global_average1 = np.mean(xcorf_function[:, :time_pts_xcorf, :], axis=0)

            temp = np.sum(xcorf_function[:, :, index_channel] * global_average1[:, index_channel], axis=1)

            norm1 = np.linalg.norm(xcorf_function[:, :, index_channel], axis=1)
            norm2 = np.linalg.norm(global_average1[:, index_channel])

            corr_coef = temp / norm1 / norm2

            ax[index_channel].plot(xcorf_day_time, corr_coef, label=line_label[j])
            ax[index_channel].set_ylabel(channel_xcor_list[index_channel] + ' coef')
            ax[index_channel].set_ylim(-1.3, 1.3)
            if index_channel == 8:
                ax[index_channel].set_xlabel('Time (day)')
                ax[index_channel].legend()
    plt.savefig(figure_name, bbox_inches='tight')


xcorf_output_dir = waveform_dir + "/cross_correlation_" + network_station1 + '_vs_' + network_station2
mkdir(xcorf_output_dir)

dt = waveform_time1[1] - waveform_time1[0]
num_windows = xcorf_function1.shape[0]
average_hours = 24  # time in hours to average the xcorr functions
average_windows = 60 * average_hours  # time in minutes
time_pts_xcorf = 600  # time range in 0.1 s for the xcor functions

# Results without bandpassing filtering
bandpass_filter = None
_, _, average_acf1 = average_xcorr_functions(xcorf_function1, average_hours, time_pts_xcorf, dt, bandpass_filter)
xcorf_time_lag, xcorf_day_time, average_acf2 = \
    average_xcorr_functions(xcorf_function2, average_hours, time_pts_xcorf, dt, bandpass_filter)

plt.close('all')
scale_factor = 10
fig, ax = plt.subplots(9, 1, figsize=(4, 12),squeeze=False)
k = -1
for i in range(9):
    for j in range(1):
        k += 1
        norm_color = cm.colors.Normalize(vmax=abs(average_acf1[:, :, k]).max() / scale_factor,
                                         vmin=-abs(average_acf1[:, :, k]).max() / scale_factor)
        ax[i, j].imshow(average_acf1[:, :, k], norm=norm_color, cmap='RdBu', aspect='auto',
                        extent=[-time_pts_xcorf * dt/2, time_pts_xcorf * dt/2, 0, 31], origin='lower')
        mean_xcorf = np.mean(average_acf1[:, :, k], axis=0)
        GF = np.diff(mean_xcorf, append=0)
        # ax[i, j].plot(xcorf_time_lag, mean_xcorf/abs(np.amax(mean_xcorf)) * 15 + 15, '-k')
        ax[i, j].plot(xcorf_time_lag, GF / abs(np.amax(GF)) * 10 + 15, '-k')
        ax[i, j].set_ylim(0, 31)
        ax[i, j].set_title(str(channel_xcor_list[k])[0:7])

        if j == 0:
            ax[i, j].set_ylabel('Days')
        if i == 2:
            ax[i, j].set_xlabel('Time (s)')

plt.savefig(xcorf_output_dir + '/original_waveform_xcor.png')

scale_factor = 1
fig, ax = plt.subplots(9, 1, figsize=(4, 12), squeeze=False)
k = -1
for i in range(9):
    for j in range(1):
        k += 1
        norm_color = cm.colors.Normalize(vmax=abs(average_acf2[:, :, k]).max() / scale_factor,
                                         vmin=-abs(average_acf2[:, :, k]).max() / scale_factor)
        ax[i, j].imshow(average_acf2[:, :, k], cmap='RdBu', norm=norm_color, aspect='auto',
                        extent=[-time_pts_xcorf * dt/2, time_pts_xcorf * dt/2, 0, 31], origin='lower')
        mean_xcorf = np.mean(average_acf2[:, :, k], axis=0)
        GF = np.diff(mean_xcorf, append=0)
        # ax[i, j].plot(xcorf_time_lag, mean_xcorf/abs(np.amax(mean_xcorf)) * 15 + 15, '-k')
        ax[i, j].plot(xcorf_time_lag, GF / abs(np.amax(GF)) * 10 + 15, '-k')
        ax[i, j].set_ylim(0, 31)
        ax[i, j].set_title(str(channel_xcor_list[k])[0:7])

        if j == 0:
            ax[i, j].set_ylabel('Days')
        if i == 2:
            ax[i, j].set_xlabel('Time (s)')

plt.savefig(xcorf_output_dir + '/separated_noise_xcor.png')


# Results with bandpassing filtering
bandpass_filter = np.array([0.5, 3])  # [0.1, 1] [1, 2][2, 4.5]
file_name_str = '_' + str(bandpass_filter[0]) + '_' + str(bandpass_filter[1]) + 'Hz'

_, _, average_acf1 = average_xcorr_functions(xcorf_function1, average_hours, time_pts_xcorf, dt, bandpass_filter)
xcorf_time_lag, xcorf_day_time, average_acf2 = \
    average_xcorr_functions(xcorf_function2, average_hours, time_pts_xcorf, dt, bandpass_filter)

scale_factor = 15
plt.close('all')
fig, ax = plt.subplots(9, 1, figsize=(4, 12), squeeze=False, sharex=True, sharey=True)
k = -1
for i in range(9):
    for j in range(1):
        k += 1
        norm_color = cm.colors.Normalize(vmax=abs(average_acf1[:, :, k]).max() / scale_factor,
                                         vmin=-abs(average_acf1[:, :, k]).max() / scale_factor)
        ax[i, j].imshow(average_acf1[:, :, k], norm=norm_color, cmap='RdBu', aspect='auto',
                        extent=[-time_pts_xcorf * dt/2, time_pts_xcorf * dt/2, 0, 31], origin='lower')
        mean_xcorf = np.mean(average_acf1[:, :, k], axis=0)
        GF = np.diff(mean_xcorf, append=0)
        #ax[i, j].plot(xcorf_time_lag, mean_xcorf/abs(np.amax(mean_xcorf)) * 40 + 15, '-k')
        ax[i, j].plot(xcorf_time_lag, GF / abs(np.amax(GF)) * 10 + 15, '-k')
        ax[i, j].set_ylim(0, 31)
        ax[i, j].set_title(str(channel_xcor_list[k])[0:7])

        if j == 0:
            ax[i, j].set_ylabel('Days')
        if i == 8:
            ax[i, j].set_xlabel('Time (s)')

plt.savefig(xcorf_output_dir + '/original_waveform_xcor' + file_name_str + '.png')

scale_factor = 15
fig, ax = plt.subplots(9, 1, figsize=(4, 12), squeeze=False, sharex=True, sharey=True)
k = -1
for i in range(9):
    for j in range(1):
        k += 1
        norm_color = cm.colors.Normalize(vmax=abs(average_acf2[:, :, k]).max() / scale_factor,
                                         vmin=-abs(average_acf2[:, :, k]).max() / scale_factor)
        ax[i, j].imshow(average_acf2[:, :, k], cmap='RdBu', norm=norm_color, aspect='auto',
                        extent=[-time_pts_xcorf * dt/2, time_pts_xcorf * dt/2, 0, 31], origin='lower')
        mean_xcorf = np.mean(average_acf2[:, :, k], axis=0)
        GF = np.diff(mean_xcorf, append=0)
        #ax[i, j].plot(xcorf_time_lag, mean_xcorf/abs(np.amax(mean_xcorf)) * 40 + 15, '-k')
        ax[i, j].plot(xcorf_time_lag, GF / abs(np.amax(GF)) * 12 + 15, '-k')
        ax[i, j].set_ylim(0, 31)
        ax[i, j].set_title(str(channel_xcor_list[k])[0:7])

        if j == 0:
            ax[i, j].set_ylabel('Days')
        if i == 8:
            ax[i, j].set_xlabel('Time (s)')

# plt.plot(np.sum(average_acf1[:, :, 0], axis=0))
# ii=8
# plt.plot(np.sum(average_acf2[:, :, ii], axis=0))
# plt.plot(np.sum(average_acf1[:, :, ii], axis=0))

plt.savefig(xcorf_output_dir + '/separated_noise_xcor' + file_name_str + '.png')


##################################################################
average_acf1 = average_acf1 - np.mean(average_acf1, axis=0)
average_acf2 = average_acf2 - np.mean(average_acf2, axis=0)
plt.close('all')
fig, ax = plt.subplots(3, 3, figsize=(10, 10))
k = -1
for i in range(3):
    for j in range(3):
        k += 1
        norm_color = cm.colors.Normalize(vmax=abs(average_acf1[:, :, k]).max() / 1,
                                         vmin=-abs(average_acf1[:, :, k]).max() / 1)
        ax[i, j].imshow(average_acf1[:, :, k], norm=norm_color, cmap='RdBu', aspect='auto',
                        extent=[0, time_pts_xcorf * dt, 0, 31])
        ax[i, j].set_title(str(channel_xcor_list[k])[0:7])

        if j == 0:
            ax[i, j].set_ylabel('Days')
        if i == 2:
            ax[i, j].set_xlabel('Time (s)')

fig, ax = plt.subplots(3, 3, figsize=(10, 10))
k = -1
for i in range(3):
    for j in range(3):
        k += 1
        norm_color = cm.colors.Normalize(vmax=abs(average_acf2[:, :, k]).max() / 1,
                                         vmin=-abs(average_acf2[:, :, k]).max() / 1)
        ax[i, j].imshow(average_acf2[:, :, k], cmap='RdBu', norm=norm_color, aspect='auto',
                        extent=[0, time_pts_xcorf * dt, 0, 31])
        ax[i, j].set_title(str(channel_xcor_list[k])[0:7])

        if j == 0:
            ax[i, j].set_ylabel('Days')
        if i == 2:
            ax[i, j].set_xlabel('Time (s)')
