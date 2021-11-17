import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import butter, filtfilt, fftconvolve
import h5py
import obspy

from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

import matplotlib
matplotlib.rcParams.update({'font.size': 18})
from utilities import mkdir

# %%
working_dir = os.getcwd()
# waveforms
network_station = "IU.POHA" # HV.HSSD "HV.WRM" "IU.POHA" "HV.HAT"
waveform_dir = working_dir + '/continuous_waveforms'
model_dataset_dir = "Model_and_datasets_1D_all"
# model_dataset_dir = "Model_and_datasets_1D_STEAD2"
# model_dataset_dir = "Model_and_datasets_1D_STEAD_plus_POHA"
bottleneck_name = "LSTM"


# waveform_mseed = waveform_dir + '/' + 'IU.POHA.00.20210630-20210801.mseed'
waveform_mseed = waveform_dir + '/' + 'IU.POHA.00.20210731-20210901.mseed'
#waveform_mseed = waveform_dir + '/HV_data_20210731-20210901/' + network_station + '.*.20210731-20210901.mseed'

tr = obspy.read(waveform_mseed)
tr.merge(fill_value=0)  # in case that there are segmented traces

npts0 = tr[0].stats.npts  # number of samples
dt0 = tr[0].stats.delta  # dt

# event catalog
event_catalog = waveform_dir + '/' + 'catalog.20210731-20210901.xml'

# station information
station = obspy.read_inventory(waveform_dir + '/stations/IU.POHA.00.BH1.xml')
#station = obspy.read_inventory(waveform_dir + '/stations/HV.HAT.*.HHE.xml')
sta_lat = station[0][0].latitude
sta_lon = station[0][0].longitude

# read the catalog
events0 = obspy.read_events(event_catalog)
# this is to show the large earthquake occur
events = events0.filter("magnitude >= 5.5")
# estimate the arrival time of each earthquake to the station
t0 = tr[0].stats.starttime
event_arrival_P = np.zeros(len(events))
event_arrival_S = np.zeros(len(events))
event_time_P = []
epi_distance = np.zeros(len(events))
event_magnitude = np.array([event.magnitudes[0].mag for event in events])
for i_event in range(len(events)):
    event = events[i_event]
    # print(event)
    # print(event.origins[0])
    # % % extract the event information
    event_time = event.origins[0].time
    event_lon = event.origins[0].longitude
    event_lat = event.origins[0].latitude
    event_dep = event.origins[0].depth / 1e3

    # % % estimate the distance and the P arrival time from the event to the station
    try:
        distance_to_source = locations2degrees(sta_lat, sta_lon, event_lat, event_lon)
        epi_distance[i_event] = distance_to_source
        model = TauPyModel(model='iasp91')

        arrivals = model.get_ray_paths(event_dep, distance_to_source, phase_list=['P'])
        P_arrival = arrivals[0].time
        arrivals = model.get_ray_paths(event_dep, distance_to_source, phase_list=['S'])
        S_arrival = arrivals[0].time
        # the relative arrival time on the waveform when the event signal arrives
        event_info = {"time": event_time + P_arrival, "text": []} #str(event.magnitudes[0].mag)
        event_time_P.append(event_info)
        event_arrival_P[i_event] = event_time - t0 + P_arrival
        event_arrival_S[i_event] = event_time - t0 + S_arrival
    except:
        event_arrival_P[i_event] = np.nan
        event_arrival_S[i_event] = np.nan

# load the separated waveforms
waveform_output_dir = waveform_dir + '/' + model_dataset_dir + '/' + network_station

with h5py.File(waveform_output_dir + '/' + bottleneck_name + '_processed_waveforms.hdf5', 'r') as f:
    waveform_time = f['waveform_time'][:]
    waveform_original = f['waveform_original'][:]
    waveform_recovered = f['waveform_recovered'][:]
    noise_recovered = f['noise_recovered'][:]

dt = waveform_time[1] - waveform_time[0]
noise = noise_recovered

# reshape the original waveform and the recovered noise
waveform_original = np.reshape(waveform_original[:, np.newaxis, :], (-1, 600, 3))
noise = np.reshape(noise[:, np.newaxis, :], (-1, 600, 3))

# test the autocorrelation methodology
batch_size = 44760
data_test1 = waveform_original[:batch_size, :, :]
data_test2 = noise[:batch_size, :, :]

# zero-pad to 2048 points
pad_size = 2048
data_test1 = np.concatenate((data_test1, np.zeros((batch_size, pad_size - data_test1.shape[1], data_test1.shape[2])))
                            , axis=1)
data_test2 = np.concatenate((data_test2, np.zeros((batch_size, pad_size - data_test2.shape[1], data_test2.shape[2])))
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
        xcorf_function1[:, :, k] = np.real(calculate_xcorf(data_test1[:, :, i], data_test1[:, :, j]))
        xcorf_function2[:, :, k] = np.real(calculate_xcorf(data_test2[:, :, i], data_test2[:, :, j]))

# with h5py.File(waveform_output_dir + '/' + bottleneck_name + '_xcorr_functions.hdf5', 'w') as f:
#     f.create_dataset("original_xcor", data=xcorf_function1, compression="gzip")
#     f.create_dataset("noise_xcor", data=xcorf_function2, compression="gzip")
#     f.create_dataset("waveform_time", data=waveform_time)
#     f.create_dataset("channel_xcor_list", data=channel_xcor_list)
#
# # Read results and make plots
# with h5py.File(waveform_output_dir + '/' + bottleneck_name + '_xcorr_functions.hdf5', 'r') as f:
#     xcorf_function1 = f["original_xcor"][:]  # xcor from original waveforms
#     xcorf_function2 = f["noise_xcor"][:]  # xcor from separated noise
#     waveform_time = f["waveform_time"][:]
#     channel_xcor_list = f["channel_xcor_list"][:]


def average_xcorr_functions(xcorf_funciton, average_hours, time_pts_xcorf, dt, bandpass_filter=None):
    """Function to average the xcorr function"""
    num_windows = xcorf_funciton.shape[0]  # 1 min for each window
    average_acf = np.zeros((int(num_windows / average_windows) + 1, time_pts_xcorf, 9))
    xcorf_day_time = np.arange(int(num_windows / average_windows) + 1) * average_hours / 24
    xcorf_time_lag = np.arange(time_pts_xcorf) * dt

    for i, j in enumerate(np.arange(0, num_windows, average_windows)):
        average_acf[i, :, :] = np.mean(xcorf_funciton[j:(j + average_windows), :time_pts_xcorf, :], axis=0)

    # if bandpass_filter is not None:
    #     aa, bb = butter(4, bandpass_filter * 2 * dt, 'bandpass')
    #     xcorf_funciton = filtfilt(aa, bb, xcorf_funciton, axis=1)
    if bandpass_filter is not None:
        aa, bb = butter(4, bandpass_filter * 2 * dt, 'bandpass')
        average_acf = filtfilt(aa, bb, average_acf, axis=1)

    return xcorf_time_lag, xcorf_day_time, average_acf


# Calculate the correlation coef with the global average
def plot_correlation_coefficent(average_acf1, average_acf2, xcorf_day_time, figure_name, title=None):
    line_label = ['original waveform', 'separated noise']
    fig, ax = plt.subplots(9, 1, sharex=True, sharey=True, figsize=(4, 14))

    for index_channel in range(9):

        for j, xcorf_function in enumerate([average_acf1, average_acf2]):
            global_average1 = np.mean(xcorf_function[:, :time_pts_xcorf, :], axis=0)

            temp = np.sum(xcorf_function[:, :, index_channel] * global_average1[:, index_channel], axis=1)

            norm1 = np.linalg.norm(xcorf_function[:, :, index_channel], axis=1)
            norm2 = np.linalg.norm(global_average1[:, index_channel])

            corr_coef = temp / norm1 / norm2

            ax[index_channel].plot(xcorf_day_time, corr_coef, label=line_label[j])
            ax[index_channel].plot(event_arrival_S / 24 / 3600, np.ones(event_arrival_S.shape) * 1.3,
                          'x', color='k', linewidth=4)

            ax[index_channel].set_ylabel(channel_xcor_list[index_channel] + ' coef', fontsize=14)
            ax[index_channel].set_ylim(-1.3, 1.4)
            ax[index_channel].set_yticks([-1, 1])
            if index_channel == 0:
                if title is not None:
                    ax[index_channel].set_title(title)

            if index_channel == 8:
                ax[index_channel].set_xlabel('Time (day)', fontsize=14)
                ax[index_channel].legend(fontsize=14)
    plt.savefig(figure_name, bbox_inches='tight')


dt = waveform_time[1] - waveform_time[0]
num_windows = xcorf_function1.shape[0]
average_hours = 6  # time in hours to average the xcorr functions
average_windows = 60 * average_hours  # time in minutes
time_pts_xcorf = 200  # time range in 0.1 s for the xcor functions

# Results without bandpassing filtering
bandpass_filter = None
_, _, average_acf1 = average_xcorr_functions(xcorf_function1, average_hours, time_pts_xcorf, dt, bandpass_filter)
xcorf_time_lag, xcorf_day_time, average_acf2 = \
    average_xcorr_functions(xcorf_function2, average_hours, time_pts_xcorf, dt, bandpass_filter)

plt.close('all')
scale_factor = 6
fig, ax = plt.subplots(3, 3, figsize=(6, 10))
fig.suptitle('(a) Raw waveform')
k = -1
for i in range(3):
    for j in range(3):
        k += 1
        norm_color = cm.colors.Normalize(vmax=abs(average_acf1[:, :, k]).max() / scale_factor,
                                         vmin=-abs(average_acf1[:, :, k]).max() / scale_factor)
        ax[i, j].imshow(average_acf1[:, :, k], norm=norm_color, cmap='RdBu', aspect='auto',
                        extent=[0, time_pts_xcorf * dt, 0, 31], origin='lower')

        ax[i, j].plot(np.ones(event_arrival_S.shape) * (time_pts_xcorf - 10) * dt,
                      event_arrival_S/24/3600, 'x', color='k', linewidth=4)

        ax[i, j].set_title(str(channel_xcor_list[k])[0:7])
        ax[i, j].axes.xaxis.set_visible(False)
        ax[i, j].axes.yaxis.set_visible(False)

        if j == 0:
            ax[i, j].set_ylabel('Days')
            ax[i, j].axes.yaxis.set_visible(True)
        if i == 2:
            ax[i, j].set_xlabel('Time (s)')
            ax[i, j].axes.xaxis.set_visible(True)

plt.savefig(waveform_output_dir + '/original_waveform_xcor.png')

scale_factor = 6
fig, ax = plt.subplots(3, 3, figsize=(6, 10))
fig.suptitle('(b) Separated noise')
k = -1
for i in range(3):
    for j in range(3):
        k += 1
        norm_color = cm.colors.Normalize(vmax=abs(average_acf2[:, :, k]).max() / scale_factor,
                                         vmin=-abs(average_acf2[:, :, k]).max() / scale_factor)
        ax[i, j].imshow(average_acf2[:, :, k], cmap='RdBu', norm=norm_color, aspect='auto',
                        extent=[0, time_pts_xcorf * dt, 0, 31], origin='lower')

        ax[i, j].plot(np.ones(event_arrival_S.shape) * (time_pts_xcorf - 10) * dt,
                      event_arrival_S / 24 / 3600, 'x', color='k', linewidth=4)

        ax[i, j].set_title(str(channel_xcor_list[k])[0:7])
        ax[i, j].axes.xaxis.set_visible(False)
        ax[i, j].axes.yaxis.set_visible(False)

        if j == 0:
            ax[i, j].set_ylabel('Days')
            ax[i, j].axes.yaxis.set_visible(True)
        if i == 2:
            ax[i, j].set_xlabel('Time (s)')
            ax[i, j].axes.xaxis.set_visible(True)

plt.savefig(waveform_output_dir + '/separated_noise_xcor.png')

# plot the comparison of correlation coefficients with global average function
plt.figure(3)
figure_name = waveform_output_dir + '/unfilter_corr_coef_comparision.png'
plot_correlation_coefficent(average_acf1, average_acf2, xcorf_day_time, figure_name)

# Results with bandpassing filtering
bandpass_filter = np.array([0.1, 1])  # [0.1, 1] [1, 2][2, 4.5]
file_name_str = '_' + str(bandpass_filter[0]) + '_' + str(bandpass_filter[1]) + 'Hz'

_, _, average_acf1 = average_xcorr_functions(xcorf_function1, average_hours, time_pts_xcorf, dt, bandpass_filter)
xcorf_time_lag, xcorf_day_time, average_acf2 = \
    average_xcorr_functions(xcorf_function2, average_hours, time_pts_xcorf, dt, bandpass_filter)

scale_factor = 4
plt.close('all')
fig, ax = plt.subplots(3, 3, figsize=(6, 10))
fig.suptitle('(a) Raw waveform [' + str(bandpass_filter[0]) + ' ' + str(bandpass_filter[1]) + '] Hz')
k = -1
for i in range(3):
    for j in range(3):
        k += 1
        norm_color = cm.colors.Normalize(vmax=abs(average_acf1[:, :, k]).max() / scale_factor,
                                         vmin=-abs(average_acf1[:, :, k]).max() / scale_factor)
        ax[i, j].imshow(average_acf1[:, :, k], norm=norm_color, cmap='RdBu', aspect='auto',
                        extent=[0, time_pts_xcorf * dt, 0, 31], origin='lower')

        ax[i, j].plot(np.ones(event_arrival_S.shape) * (time_pts_xcorf - 10) * dt,
                      event_arrival_S / 24 / 3600, 'x', color='k', linewidth=4)

        ax[i, j].set_title(str(channel_xcor_list[k])[0:7])
        ax[i, j].axes.xaxis.set_visible(False)
        ax[i, j].axes.yaxis.set_visible(False)

        if j == 0:
            ax[i, j].set_ylabel('Days')
            ax[i, j].axes.yaxis.set_visible(True)
        if i == 2:
            ax[i, j].set_xlabel('Time (s)')
            ax[i, j].axes.xaxis.set_visible(True)

plt.savefig(waveform_output_dir + '/original_waveform_xcor' + file_name_str + '.png')

scale_factor = 8
fig, ax = plt.subplots(3, 3, figsize=(6, 10))
fig.suptitle('(b) Separated noise [' + str(bandpass_filter[0]) + ' ' + str(bandpass_filter[1]) + '] Hz')
k = -1
for i in range(3):
    for j in range(3):
        k += 1
        norm_color = cm.colors.Normalize(vmax=abs(average_acf2[:, :, k]).max() / scale_factor,
                                         vmin=-abs(average_acf2[:, :, k]).max() / scale_factor)
        ax[i, j].imshow(average_acf2[:, :, k], cmap='RdBu', norm=norm_color, aspect='auto',
                        extent=[0, time_pts_xcorf * dt, 0, 31], origin='lower')

        ax[i, j].plot(np.ones(event_arrival_S.shape) * (time_pts_xcorf - 10) * dt,
                      event_arrival_S / 24 / 3600, 'x', color='k', linewidth=4)

        ax[i, j].set_title(str(channel_xcor_list[k])[0:7])
        ax[i, j].axes.xaxis.set_visible(False)
        ax[i, j].axes.yaxis.set_visible(False)

        if j == 0:
            ax[i, j].set_ylabel('Days')
            ax[i, j].axes.yaxis.set_visible(True)
        if i == 2:
            ax[i, j].set_xlabel('Time (s)')
            ax[i, j].axes.xaxis.set_visible(True)

plt.savefig(waveform_output_dir + '/separated_noise_xcor' + file_name_str + '.png')

# plot the comparison of correlation coefficients with global average function
plt.figure(3)
figure_name = waveform_output_dir + '/filter_corr_coef_comparision' + file_name_str + '.png'
plot_correlation_coefficent(average_acf1, average_acf2, xcorf_day_time,
                            figure_name, title='(a) ' + str(bandpass_filter[0]) + '-' + str(bandpass_filter[1]) + ' Hz')

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
