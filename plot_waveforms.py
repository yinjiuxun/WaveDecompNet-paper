#%%
import os
import numpy as np
import datetime
from matplotlib import pyplot as plt

# import the Obspy modules that we will use in this exercise
import obspy
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

# functions for fft
from scipy.fft import fft, fftfreq, fftshift

#%% Function to plot waveforms (not in use now)
def plot_waveforms(st, starttime=None, endtime=None, ylim=40e-2):
  fig, ax = plt.subplots(len(st),1,sharex=True,sharey=True,squeeze=False,figsize=(10,12))

  for ista in range(len(st)):
    station = st[ista].stats.station
    channel = st[ista].stats.channel
    
    ax[ista][0].plot(st[ista].times("matplotlib"), st[ista].data*1e3, "k-", linewidth=0.5)
    ax[ista][0].xaxis_date()

    if starttime is None:
      starttime_lim = st[ista].stats.starttime.datetime
    else:
      starttime_lim = starttime
    if endtime is None:
      endtime_lim = st[ista].stats.endtime.datetime
    else:
      endtime_lim = endtime

    ax[ista][0].set_xlim([starttime_lim, endtime_lim])
    ax[ista][0].set_ylim([-ylim,ylim])
    ax[ista][0].set_title(station + '.' + channel)
    ax[ista][0].set_ylabel('V (mm/s)')

  fig.autofmt_xdate()
  
#%% 
def SNR(signal, noise):
    snr = 10 * np.log10(signal.std() / noise.std())
    return snr

def spectrum(data, dt, normalized = True):
    wave_spect = abs(fft(data))
    
    if normalized:
        wave_spect = wave_spect/np.max(wave_spect)
        
    freq = fftfreq(len(data), dt)
    
    wave_spect = wave_spect[freq > 0]
    freq = freq[freq > 0]
    
    return freq, wave_spect


#%%
working_dir = '/Users/Yin9xun/Work/island_stations/waveforms/clear'

tr = obspy.read(working_dir + '/*.mseed')

#%% check the spectra of the data
for i_st in range(0, len(tr), 3):
    st = tr[i_st]
    data = st.data
    time = st.times()
    dt = st.stats.delta
    
    freq, wave_spect = spectrum(data, dt)
    
    plt.figure()
    plt.loglog(freq[freq>0], wave_spect[freq>0], linewidth=0.5)
    plt.show()    

#%% compare spectrum of noise vs data
for i_st in range(0, len(tr)):
    st = tr[i_st]
    data = st.data
    time = st.times()
    dt = st.stats.delta
    
    # noise is 3500 s before P arrival, signal is 3600 s after P arrival
    noise = data[time <3600]
    signal = data[(time >=3600) & (time <7200)]
    
    snr = SNR(signal, noise)
    
    signal_freq, signal_spect = spectrum(signal, dt, normalized = False)
    noise_freq, noise_spect = spectrum(noise, dt, normalized = False)
    
    plt.figure(int(i_st/3)+1)
    plt.loglog(signal_freq, signal_spect, color='black', linewidth=0.5)
    plt.loglog(noise_freq, noise_spect, color='gray', linewidth=0.5, alpha=0.4)   
    plt.show()   
    
    plt.figure(100)
    plt.loglog(noise_freq, noise_spect, linewidth=0.5, alpha=0.5)
    plt.show()
    
# The noise and seismic signal seems to be separated in the frequency domain
    
#%% filter the waveform 
for i_st in range(0, len(tr)):
    st = tr[i_st]
    st_filt = st.copy()
    st_filt = st_filt.filter('lowpass', freq=0.2)
    
    data = st.data
    data_filt = st_filt.data
    time = st.times()
    
    plt.figure(int(i_st/3)+1)
    plt.subplot(3,1,i_st%3 + 1)
    plt.plot(time, data, color='gray', linewidth=1)
    plt.plot(time, data_filt, '-k', linewidth=0.5)
    plt.show()
    
    
#
#%% TODO: try the thresholding method in time-frequency domain or in the wavelet domain    
    


#%% TODO: calculate the SNR for each channel
# equation to calculate SNR
st = tr[2]
data = st.data
time = st.times()

# noise is 3500 s before P arrival, signal is 3600 s after P arrival
noise = data[time <3500]
signal = data[(time >=3600) & (time <7200)]

snr = SNR(signal, noise)
plt.figure()
plt.plot(time, data)
plt.show()
