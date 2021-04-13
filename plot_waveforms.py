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

from scipy import signal as sgn

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
working_dir = '/Users/Yin9xun/Work/island_stations/waveforms'


#%% Read catalog first
catalog = obspy.read_events(working_dir + '/events_by_distance.xml')
print(catalog)

#%%
for i_event in range(0,len(catalog),20):
    event = catalog[i_event]
    #% % extract the event information
    event_time = event.origins[0].time
    event_lon = event.origins[0].longitude
    event_lat = event.origins[0].latitude
    event_dep = event.origins[0].depth/1e3
    event_mag = event.magnitudes[0].mag
    
    # read event wavefroms
    
    event_name = "IU.XMAS" + ".M" + str(event_mag) + "." + event_time.strftime("%Y%m%d-%H%M%S")
    fname = "/events_data/" + event_name + '.mseed'
    
    try:
        tr = obspy.read(working_dir + fname)
    except:
        print("Issue with " + "event " + event_time.strftime("%Y%m%d-%H%M%S"))
        continue

# look at the waveforms 0: BH1, 1: BH2, 2: BHZ
    st = tr[0]
    #st = st.filter('lowpass', freq=0.05)
    data = st.data
    time = st.times()
    dt = st.stats.delta
    fs = 1/dt
    
    freq, wave_spect = spectrum(data, dt)

    f, t, Sxx = sgn.spectrogram(data, fs, mode='magnitude', nperseg=int(100/dt), 
                                noverlap=int(90/dt), window='hann')

    vmax = np.max(Sxx.flatten())
    vmin = np.min(Sxx.flatten())
    vmin = 1e-12
    
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.plot(time, data)
    plt.title('M' + str(event_mag))
    plt.subplot(2,1,2)
    sb = plt.pcolormesh(t, f, np.log10(Sxx), shading='gouraud', vmax=np.log10(vmax)/1.4, vmin=np.log10(vmin))
    plt.colorbar(sb)
    plt.yscale('log')
    plt.ylim(1e-3,2)
    #plt.show()    
    plt.savefig(working_dir + '/events_data/figures/' + event_name + '.png')
    plt.close()
    
#%% Estimate the SNR for each traces
sta_lat = 2.0448
sta_lon = -157.4457

mag_all = np.zeros(len((catalog)))
distance_all = np.zeros(len(catalog))
SNR_all = np.zeros((len(catalog), 3)) # for all BH1, BH2, BHZ

for i_event in range(len(catalog)):
    event = catalog[i_event]
    #% % extract the event information
    event_time = event.origins[0].time
    event_lon = event.origins[0].longitude
    event_lat = event.origins[0].latitude
    event_dep = event.origins[0].depth/1e3
    event_mag = event.magnitudes[0].mag
    
    # distance from the event to the station
    distance_all[i_event] = locations2degrees(sta_lat, sta_lon, event_lat, event_lon)
    mag_all[i_event] = event_mag

    fname = "/events_data/IU.XMAS" + ".M" + str(event_mag) + "." + event_time.strftime("%Y%m%d-%H%M%S") + '.mseed'
    
    try:
        tr = obspy.read(working_dir + fname)
    except:
        print("Issue with " + "event " + event_time.strftime("%Y%m%d-%H%M%S"))
        
        mag_all[i_event] = np.NaN
        distance_all[i_event] = np.NaN
        SNR_all[i_event] = np.NaN
        
        continue

# Calculate the SNR for each component
    for i_component in range(3):
        st = tr[i_component]
        #st = st.filter('lowpass', freq=0.1)
        data = st.data
        time = st.times()
        signal = data[time >=3600]
        noise = data[time < 3600]
        SNR_all[i_event, i_component] = SNR(signal, noise)
      
#%% Check the SNR distribution
for i_component, component in enumerate(['BH1', 'BH2', 'BHZ']):
    plt.subplot(1,3,i_component+1)
    sb = plt.scatter(mag_all,distance_all,s=mag_all*2, c=SNR_all[:,1], cmap = 'viridis')
    plt.colorbar(sb)
    plt.title(component)
    plt.xlabel('magnitude')
    plt.ylabel('distance')
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
