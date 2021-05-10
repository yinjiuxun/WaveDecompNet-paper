import os
import matplotlib.pyplot as plt
from pyrocko.gf import LocalEngine, Target, DCSource, ws
from pyrocko import trace
from pyrocko.marker import PhaseMarker
from pyrocko.gf import seismosizer
import numpy as np
# import Ricker
import ricker


def random_ricker(synthetic_length, dt, max_amplitude=10, min_duration=0.1, max_duration=30):
    """Return a randomly produced Ricker wavelet with length synthetic_length and dt = dt:

    syn_signal = random_ricker(synthetic_length, dt, max_amplitude=10, min_duration=0.1, max_duration=30)

    """

    peak_loc = np.random.random() * synthetic_length
    fc = 1 / np.random.uniform(low=min_duration, high=max_duration)
    amp = np.random.random() * max_amplitude

    # produce the Ricker wavelet
    syn_signal = ricker.ricker(f=fc, len=synthetic_length, dt=dt, peak_loc=peak_loc)
    syn_signal = syn_signal / np.amax(abs(syn_signal))  # normalize the Ricker wavelet
    syn_signal = amp * syn_signal

    return syn_signal


def pyrocko_synthesis(src_info, store_superdirs, store_id, channel_codes='ENZ'):
    """ Given the source and station parameter, return the synthetic seismograms

        synthetics = pyrocko_synthesis(src_info, store_superdirs, store_id, channel_codes='ENZ')
        src_info is a dictionary including the source location, focal mechanism and duration
        store_superdirs: a list of Green's function store path
        store_id: Green's function store

    """

    # The store we are going extract data from:
    store_id = store_id
    # First, download a Greens Functions store. If you already have one that you
    # would like to use, you can skip this step and point the *store_superdirs* in
    # the next step to that directory.
    if not os.path.exists(store_superdirs[0] + '/' + store_id):
        ws.download_gf_store(site='kinherd', store_id=store_id)
    # We need a pyrocko.gf.Engine object which provides us with the traces
    # extracted from the store. In this case we are going to use a local
    # engine since we are going to query a local store.
    engine = LocalEngine(store_superdirs=store_superdirs)
    # Define a list of pyrocko.gf.Target objects, representing the recording
    # devices. In this case one station with a three component sensor will
    # serve fine for demonstation.

    targets = [
        Target(
            lat=0.,
            lon=0.,
            store_id=store_id,
            codes=('', 'STA', '', channel_code))
        for channel_code in channel_codes]

    # Let's use a double couple source representation.
    source_dc = DCSource(
        lat=src_info['lat'],
        lon=src_info['lon'],
        depth=src_info['depth'],
        strike=src_info['strike'],
        dip=src_info['dip'],
        rake=src_info['rake'],
        magnitude=8.)
    source_dc.stf = seismosizer.HalfSinusoidSTF(duration=src_info['duration'])
    # Processing that data will return a pyrocko.gf.Reponse object.
    response = engine.process(source_dc, targets)
    # This will return a list of the requested traces:
    synthetics = response.pyrocko_traces()

    return synthetics


def random_pyrocko_synthetics(store_superdirs, store_id, max_amplitude=10):
    """

    Randomly produce the synthetic velocity waveforms in ENZ components
    time, synthetic_waveforms, src_info, channel_codes = random_pyrocko_synthetics(store_superdirs, store_id)

    """
    # Generate random number of parameters
    channel_codes = 'ENZ'
    amp = np.random.random() * max_amplitude
    while True:
        lat = np.random.uniform(-14, 14)
        lon = np.random.uniform(-14, 14)
        depth = np.random.uniform(5e3, 3e4)
        strike = np.random.uniform(-180, -180)
        dip = np.random.uniform(0, 90)
        rake = np.random.uniform(-180, 180)
        duration = np.random.uniform(0, 50)
        src_info = {'lat': lat, 'lon': lon, 'depth': depth,
                    'strike': strike, 'dip': dip, 'rake': rake,
                    'duration': duration}
        try:
            synthetic_traces = pyrocko_synthesis(src_info=src_info,
                                                 store_superdirs=store_superdirs,
                                                 store_id=store_id)
            break
        except:
            print('Parameters out of range')

    time = np.arange(synthetic_traces[0].tmin, synthetic_traces[0].tmax + synthetic_traces[0].deltat,
                     synthetic_traces[0].deltat)

    synthetic_waveforms = []
    for i in range(len(synthetic_traces)):
        time = np.arange(synthetic_traces[i].tmin, synthetic_traces[i].tmax + synthetic_traces[i].deltat,
                         synthetic_traces[i].deltat)
        synthetic_traces[i].differentiate(n=1, order=2)  # convert disp to velocity
        vel = synthetic_traces[i].ydata
        synthetic_waveforms.append(vel)

    synthetic_waveforms = np.array(synthetic_waveforms)
    synthetic_waveforms = amp * synthetic_waveforms / np.amax(abs(synthetic_waveforms))

    return time, synthetic_waveforms, src_info, channel_codes

# # Test the functions and visualize the waveforms
# # Ricker wavelets
# synthetic_length = 60
# dt = 0.1
# ricker_waveform = random_ricker(synthetic_length=synthetic_length, dt=dt)
# time = np.arange(0, synthetic_length, dt)
# plt.close('all')
# plt.plot(time, ricker_waveform)
# plt.show()


# # Pyrocko synthetics
# time, synthetic_waveforms, src_info, channels = random_pyrocko_synthetics(store_superdirs=['./pyrocko_synthetics'],
#                                                          store_id='ak135_2000km_1Hz')
# plt.close('all')
# fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
# for i in range(3):
#     ax[i].plot(time, synthetic_waveforms[i, :])
#     ax[i].set_xlabel('Time (s)')
#     ax[i].set_ylabel('Vel. in ' + channels[i] + ' (m/s)')
# plt.show()


