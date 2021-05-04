import os
import matplotlib.pyplot as plt
from pyrocko.gf import LocalEngine, Target, DCSource, ws
from pyrocko import trace
from pyrocko.marker import PhaseMarker
from pyrocko.gf import seismosizer
import numpy as np


def pyrocko_synthesis(stlat, stlon, depth, FM, duration, channel_codes='ENZ'):
    """ Given the source and station parameter, return the synthetic seismograms
        synthetics = pyrocko_synthesis(stlat, stlon, depth, FM, duration)
        stlat and stlon are the station location
        depth: source depth in positive meters
        FM: a list with [strike, dip, rake]
        duration: duration of the STF
    """

    # The store we are going extract data from:
    store_id = 'ak135_2000km_1Hz'
    # First, download a Greens Functions store. If you already have one that you
    # would like to use, you can skip this step and point the *store_superdirs* in
    # the next step to that directory.
    if not os.path.exists(store_id):
        ws.download_gf_store(site='kinherd', store_id=store_id)
    # We need a pyrocko.gf.Engine object which provides us with the traces
    # extracted from the store. In this case we are going to use a local
    # engine since we are going to query a local store.
    engine = LocalEngine(store_superdirs=['.'])
    # Define a list of pyrocko.gf.Target objects, representing the recording
    # devices. In this case one station with a three component sensor will
    # serve fine for demonstation.

    targets = [
        Target(
            lat=stlat,
            lon=stlon,
            store_id=store_id,
            codes=('', 'STA', '', channel_code))
        for channel_code in channel_codes]

    # Let's use a double couple source representation.
    source_dc = DCSource(
        lat=0.,
        lon=0.,
        depth=depth,
        strike=FM[0],
        dip=FM[1],
        rake=FM[2],
        magnitude=8.)
    source_dc.stf = seismosizer.HalfSinusoidSTF(duration=duration)
    # Processing that data will return a pyrocko.gf.Reponse object.
    response = engine.process(source_dc, targets)
    # This will return a list of the requested traces:
    synthetics = response.pyrocko_traces()

    return synthetics


while True:
    # Generate random number of parameters
    channel_codes = 'ENZ'
    lat = np.random.uniform(-14, 14)
    lon = np.random.uniform(-14, 14)
    depth = np.random.uniform(5e3, 3e4)
    strike = np.random.uniform(-180, -180)
    dip = np.random.uniform(0, 90)
    rake = np.random.uniform(-180, 180)
    duration = np.random.uniform(0, 50)

    try:
        synthetic_traces = pyrocko_synthesis(stlat=lat, stlon=lon, depth=depth,
                                             FM=[strike, dip, rake], duration=duration)
        break
    except:
        print('Parameters out of range')

plt.close('all')
fig, ax = plt.subplots(3, 1, sharex=True)
for i in range(len(synthetic_traces)):
    time = np.arange(synthetic_traces[i].tmin, synthetic_traces[i].tmax + synthetic_traces[i].deltat,
                     synthetic_traces[i].deltat)
    synthetic_traces[i].differentiate(n=1, order=2)  # convert disp to velocity
    vel = synthetic_traces[i].ydata
    ax[i].plot(time, vel)
    ax[i].set_xlabel('Time (s)')
    ax[i].set_ylabel('Vel. in ' + channel_codes[i] + ' (m/s)')
plt.show()

# # In addition to that it is also possible to extract interpolated travel times
# # of phases which have been defined in the store's config file.
# store = engine.get_store(store_id)
#
# markers = []
# for t in targets:
#     dist = t.distance_to(source_dc)
#     depth = source_dc.depth
#     arrival_time = store.t('begin', (depth, dist))
#     m = PhaseMarker(tmin=arrival_time,
#                     tmax=arrival_time,
#                     phasename='P',
#                     nslc_ids=(t.codes,))
#     markers.append(m)
#
#
# # Finally, let's scrutinize these traces.
# trace.snuffle(synthetic_traces, markers=markers)
