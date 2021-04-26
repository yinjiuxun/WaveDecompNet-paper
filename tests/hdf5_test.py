#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 12:03:35 2021

@author: Yin9xun
"""
# %%
import threshold_functions as threshold
import os
import numpy as np
import sys
import datetime
from matplotlib import pyplot as plt

# import the ASDF format module
import asdf

# import the HDF5 module
import h5py

# import the Obspy modules that we will use in this exercise
import obspy

# functions for STFT (spectrogram)
from scipy import signal as sgn

# import the modules with different threshold functions
sys.path.append('./')


#%%
f = h5py.File("mydataset.hdf5", 'r')
temp1 = f['/waveforms/BH1'][:]
name = f['/event_name'][:]
f.close()

#%%
dset3 = f.create_dataset('subgroup2/dataset_three', (10,), dtype='i')
dset3.name

#%%
dataset_three = f['subgroup2/dataset_three']

#%%
for name in f:
    print(name)
#%%
list(f.keys())
dset = f['mydataset']

#%% Functional form to save the processed waveforms to HDF5 format


def save_event_waveforms_hdf5(output_name):
    with h5py.File(output_name, 'w') as f:
        f.attrs.create('event_name', event_name)
        f.attrs.create('event_mag', event_mag)
        f.attrs.create(
            'event_time', event_time.datetime.strftime('%Y%m%d-%H:%M:%S'))
        f.attrs.create('event_lon', event_lon)
        f.attrs.create('event_lat', event_lat)
        f.attrs.create('event_dep', event_dep)
        f.create_dataset('time', data=time)

        # original waveforms
        grp = f.create_group('waveforms')
        grp.create_dataset("BH1", data=waveform_dict['BH1'])
        grp.create_dataset("BH2", data=waveform_dict['BH2'])
        grp.create_dataset("BHZ", data=waveform_dict['BHZ'])
        # denoised waveforms
        grp = f.create_group('waveforms_denoised')
        grp.create_dataset("BH1", data=waveform_denoised_dict['BH1'])
        grp.create_dataset("BH2", data=waveform_denoised_dict['BH2'])
        grp.create_dataset("BHZ", data=waveform_denoised_dict['BHZ'])


def save_noise_hdf5(output_name):
    with h5py.File(output_name, 'w') as f:
        f.create_dataset('noise_time', data=time)

        # storing the noise in each components
        grp = f.create_group('noise')
        grp.create_dataset("BH1", data=noise_dict['BH1'])
        grp.create_dataset("BH2", data=noise_dict['BH2'])
        grp.create_dataset("BHZ", data=noise_dict['BHZ'])
