#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:08:42 2021

@author: Yin9xun
"""
#%%
from asdf import AsdfFile

tree = {'hello': 'world'}
ff = AsdfFile(tree)
ff.write_to('test.asdf')

#%%
import asdf
import numpy as np

ff = AsdfFile()
ff.tree['hello'] = 'world'
ff.write_to('test2.asdf')

#Create some data
sequence = np.arange(100)
squares = sequence**2
random = np.random.random(100)

tree = {
        'foo': 42,
        'name': 'Monty',
        'sequence': sequence,
        'powers': {'squares': squares},
        'random': random
        }

af = asdf.AsdfFile(tree)
af.write_to('example.asdf')

#%%
af = asdf.open('example.asdf')
af.tree

af.tree['powers']['squares']

import numpy as np
expected = [x ** 2 for x in range(100)]
np.equal(af.tree['powers']['squares'], expected).all()



#%% functional form to store into ASDF format
def save_event_waveforms_asdf(output_name):
    # save the event information and denoised waveform to ASDF files
    tree = {
        'event_name': event_name,
        'event_mag': event_mag,
        'event_time': event_time.datetime,
        'event_lon': event_lon,
        'event_lat': event_lat,
        'event_dep': event_dep,
        'waveform_time': time,
        'waveforms': waveform_dict,
        'waveforms_denoised': waveform_denoised_dict
    }

    waveform_ff = asdf.AsdfFile(tree)
    waveform_ff.write_to(output_name)


def save_noise_asdf(output_name):
    # save the separated noise to ASDF files
    tree = {
        'noise': noise_dict,
        'noise_time': time
    }
    noise_ff = asdf.AsdfFile(tree)
    noise_ff.write_to(output_name)