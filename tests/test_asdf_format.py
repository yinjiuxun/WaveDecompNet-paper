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