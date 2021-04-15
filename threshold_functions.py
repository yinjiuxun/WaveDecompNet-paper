#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:33:23 2021

@author: Yin9xun
"""

#%%
import numpy as np
#%% Define different threshold functions
# soft thresholding
def soft_threshold(y, gamma):
    '''y_thresholded = soft_threshold(y, gamma)
    scale the amplitude by |y-gamma|/|y| when |yi| > gamma, 
    set yi = 0 if yi <= gamma'''
    
    II1 = np.abs(y) > gamma
    II2 = np.abs(y) <= gamma
    x = y.copy()
    x[II1] = x[II1] * (np.abs(x[II1]) - gamma) / np.abs(x[II1])
    x[II2] = 0
    return x

# hard thresholding
def hard_threshold(y, gamma):
    '''y_thresholded = hard_threshold(y, gamma) 
    set yi = 0 if yi <= gamma,
    unmodified if yi > gamma '''
    
    x = y.copy()
    x[np.abs(x)<gamma] = 0
    return x

# scale the coef to denoise
def scale_to_denoise(Sxx, gammaN, Mmax):
    '''y_thresholded = scale_to_denoise(Sxx, gammaN, Mmax) 
    set Sxx_i = 0 if Sxx_i <= gamma,
    scale Sxx_i by |Sxx_i-gammaN|/|Sxx_i| when Mmax >= |Sxx_i| > gammaN,
    unmodified Sxx_i if Sxx_i > Mmax '''
    
    II1 = np.abs(Sxx) > Mmax
    II2 = (np.abs(Sxx) > gammaN) & (np.abs(Sxx) <= Mmax)
    II3 = np.abs(Sxx) < gammaN
    
    Sxx_processed = Sxx.copy()
    #Twxo_processed[II1] = Twxo[II1] * (np.abs(Twxo[II1]) - Mmax) / np.abs(Twxo[II1])
    Sxx_processed[II1] = Sxx[II1]
    Sxx_processed[II2] = Sxx[II2] * (np.abs(Sxx[II2]) - gammaN) / np.abs(Sxx[II2])
    Sxx_processed[II3] = 0
    
    return Sxx_processed