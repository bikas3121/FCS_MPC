#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods to operate on FFT data.

A simple peak finding method for FFT/PSD data with a "very large" single frequency peak.

@author: Arnfinn Aas Eielsen
@license: BSD 3-Clause
"""

import numpy as np
from scipy import integrate

def find_psd_peak(Pxx, f, EQNBW=1, f_find=-1):
    """
    Attempt to find the power and frequency of a windowed sinusoid (a peak in a given PSD estimate)
    using a very simple peak finding algorithm (assuming peaks are "big" and "sharp").
    It simply picks a (local) maximum and finds the (indices for) the peak base by stepping
    the abscissa (index) on both sides for as long as the ordinate (power) is decreasing.
    """

    if f_find == -1: # assume the maximum is an actual peak
        k_max = np.argmax(Pxx)
    elif f_find >= f[0] and f_find <= f[-1]: # peak frequency specified (e.g. a harmonic)
        # find closest bin to specified freq.
        k_find = np.argmin(np.abs(f - f_find))
        # check neighbour values for a larger maximum
        k_left_bin = np.amax([0, k_find-1]) # type: ignore
        k_right_bin = np.amin([k_find+1, Pxx.size-1])
        k_local_max = np.argmax(Pxx[k_left_bin:k_right_bin])
        k_max = k_left_bin + k_local_max
    else:
        # throw an error here
        raise NameError('Invalid Arguments')

    # step down the peak towards the left
    k_left = k_max - 1
    while k_left >= 0 and Pxx[k_left] <= Pxx[k_left + 1]:
        k_left = k_left - 1
    k_left = k_left + 1 # index to the left base of the peak
    
    # step down the peak towards the right
    k_right = k_max + 1
    while k_right < Pxx.size and Pxx[k_right] <= Pxx[k_right - 1]:
        k_right = k_right + 1
    k_right = k_right - 1 # index to the right base of the peak
    
    # estimate a more exact frequency for the peak by computing the central moment of the peak
    f_ = f[k_left:k_right]
    Pxx_ = Pxx[k_left:k_right]
    Pxx_f_dot_product = np.dot(f_, Pxx_)
    Pxx_sum = np.sum(Pxx_)
    peak_f = Pxx_f_dot_product/Pxx_sum
    
    # find the power (area) of the peak
    if k_left < k_right: # more than one point/frequency bin
        # compute power by approximating the area using the Simpson rule
        power = integrate.simpson(y=Pxx[k_left:k_right], x=f[k_left:k_right])
    elif k_right > 0 and k_right < Pxx.size: # single point, but not at the "edges"
        # use the current bin width
        power = Pxx[k_right]*(f[k_right+1]-f[k_right-1])/2
    else: # resort to using the average bin width at the "edges"
        power = Pxx[k_right]*np.mean(np.diff(f)) 
    
    # attempt to correct for any nearby peak overlapping or edge case
    if power < EQNBW*Pxx[k_max]:
        power = EQNBW*Pxx[k_max] # resort to using equivalent noise bandwidth to estimate power
        peak_f = f[k_max]
    
    return power, peak_f, k_max, k_left, k_right
