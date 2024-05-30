#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Power spectral density (PSD) estimation using a modified version of the Welch method.

@author: Arnfinn Eielsen
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal

def welch_psd(x, L, Fs=1.0, ONE_SIDED=1):
    """
    Compute auto-correlation PSD estimate Pxx from
    x - input time-series
    L - number of averages
    Fs - Samping frequency
    ONE_SIDED - return one-sided spectrum (default)

    The method is modified to support PSD measurements:
    1) No segment overlap (no real need as time-series has to be long for freq. resolution)
    2) Remove mean value (can interfere with peak finding, also not needed for dynamic meas.)
    3) Windowing for high dynamic range (Kaiser with beta = 38, avoid leakage when noise is small)
    """

    N = x.size # length of original sequence
    M = math.floor(N/L) # length of sequence segments
    f = np.arange(0, 1, 1/M) # normalized PSD frequencies
    
    # windowing (esp. useful if the segments are short)
    #WIN = np.ones(M) # i.e. no window
    WIN = np.kaiser(M, 38) # Kaiser window for large dynamic range
    
    x = x - np.mean(x) # remove mean value to minimise DC component
    
    # PSD for input and output sequence, cross-PSD between input and output sequence
    Pxx = np.zeros(M)
    for k in range(L):
        x_seg = x[k*M:(k+1)*M] # segment
        
        x_win = x_seg*WIN # windowing segment
        Xft = np.fft.fft(x_win)/math.sqrt(2*math.pi*M)
        
        Pxx = Pxx + np.abs(Xft)**2; # averaging the auto-correlation PSD
    
    Pwin = sum(abs(WIN)**2)/M; # window "power" correction (for Welch method)
    Pxx = Pxx/(L*Pwin); # scale and correct average
    
    # One-sided spectrum
    if ONE_SIDED:
        f = np.array_split(f, 2)
        f = f[0]
        
        Pxx = np.array_split(Pxx, 2);
        Pxx = 2*Pxx[0]
    
    Pxx = Pxx/(Fs/(2*np.pi))
    f = f*Fs
    
    return Pxx, f

def main():
    """
    Test the method and compare to SciPy library
    """
    Fs = 1.0e6 # sampling rate
    Ts = 1/Fs

    t = np.arange(0, 0.2, Ts) # time vector
    Fx = 999
    x = 1.0*np.cos(2*np.pi*Fx*t)
    x = x + 0.5*x**2 + 0.25*x**3 + 0.125*x**4 + 0.0625*x**5
    x = x + 0.01*np.random.randn(t.size)

    Pxx, f = welch_psd(x, 10, Fs)

    f_cmp, Pxx_cmp = signal.welch(x, fs=Fs, nperseg=t.size/10) # compare to library

    plt.loglog(f, Pxx, lw=0.5)
    plt.loglog(f_cmp, Pxx_cmp, lw=0.5)
    plt.ylim([1e-13, 1])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (V$^2$/Hz)')
    plt.show()

if __name__ == "__main__":
    main()
