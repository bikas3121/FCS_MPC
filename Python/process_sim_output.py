
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Outputs of the process simulations 
@author: Arnfinn A. Eielsen and Bikash Adhikari
@date: 22.02.2024
@license: BSD 3-Clause
"""

# imports
import numpy as np
from scipy import signal
import datetime
from scipy import interpolate
from figures_of_merit import FFT_SINAD, TS_SINAD


class sinad_comp:
    FFT = 1  # FFT based
    CFIT = 2  # curve fit

def process_sim_output(ty, y, Fc, Fs, Nf, TRANSOFF, SINAD_COMP_SEL, plot=False, descr=''):
    # Filter the output using a reconstruction (output) filter
    #print(ty.shape)
    #print(y.shape)
    
    match 1:
        case 1:
            Wc = 2*np.pi*Fc
            b, a = signal.butter(Nf, Wc, 'lowpass', analog=True)  # filter coefficients
            Wlp = signal.lti(b, a)  # filter LTI system instance

            y = y.reshape(-1, 1)  # ensure the vector is a column vector
            y_avg_out = signal.lsim(Wlp, y, ty, X0=None, interp=False)  # filter the output
            y_avg = y_avg_out[1]  # extract the filtered data; lsim returns (T, y, x) tuple, want output y
        case 2:
            bd, ad = signal.butter(Nf, Fc, fs=Fs)
            y = y.reshape(-1, 1).squeeze()  # ensure the vector is a column vector
            y_avg = signal.lfilter(bd, ad, y)
        case 3:
            y = y.reshape(-1, 1)  # ensure the vector is a column vector
            y_avg = y.squeeze()
    
    match SINAD_COMP_SEL:
        case sinad_comp.FFT:  # use FFT based method to detemine SINAD
            R = FFT_SINAD(y_avg[TRANSOFF:-TRANSOFF], Fs, plot, descr)
        case sinad_comp.CFIT:  # use time-series sine fitting based method to detemine SINAD
            y_avg = y_avg.reshape(1, -1).squeeze()
            R = TS_SINAD(y_avg[TRANSOFF:-TRANSOFF], ty[TRANSOFF:-TRANSOFF], plot, descr)

    ENOB = (R - 1.76)/6.02

    # Print FOM
    print(descr + ' SINAD: {}'.format(R))
    print(descr + ' ENOB: {}'.format(ENOB))

    return y_avg, ENOB