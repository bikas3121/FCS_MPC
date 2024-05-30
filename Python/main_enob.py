#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""  Run DAC simulation using various noise shaping techniques

@author: Bikash Adhikari 
@date: 23.02.2024
@license: BSD 3-Clause
"""

# %%
import numpy as np
from scipy  import signal
import scipy
import csv
import matplotlib.pyplot as plt
import statistics

from DirectQantization import quantise_signal, generate_code, generate_dac_output
from  quantiser_configurations import quantiser_configurations, get_measured_levels
from MHOQ import MHOQ
from signalProcessing import signalProcessing as SP
from plot_variance import plot_variance
from process_sim_output import process_sim_output


# Generate test signal
def test_signal(SCALE, MAXAMP, FREQ, Rng,  OFFSET, t):
    """
    Generate a test signal (carrier)

    Arguments
        SCALE - percentage of maximum amplitude
        MAXAMP - maximum amplitude
        FREQ - signal frequency in hertz
        OFFSET - signal offset
        t - time vector
    
    Returns
        x - sinusoidal test signal
    """
    return (SCALE/100)*MAXAMP*np.cos(2*np.pi*FREQ*t) + OFFSET  + Rng/2

# %% Chose how to compute SINAD
class sinad_comp:
    CFIT = 1        # curve fitting
    FFT = 2         # fft based 

# Choose sinad computation method
SINAD_COMP_SEL = sinad_comp.CFIT
#SINAD_COMP_SEL = sinad_comp.FFT
# %% Quantiser configurations 
Qconfig = 5
Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(Qconfig)

# %% Sampling frequency and rate
Fs = 1e6
Ts = 1/Fs

# %% Output low-pass filter cutoff order and cutoff frequency
N_lp = 2
Fc_lp = 1e5 # cutoff frequency

# %% Carrier signal
Xcs_SCALE = 100
Xcs_FREQ = 999

# %% Generate time vector

match 1:
    case 1:  # specify duration as number of samples and find number of periods
        Nts = 1e5  # no. of time samples
        Np = np.ceil(Xcs_FREQ*Ts*Nts).astype(int) # no. of periods for carrier

    case 2:  # specify duration as number of periods of carrier
        Np = 10 # no. of periods for carrier
        
Npt = 1  # no. of carrier periods to use to account for transients
Np = Np + Npt

t_end = Np/Xcs_FREQ  # time vector duration
t = np.arange(0, t_end, Ts)  # time vector

# %% Generate carrier/test signal
SIGNAL_MAXAMP = Rng/2 - Qstep  # make headroom for noise dither (see below)
SIGNAL_OFFSET = -Qstep/2  # try to center given quantiser type
Xcs = test_signal(Xcs_SCALE, SIGNAL_MAXAMP, Xcs_FREQ, Rng,  SIGNAL_OFFSET, t)


# %% Reconstruction filter parameters 
match 1:
    case 1:
        Wn = Fc_lp / (Fs / 2)
        # Butterworth Filter properties 
        b, a = signal.butter(N_lp, Wn, 'low')  # tf numerator and denominator
        butter = signal.dlti(*signal.butter(2, Wn,'low'))
        w, h = signal.dimpulse(butter, n = 10)  #w - angular frequency, h= frequency response
        h = h[0]  
    case 2:  # % Perception filter - Goodwin Paper -  equivalent low pass filter
        b = np.array([1, 0.91, 0])
        a = np.array([1 , -1.335, 0.644])
        percep = signal.dlti(b, a)
        w, h = signal.dimpulse(percep, n = 10)  #w - angular frequency, h= frequency response
        h = h[0]  
        # Mns_tf = signal.TransferFunction( bns, ans, dt=1)  # Mns = 1 - Hns
        # Mns = Mns_tf.to_ss()
        # A1, B1, C1, D1 = balreal(Mns.A, Mns.B, Mns.C, Mns.D)

A, B, C, D = signal.tf2ss(b,a) # Transfer function to StateSpace
e, v = np.linalg.eig(A)
# %% Quantiser levels : Ideal and Measured

# Ideal quantiser levels
YQns = YQ     

# Measured quantiser levels
MLns = get_measured_levels(Qconfig)

# %% Quatniser Model
# Quantiser model: 1 - Ideal , 2- Calibrated
QMODEL = 1

# %% Direct Quantization 
C_DIR = quantise_signal(Xcs, Qstep, YQns, Qtype).astype(int)
match QMODEL:
    case 1:
        Xcs_DIR = generate_dac_output(C_DIR, YQns)
    case 2:
        Xcs_DIR = generate_dac_output(C_DIR, MLns)

# %% MPC : Prediction horizon
N = 2

#%% Numerical MPC: Solving MHOQ numerically using Gurobi MILP formulation

MHOQ = MHOQ(Nb, Qstep, QMODEL, A, B, C, D)
# Get Codes
C_MHOQ = MHOQ.get_codes(N, Xcs, YQns, MLns)

match QMODEL:
    case 1:
        Xcs_MHOQ = generate_dac_output(C_MHOQ, YQns)
    case 2:
        Xcs_MHOQ = generate_dac_output(C_MHOQ, MLns)

# %% Signal Processing
tm = t[:Xcs_MHOQ.size]

TRANSOFF = np.floor(Npt*Fs/Xcs_FREQ).astype(int)  # remove transient effects from output
sp = SP(Xcs, b, a, TRANSOFF)

# Filterted reference
F_Xcs = sp.referenceFilter
# Direct quantisation 
F_Xcs_DIR, err_DIR, var_DIR = sp.signalFilter(Xcs_DIR)
# MHOQ
F_Xcs_MHOQ, err_MHOQ, var_MHOQ = sp.signalFilter(Xcs_MHOQ)

fig, ax = plt.subplots()
ax.plot(t[TRANSOFF: TRANSOFF + len(F_Xcs_DIR)], F_Xcs_DIR.squeeze())
ax.plot(t[TRANSOFF: TRANSOFF + len(F_Xcs_MHOQ)], F_Xcs_MHOQ.squeeze())

plot_variance(var_dir = var_DIR,  var_mhoq = var_MHOQ)

# %% Variance Pots
yd = Xcs_DIR[0,:len(tm)].squeeze()
ym = Xcs_MHOQ.squeeze()

ym_avg, ENOB_M = process_sim_output(tm, yd, Fc_lp, Fs, N_lp, TRANSOFF, SINAD_COMP_SEL, True, 'non-linear')
ym_avg, ENOB_M = process_sim_output(tm, ym, Fc_lp, Fs, N_lp, TRANSOFF, SINAD_COMP_SEL, True, 'non-linear')
