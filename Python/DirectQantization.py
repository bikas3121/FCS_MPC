
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Implementation of computationally efficient direct (nearest neighbor quantiser) 
@author: Bikash Adhikari
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
from scipy import linalg
from quantiser_configurations import quantiser_type


def quantise_signal(Xcs, Qstep, Qlevels, Qtype):
    # Perfrom the quatnization of the input signal
    # Q _ quantizer set or the quantization levels
    # ref - input/reference signal
    # Returns the quantized values of the refernce signal (Xcs)

    """ INPUTS:

    Parameters:
    -------------
    Xcs         - Reference signal
    Qstep       - Qantizer step size
    Qlevels    - Quantizer levels
    Qtype       - Quantizer type; midread or midrise

    Returns:
    ---------
    q_Xcs       - Quantized signal with Qstep, step size
    """

    # Range of the quantizer
    Vmax = np.max(Qlevels)
    Vmin = np.min(Qlevels)
    # Select quantizer type
    match Qtype:
        case quantiser_type.midtread:
            q_Xcs = np.floor(Xcs/Qstep +1/2)*Qstep
        case quantiser_type.midriser:
            q_Xcs = np.floor(Xcs/Qstep )*Qstep +1/2
    # Quatizer saturation within its range
    np.place(q_Xcs, q_Xcs> Vmax, Vmax)
    np.place(q_Xcs, q_Xcs < Vmin, Vmin)

    return q_Xcs


def generate_code(Nb, q_Xcs, Qstep,  Qtype):
    """ Convert quantised signal into the codes corresponding to the quantiser configurations

    Parameters:
    ------------
    Nb  -       Number of bits
    q_Xcs -     Quantised signal
    Qstep -     Quantiser step size 
    Qtype -     Quantiser type

    Returns:
    ----------
    C_Xcs   - Codes
    """
    match Qtype:
        case quantiser_type.midtread:
            C_Xcs =   q_Xcs/Qstep + 2**(Nb-1) # Mid tread
        case quantiser_type.midriser:
            C_Xcs =   q_Xcs/Qstep + 2**(Nb-1) -1/2 # Mid rise 

    return C_Xcs.astype(int)


def generate_dac_output(C, L):
    """ Generate DAC output : Static DAC output
        Table look-up
    Parameters:
    -----------
    C   - Codes
    L   - Levels

    Returns:
    ----------
    Y - Emulated static DAC output
    """

    # Ensure both C and L have same shape
    C = C.reshape(1,-1)
    L = L.reshape(1,-1)

    Y = np.zeros(C.shape)
    for k in range(0, C.shape[0]):
        Y[k,:] = L[k,C[k,:]]

    return Y




