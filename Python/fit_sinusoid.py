#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinusoidal fitting method with inital value guessing

Provies a method for fitting the parameters of a sinusoidal function to
a given input signal (assuming it is dominantly sinusoidal)

First an attempt is made to generate rough esimtates of sinusoidal wave
parameters before passing them on to the standard curve fitting method in
NumPy. Initial values have to be reasonably accurate to ensure convergence
to something reasonable (as this is a non-linear problem with non-unique solutions).

This implements the sinusoidal fitting described in IEEE Std 1658-2011
for testing Digital-to-Analog Converter Devices.

@author: Arnfinn Aas Eielsen
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
from scipy.optimize import curve_fit


def sin_p(x, A, f, phi, C):
    """ 
    Sinusoidal signal, parameterisation alt. 1
    """
    y1 = A*np.sin(2*np.pi*f*x + 2*np.pi*phi) + C
    return y1


def cos_sin_p(x, A0, B0, f0, C0):
    """
    Sinusoidal signal, parameterisation alt. 2
    """
    y2 = A0*np.cos(2*np.pi*f0*x) + B0*np.sin(2*np.pi*f0*x) + C0
    return y2


def fit_sinusoid(x, y, fcn_alt):
    """
    Assuming a dominantly single sinusoidal input signa (with some noise and distortion);
    guessing the initial values using a heuristic method,
    then running standard least-squares curve fit.
    There are two common parametrisations for a sinusoid;
    the method can be configured to use either.
    """
    
    # signal stats
    y_mean = np.mean(y)
    y_std = np.std(y)
    
    # amplitude guess
    A_guess = y_std*np.sqrt(2)

    # offset guess
    C_guess = y_mean

    # frequency and phase guesses
    th = y_std/4  # gating threshold, assume noise amp. is smaller than RMS(sine)/4
    yg = schmitt(y - y_mean, [-th, th])  # gate the signal using Schmitt trigger
    ygd = np.diff(yg)  # use difference for zero crossing detection
    
    # find zero crossing points from low to high (rising)
    idx = []
    for k in range(ygd.size):
        if ygd[k] == 1:
            idx.append(k)

    x_zero_up = [x[i] for i in idx]  # pick timestamps at zero crossings
    
    # guess the period as the average time intervals between rising zero crossings
    T_guess = np.mean(np.diff(x_zero_up))  # period guess (mean value of time between zero crossings)
    f_guess = 1/T_guess;  # frequency guess
    
    # find zero crossing points (rising and falling)
    idx = []
    for k in range(ygd.size):
        if ygd[k] == 1 or ygd[k] == -1:
            idx.append(k)
            
    x_zero = [x[i] for i in idx]  # pick timestamps at zero crossings
    d_zero = [ygd[i] for i in idx]  # pick gated output values at zero crossings
    
    # guessing the phase, this estimate is impacted by the Schitt trigger gate threshold
    if d_zero[0] == 1:
        phi_guess = 1 - x_zero[0]*(f_guess)  # phase guess
    elif d_zero[0] == -1:
        phi_guess = 0.5 - x_zero[0]*(f_guess)  # phase guess
    
    p_opt = []
    match fcn_alt:
        case 1:  # parameterised sinusoid alt. 1
            p1_guess = [A_guess, f_guess, phi_guess, C_guess]  # initial vals
            print("p_guess: ", p1_guess)
            p_opt, p_cov = curve_fit(sin_p, x, y, p0=p1_guess)
        case 2:  # parameterised sinusoid alt. 2
            A0_guess = A_guess*np.sin(2*np.pi*phi_guess)
            B0_guess = A_guess*np.cos(2*np.pi*phi_guess)
            f0_guess = f_guess
            C0_guess = C_guess
            p2_guess = [A0_guess, B0_guess, f0_guess, C0_guess]  # initial vals
            print("p_guess: ", p2_guess)
            p_opt, p_cov = curve_fit(cos_sin_p, x, y, p0=p2_guess)
        
    return p_opt


def schmitt(x, thresholds):
    """
    Implement the behaviour of a Schmitt trigger.
    """ 
    lim = 0  # store current state
    yg = np.zeros(x.size)  # gated signal (output)
    for k in range(x.size):
        if (lim == 0):
            yg[k] = 0
        elif (lim == 1):
            yg[k] = 1
        # change state if signal crosses threshold (low or high)
        if (x[k] <= thresholds[0]):  # going low
            lim = 0
            yg[k] = 0
        elif (x[k] >= thresholds[1]):  # going high
            lim = 1 
            yg[k] = 1
    
    return yg


def main():
    """
    Test the fitting method.
    """
    
    import matplotlib.pyplot as plt

    rng = np.random.default_rng()  # set up random number generator

    fcn_alt = 1 # choose function parameterisation to use
    # experience so far indicates that param. alt. 1 possibly gives better results

    # Pick some random parameters form sinusoid
    p_true = np.array(rng.uniform(size = 4))/rng.uniform(size = 4)
    p_true = [12, 23, 0.45, 2.1]
    p_true[2] = p_true[2]%1  # wrap phase to 2*pi*[0, 1]

    print("p_true: ", p_true)  # true params.

    p_true_2 = np.zeros(4)  # store params. for parametrisation alt. 2

    match fcn_alt:
        case 1:
            print("Alt 1")
        case 2:
            p_true_2[0] = p_true[0]*np.sin(2*np.pi*p_true[2]) # A0
            p_true_2[1] = p_true[0]*np.cos(2*np.pi*p_true[2]) # B0
            p_true_2[2] = p_true[1] # f
            p_true_2[3] = p_true[3] # C0
            print("Alt 2")

    x = np.linspace(0, 5/p_true[1], 1000)  # generate 5 periods

    match fcn_alt:
        case 1:
            print("p_true: ", p_true)  # true params.
            y_signal = sin_p(x, *p_true)
        case 2:
            print("p_true_2: ", p_true_2)  # true params.
            y_signal = cos_sin_p(x, *p_true_2)

    y_noise = 0.1*np.std(y_signal)*rng.normal(size = x.size)

    y = y_signal + y_noise

    y_mean = np.mean(y)
    y_std = np.std(y)

    plt.plot(x, y, 'b-', label='input time-series')

    match fcn_alt:
        case 1:
            p_opt = fit_sinusoid(x, y, 1)
            print("p_opt: ", p_opt)  # fitted params.
            plt.plot(x, sin_p(x, *p_opt), 'g--', label='fit: A=%5.3f, f=%5.3f, phi=%5.3f, C=%5.3f' % tuple(p_opt))
        case 2:
            p_opt = fit_sinusoid(x, y, 2)
            print("p_opt: ", p_opt)  # fitted params.
            plt.plot(x, cos_sin_p(x, *p_opt), 'g--', label='fit: A0=%5.3f, B0=%5.3f, f0=%5.3f, C0=%5.3f' % tuple(p_opt))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()