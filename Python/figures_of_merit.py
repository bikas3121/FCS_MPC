#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of some figures-of-metrit (FOM) for DAC testing

@author: Arnfinn Aas Eielsen
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
import math

from scipy import signal
from scipy import integrate
from matplotlib import pyplot as plt

from welch_psd import welch_psd
from psd_measurements import find_psd_peak
from fit_sinusoid import fit_sinusoid, sin_p


def TS_SINAD(x, t, make_plot=False, plot_label=''):
    """
    Take a time-series for computation of the SINAD using a curve-fitting method.
    Use at least 5 periods of the fundamental carrier signal for a good estimate
    (as prescribed in IEEE Std 1658-2011).
    """

    p_opt = fit_sinusoid(t, x, 1)
    print("p_opt: ", p_opt)  # fitted params.
    x_fit = sin_p(t, *p_opt)

    if make_plot:
        plt.plot(t, x, 'r--', label=plot_label)
        plt.plot(t, x_fit, 'g--', label='fit: A=%5.3f, f=%5.3f, phi=%5.3f, C=%5.3f' % tuple(p_opt))

        plt.xlabel('t')
        plt.ylabel('out')
        plt.legend()
        plt.show()

    error = x - x_fit

    sine_amp = p_opt[0]
    power_c = sine_amp**2/2
    power_noise = np.var(error)

    SINAD = 10*np.log10(power_c/power_noise)

    return SINAD


def FFT_SINAD(x, Fs, make_plot=False, plot_label=''):
    """
    Take a time-series for computation of the SINAD using an FFT-based method.
    Typically needs a farily long time-series for sufficient frequency resolution.
    Rule of thumb: More than 100 periods of the fundamental carrier.
    """

    L = 4  # number of averages for PSD estimation

    N = x.size  # length of original sequence
    M = math.floor(N/L)  # length of sequence segments
    WIN = np.kaiser(M, 38)  # window for high dynamic range

    match 1:
        case 1:
            Pxx, f = welch_psd(x, L, Fs)
        case 2:
            # use library fcn.
            f, Pxx = signal.welch(x, window=WIN, fs=Fs)  # type: ignore

    df = np.mean(np.diff(f))

    # approximate noise floor
    noise_floor = np.median(Pxx)

    # equiv. noise bandwidth
    EQNBW = (np.mean(WIN**2)/((np.mean(WIN))**2))*(Fs/M)
    
    if make_plot:
        plt.loglog(f, Pxx, lw=0.5, label=plot_label)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (V$^2$/Hz)')
        plt.grid()
        plt.legend()
    
    power_c = 0

    match 1:
        case 1:  # use a simple peak-finding algorithm (very similar to MATLAB)
            # make an artificial peak at DC to detect and remove
            Pxx[0] = 0.99*np.max(Pxx)
            power_dc, peak_f_dc, k_max_dc, k_left_dc, k_right_dc = find_psd_peak(Pxx, f, EQNBW, 0)
            if make_plot:
                plt.vlines(x = f[k_max_dc], ymin = Pxx[k_right_dc], ymax = Pxx[k_max_dc], color = "r", lw=0.25)
                plt.hlines(y = Pxx[k_right_dc], xmin = f[k_left_dc], xmax = f[k_right_dc], color = "r")
            # setting to zero to eliminate adding to the total noise power
            Pxx[k_left_dc:k_right_dc] = 0

            # find the maximal peak in the PSD and assume this is the carrier
            power_c, peak_f_c, k_max_c, k_left_c, k_right_c = find_psd_peak(Pxx, f, EQNBW)
            if make_plot:
                plt.vlines(x = f[k_max_c], ymin = Pxx[k_left_c], ymax = Pxx[k_max_c], color = "r", lw=0.25)
                plt.hlines(y = Pxx[k_left_c], xmin = f[k_left_c], xmax = f[k_right_c], color = "r")
            # setting to zero to eliminate adding to the total noise power
            Pxx[k_left_c:k_right_c] = 0

        case 2:  # use scipy.signal.find_peaks()
            # make an artificial peak at DC to detect and remove
            Pxx[0] = 0
            Pxx[1] = 0.99*np.max(Pxx)

            # tune some magic numbers
            th = (np.max(Pxx) - noise_floor)*0.99  # force finding only maximum
            rel_th = noise_floor/(np.max(Pxx) - noise_floor)
            pk_width = np.floor(EQNBW/df)
            pks, pk_props = signal.find_peaks(Pxx, width=pk_width, prominence=th, rel_height=1-rel_th)
            
            if make_plot:
                plt.loglog(f[pks], Pxx[pks], "x")

            if make_plot:
                plt.vlines(x=f[pks], ymin=(Pxx[pks] - pk_props["prominences"]), ymax=Pxx[pks], color = "C1", lw=0.25)
            left_ips = np.floor(pk_props["left_ips"]).astype(int)
            right_ips = np.ceil(pk_props["right_ips"]).astype(int)
            if make_plot:
                plt.hlines(y=pk_props["width_heights"], xmin=f[left_ips], xmax=f[right_ips], color = "C1")
            
            k_left_dc = left_ips[0]  # assume first peak is DC
            k_right_dc = right_ips[0]
            # setting to zero to eliminate adding to the total noise power
            Pxx[k_left_dc:k_right_dc] = 0

            k_left_c = left_ips[1]  # assume second peak is fundamental
            k_right_c = right_ips[1]
            power_c = integrate.simpson(y=Pxx[k_left_c:k_right_c], x=f[k_left_c:k_right_c])
            # setting to zero to eliminate adding to the total noise power
            Pxx[k_left_c:k_right_c] = 0
    
    if make_plot:
        plt.show()
    
    # compute the remaining harmonic and noise distortion.
    power_noise = integrate.simpson(y=Pxx, x=f)

    SINAD = 10*np.log10(power_c/power_noise)

    return SINAD


def main():
    """
    Test the methods.
    """
    Fs = 1.0e6  # sampling rate
    Ts = 1/Fs

    t = np.arange(0, 0.1, Ts)  # time vector
    Fx = 999
    x = 1.0*np.cos(2*np.pi*Fx*t)
    x = x + 0.5*x**2 + 0.25*x**3 + 0.125*x**4 + 0.0625*x**5
    x = x + 0.01*np.random.randn(t.size)

    R_FFT = FFT_SINAD(x, Fs)
    R_TS = TS_SINAD(x, t)

    print("SINAD from FFT: {} SINAD from curvefit: {}".format(R_FFT, R_TS))


if __name__ == "__main__":
    main()
