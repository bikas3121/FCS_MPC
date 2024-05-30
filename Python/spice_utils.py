#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""File wrangling for SPICE simulations.

@author: Trond Ytterdal, Bikash Adhikari, Arnfinn Eielsen
@date: 19.03.2024
@license: BSD 3-Clause
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import fileinput
import subprocess
import datetime
from scipy import signal
from scipy import interpolate
import pickle
from prefixed import Float
from tabulate import tabulate

from lin_method_util import lm, dm
from figures_of_merit import FFT_SINAD, TS_SINAD
from quantiser_configurations import qws


class sinad_comp:
    FFT = 1  # FFT based
    CFIT = 2  # curve fit


class sim_config:
    def __init__(self, qconfig, lin, dac, fs, t, fc, nf, carrier_scale, carrier_freq):
        self.qconfig = qconfig
        self.lin = lin
        self.dac = dac
        self.fs = fs
        self.t = t
        self.fc = fc
        self.nf = nf
        self.carrier_scale = carrier_scale
        self.carrier_freq = carrier_freq
    
    def __str__(self):
        s = str(self.qconfig) + '\n'
        s = s + str(self.lin) + '\n'
        s = s + str(self.dac) + '\n'
        s = s + 'Fs=' + f'{Float(self.fs):.0h}' + '\n'
        s = s + 'Fc=' + f'{Float(self.fc):.0h}' + '\n'
        s = s + 'Nf=' + f'{Float(self.nf):.0h}' + '\n'
        s = s + 'Xs=' + f'{Float(self.carrier_scale):.0h}' + '\n'
        s = s + 'Fx=' + f'{Float(self.carrier_freq):.0h}' + '\n'

        return s + '\n'


def addtexttofile(filename, text):
    f = open(filename, 'w')
    f.write(text)
    f.close()


def get_bit(value, bit_index):
    rval = value & (1 << bit_index)
    if rval != 0:
        return 1
    else:
        return 0


def get_pwl_string(c, Ts, Ns, dnum, vbpc, vdd, trisefall):
    """
    Generate picewise linear (PWL) waveform description string to be read by SPICE.
    
    Arguments
        c - codes
        Ts - sampling time (in microseconds)
        Ns - number of samples
        vbpc, vdd, trisefall - waveform specs.
    
    Returns
        rval - PWL string
    """
    if get_bit(c[0], dnum) == 0:
        rval = "0," + vdd + " "
    else:
        rval = "0," + vbpc + " "
    deltat = trisefall/2
    for i in range(0, Ns-1):
        time = (i+1)*Ts*1e6  # microseconds
        if get_bit(c[i], dnum) == 0 and get_bit(c[i+1], dnum) == 1:
            rval += " " + str(time - deltat) + "u," + vdd + " " \
                + str(time + deltat) + "u," + vbpc
        elif get_bit(c[i], dnum) == 1 and get_bit(c[i+1], dnum) == 0:
            rval += " " + str(time - deltat) + "u," + vbpc + " " \
                + str(time + deltat) + "u," + vdd
    rval = rval + "\n"

    return rval


def get_inverted_pwl_string(c, Ts, Ns, dnum, vbpc, vdd, trisefall):
    """
    Generate inverted picewise linear (PWL) waveform description string to be read by SPICE.
    
    Arguments
        c - codes
        Ts - sampling time (in microseconds)
        Ns - number of samples
        vbpc, vdd, trisefall - waveform specs.
    
    Returns
        rval - PWL string
    """
    if get_bit(c[0], dnum) == 0:
        rval = "0," + vbpc + " "
    else:
        rval = "0," + vdd + " "
    deltat = trisefall/2
    for i in range(0, Ns-1):
        time = (i+1)*Ts*1e6  # microseconds
        if get_bit(c[i], dnum) == 0 and get_bit(c[i+1], dnum) == 1:
            rval += " " + str(time - deltat) + "u," + vbpc + " " \
                + str(time + deltat) + "u," + vdd
        elif get_bit(c[i], dnum) == 1 and get_bit(c[i+1], dnum) == 0:
            rval += " " + str(time - deltat) + "u," + vdd + " " \
                + str(time + deltat) + "u," + vbpc
    rval = rval + "\n"

    return rval


def run_spice_sim(spicef, outputf, outdir='spice_output/', spice_path='ngspice'):
    """
    Run SPICE simulaton using provided filenames

    Arguments
        spicef - SPICE batch file
        outputf - Output files name
    """
    
    print(spicef)
    print(outputf)

    cmd = [spice_path, '-o', outdir + outputf + '.log',
                    # '-r', outdir + outputf + '.bin',
                    '-b', outdir + spicef]

    print(cmd)

    subprocess.run(cmd)


def run_spice_sim_parallel(spicef_list, outputf_list, outdir='spice_output/', spice_path='ngspice'):
    """
    Run SPICE simulaton using provided filenames

    Arguments
        spicef_list - SPICE batch files
        outputf_list - Output files names
    """
    
    cmd_list = []
    for k in range(0, len(spicef_list)):
        cmd = [spice_path, '-o', outdir + outputf_list[k] + '.log',
            #'-r', outdir + outputf_list[k] + '.bin',
            '-b', outdir + spicef_list[k]]
        print(cmd)
        cmd_list.append(cmd)

    procs_list = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for cmd in cmd_list]
    
    for proc in procs_list:
        print('Waiting for SPICE to return...')
        proc.wait()


def gen_spice_sim_file(C, Nb, t, Ts, QConfig, outdir, seed=1, seq=0):
    """
    Set up SPICE simulaton file for a given DAC circuit description and save.

    Arguments
        c - codes
        Nb - no. of bit
        t - time vector
        Ts - sampling time
        QConfig - quantiser config.
        outdirname - put files in this directory
        seed - randomisation seed for circuit
        seq - sequence
    """
    
    wavf = 'spice_pwl_wav.txt'
    cmdf = 'spice_cmds.txt'
    
    tempdir = 'spice_temp'
    circdir = 'spice_circuits'
    #outdir = os.path.join('spice_output', outdirname)

    if os.path.exists(outdir):
        print('Putting output files in existing directory: ' + outdir)
    else:
        os.mkdir(outdir) 

    wav_str = ''
    
    match QConfig:
        case qws.w_06bit:  # 6 bit DAC
            c = C.astype(int)
            nsamples = len(c)

            t1 = '\n'
            t2 = '\n'
            vbpc = '3.28'
            vdd = '5.0'
            Tr = 1e-3  # the rise-time for edges, in µs
            for k in range(0, Nb):  # generate PWL strings
                k_str = str(k)
                t1 += 'vdp' + k_str + ' data' + k_str + ' 0 pwl ' + \
                    get_pwl_string(c, Ts, nsamples, k, vbpc, vdd, Tr)
                t2 += 'vdn' + k_str + ' datai' + k_str + ' 0 pwl ' + \
                    get_inverted_pwl_string(c, Ts, nsamples, k, vbpc, vdd, Tr)
            wav_str = t1 + t2

            circf = 'cs_dac_06bit_ngspice.cir'  # circuit description
            spicef = 'cs_dac_06bit_ngspice_batch.cir'  # complete spice input file

            outputf = 'cs_dac_16bit_ngspice_batch_' + str(seq)
            
            ctrl_str = '\n' + '.save v(outf)' + '\n' + '.tran 10u ' + str(t[-1]) + '\n'

        case qws.w_16bit_SPICE:  # 16 bit DAC
            c = C.astype(int)
            nsamples = len(c)

            t1 = '\n'
            t2 = '\n'
            vbpc = '0'
            vdd = '1.5'
            Tr = 1e-3  # the rise-time for edges, in µs
            for k in range(0, Nb):  # generate PWL strings
                k_str = str(k+1)
                t1 += "vb" + k_str + " b" + k_str + " 0 pwl " + \
                    get_pwl_string(c, Ts, nsamples, k, vbpc, vdd, Tr)
                t2 += "vbb" + k_str + " bb" + k_str + " 0 pwl " + \
                    get_inverted_pwl_string(c, Ts, nsamples, k, vbpc, vdd, Tr)
            wav_str = t1 + t2

            seed_str = ''
            if seed == 1:
                seed_str = 'seed_1'
            elif seed == 2:
                seed_str = 'seed_2'

            # circuit description file
            circf = 'cs_dac_16bit_ngspice_' + seed_str + '.cir'
            # spice input file
            spicef = 'cs_dac_16bit_ngspice_batch_' + str(seq) + '.cir'

            # ctrl_str = '\n' + '.save v(out)' + '\n' + '.tran 10u ' + str(t[-1]) + '\n'

            outputf = 'cs_dac_16bit_ngspice_batch_' + str(seq)
            
            ctrl_str = ''

            if seed == 1:
                #ctrl_str = '\n.option method=trap XMU=0.495 gmin=1e-19 reltol=200u abstol=100f vntol=100n seed=1\n'
                ctrl_str = '\n.option method=trap TRTOL=5 gmin=1e-19 reltol=200u abstol=100f vntol=100n seed=1\n'
            elif seed == 2:
                #ctrl_str = '\n.option method=trap XMU=0.495 gmin=1e-19 reltol=200u abstol=100f vntol=100n seed=2\n'
                ctrl_str = '\n.option method=trap TRTOL=5 gmin=1e-19 reltol=200u abstol=100f vntol=100n seed=2\n'
            
            ctrl_str = ctrl_str + \
                '\n.control\n' + \
                'tran 10u ' + str(t[-1]) + '\n' + \
                'write $inputdir/' + outputf + '.bin' + ' v(out)\n' + \
                '.endc\n'
        case qws.w_6bit_2ch_SPICE:  # 6 bit DAC, 2 channels
            c1 = C[0,:].astype(int)
            c2 = C[1,:].astype(int)
            nsamples1 = len(c1)
            nsamples2 = len(c2)

            tvb1 = '\n'
            tvb2 = '\n'
            tvbb1 = '\n'
            tvbb2 = '\n'
            vbpc = '0'
            vdd = '1.5'
            Tr = 1e-3  # the rise-time for edges, in µs
            for k in range(0, Nb):  # generate PWL strings
                k_str = str(k + 1)
                tvb1 += 'vb1' + k_str + ' b1' + k_str + ' 0 pwl ' + \
                    get_pwl_string(c1, Ts, nsamples1, k, vbpc, vdd, Tr)
                tvbb1 += 'vbb1' + k_str + ' bb1' + k_str + ' 0 pwl ' + \
                    get_inverted_pwl_string(c1, Ts, nsamples1, k, vbpc, vdd, Tr)
                tvb2 += 'vb2' + k_str + ' b2' + k_str + ' 0 pwl ' + \
                    get_pwl_string(c2, Ts, nsamples2, k, vbpc, vdd, Tr)
                tvbb2 += 'vbb2' + k_str + ' bb2' + k_str + ' 0 pwl ' + \
                    get_inverted_pwl_string(c2, Ts, nsamples2, k, vbpc, vdd, Tr)
            wav_str = tvb1 + tvbb1 + tvb2 + tvbb2

            circf = 'cs_dac_06bit_2ch_TRAN.cir'  # circuit description
            spicef = 'cs_dac_06bit_2ch_TRAN_ngspice_batch.cir'  # complete spice input file

            outputf = 'cs_dac_06bit_2ch_TRAN_ngspice_batch'
            
            ctrl_str = '\n.option method=trap TRTOL=5 gmin=1e-19 reltol=200u abstol=100f vntol=100n seed=2\n'
            ctrl_str = ctrl_str + \
                '\n.control\n' + \
                'tran 10u ' + str(t[-1]) + '\n' + \
                'write $inputdir/' + outputf + '.bin' + ' v(out1) v(out2)\n' + \
                '.endc\n'
        case qws.w_16bit_2ch_SPICE:  # 16 bit DAC, 2 channels
            c1 = C[0,:].astype(int)
            c2 = C[1,:].astype(int)
            nsamples1 = len(c1)
            nsamples2 = len(c2)

            tvb1 = '\n'
            tvb2 = '\n'
            tvbb1 = '\n'
            tvbb2 = '\n'
            vbpc = '0'
            vdd = '1.5'
            Tr = 1e-3  # the rise-time for edges, in µs
            for k in range(0, Nb):  # generate PWL strings
                k_str = str(k + 1)
                tvb1 += 'vb1' + k_str + ' b1' + k_str + ' 0 pwl ' + \
                    get_pwl_string(c1, Ts, nsamples1, k, vbpc, vdd, Tr)
                tvbb1 += 'vbb1' + k_str + ' bb1' + k_str + ' 0 pwl ' + \
                    get_inverted_pwl_string(c1, Ts, nsamples1, k, vbpc, vdd, Tr)
                tvb2 += 'vb2' + k_str + ' b2' + k_str + ' 0 pwl ' + \
                    get_pwl_string(c2, Ts, nsamples2, k, vbpc, vdd, Tr)
                tvbb2 += 'vbb2' + k_str + ' bb2' + k_str + ' 0 pwl ' + \
                    get_inverted_pwl_string(c2, Ts, nsamples2, k, vbpc, vdd, Tr)
            wav_str = tvb1 + tvbb1 + tvb2 + tvbb2

            circf = 'cs_dac_16bit_2ch_TRAN.cir'  # circuit description
            spicef = 'cs_dac_16bit_2ch_TRAN_ngspice_batch.cir'  # complete spice input file

            outputf = 'cs_dac_16bit_2ch_TRAN_ngspice_batch'
            
            ctrl_str = '\n.option method=trap TRTOL=5 gmin=1e-19 reltol=200u abstol=100f vntol=100n seed=1\n'
            ctrl_str = ctrl_str + \
                '\n.control\n' + \
                'tran 10u ' + str(t[-1]) + '\n' + \
                'write $inputdir/' + outputf + '.bin' + ' v(out1) v(out2)\n' + \
                '.endc\n'

    addtexttofile(os.path.join(tempdir, cmdf), ctrl_str)

    addtexttofile(os.path.join(tempdir, wavf), wav_str)

    with open(os.path.join(outdir, spicef), 'w') as fout:
        fins = [os.path.join(circdir, circf),
                os.path.join(tempdir, cmdf),
                os.path.join(tempdir, wavf)]
        fin = fileinput.input(fins)
        for line in fin:
            fout.write(line)
        fin.close()

    print(circf)
    print(spicef)
    print(outputf)
    
    return spicef, outputf


def read_spice_bin_file(fdir, fname):
    """
    Read a given ngspice binary output file.
    Accounts for number of variables.
    Assumes variables are interleaved, with time vector first.
    """

    fpath = os.path.join(fdir, fname)
    fid = open(fpath, 'rb')
    # print("Opening file: " + fname)

    read_new_line = True
    while read_new_line:
        tline = fid.readline()

        if b'Binary:' in tline:  # marker for binary data to start
            read_new_line = False

        if b'No. Variables: ' in tline:
            nvars = int(tline.split(b':')[1])
            print(nvars)

        if b'No. Points: ' in tline:
            npoints = int(tline.split(b':')[1])
            print(npoints)
    
    data = np.fromfile(fid, dtype='float64')
    t_spice = np.array(data[::nvars])
    y_spice = np.zeros((nvars-1,npoints))
    for k in range(1, nvars):
        y_spice[k-1:] = data[k::nvars]
    
    return t_spice, y_spice


def read_spice_bin_file_with_most_recent_timestamp(fdir):
    """
    Read SPICE ouput file (assuming a certain format, i.e. not general)
    """

    binfiles = [file for file in os.listdir(fdir) if file.endswith('.bin')]
    binfiles.sort()
    fname = binfiles[-1]
    
    t_spice, y_spice = read_spice_bin_file(fdir, fname)

    return t_spice, y_spice


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


def main():
    """
    Read results from a given SPICE simulation and process the data.
    """
    outdir = 'spice_output'

    rundirs = os.listdir(outdir)
    rundirs.sort()

    print('No. dirs.: ' + str(len(rundirs)))

    method_str = 'baseline'
    #method_str = 'physical_level_calibration'
    #method_str = 'periodic_dither'
    #method_str = 'noise_dither'
    #method_str = 'digital_calibration'
    #method_str = 'dynamic_element_matching'
    method_str = 'ilc'

    matching = [s for s in rundirs if method_str in s]

    #rundir = rundirs[16]  # pick run
    rundir = matching[0]  # pick run
    
    bindir = os.path.join(outdir, rundir)

    # read pickled (marshalled) state/config object
    with open(os.path.join(bindir, 'sim_config.pickle'), 'rb') as fin:
        SC = pickle.load(fin)
    
    print(bindir)

    run_info = [['Method', 'Model', 'Fs', 'Fc', 'Fx'],
                [str(SC.lin), str(SC.dac), f'{Float(SC.fs):.0h}', f'{Float(SC.fc):.0h}', f'{Float(SC.carrier_freq):.1h}']]

    print(tabulate(run_info))

    binfiles = [file for file in os.listdir(bindir) if file.endswith('.bin')]
    binfiles.sort()

    if True:
        Nbf = len(binfiles)  # number of bin (binary data) files

        t = SC.t
        Fs = SC.fs
        Fx = SC.carrier_freq

        t_end = 3/Fx  # time vector duration
        Fs_ = Fs*72  # semi-optimal factor for most sims with different non-uniform sampling per file
        Fs_ = Fs
        print(f'Fs: {Float(Fs):.0h}')
        t_ = np.arange(0, t_end, 1/Fs_)  # time vector
        
        if Nbf == 1:  # may contain several channels in ngspice bin file
            print(os.path.join(bindir, binfiles[0]))
            t_spice, y_spice = read_spice_bin_file(bindir, binfiles[0])
            Nch = y_spice.shape[0]
            print('No. channels:')
            print(Nch)

            # Summation stage
            if SC.lin.method == lm.BASELINE or SC.lin.method == lm.ILC:
                K = np.ones((Nch,1))
                K[1] = 0.0  # null one channel (want single channel resp.)
            elif SC.lin.method == lm.DEM:
                K = np.ones((Nch,1))
            elif SC.lin.method == lm.PHYSCAL:
                K = np.ones((Nch,1))
                K[1] = 1e-2
            else:
                K = 1/Nch
            
            print('Summing gain:')
            print(K)

            y_spice_ = np.sum(K*y_spice, 0)
            ym_ = np.interp(t_, t_spice, y_spice_)  # re-sample

        else:  # assume one channel per bin file
            Nch = Nbf
            YM = np.zeros([Nch, t_.size])
            for k in range(0, Nbf):
                print(os.path.join(bindir, binfiles[k]))
                t_spice, y_spice = read_spice_bin_file(bindir, binfiles[k])
                y_resamp = np.interp(t_, t_spice, y_spice)  # re-sample
                YM[k,:] = y_resamp

                # Summation stage
                if SC.lin.method == lm.DEM:
                    K = np.ones((Nch,1))
                if SC.lin.method == lm.PHYSCAL:
                    K = np.ones((Nch,1))
                    K[1] = 1e-2
                else:
                    K = 1/Nch

                ym_ = np.sum(K*YM, 0)

        if False:
            #ym = np.interp(t, t_, ym_)  # re-sample
            #ym = interpolate.Akima1DInterpolator(t_, ym_)(t)
            ym = interpolate.PchipInterpolator(t_, ym_)(t)
            #ym = signal.resample(ym_, t.size)
            TRANSOFF = np.floor(0.5*Fs/Fx).astype(int)  # remove transient effects from output
        else:
            ym = ym_
            t = t_
            TRANSOFF = np.floor(0.25*Fs_/Fx).astype(int)  # remove transient effects from output

        Fc = SC.fc
        Nf = SC.nf

        ym_avg, ENOB_M = process_sim_output(t, ym, Fc, Fs_, Nf, TRANSOFF, sinad_comp.CFIT, False, 'SPICE')

        plt.plot(t,ym)
        plt.plot(t,ym_avg)
        
        results_tab = [['Config', 'Method', 'Model', 'Fs', 'Fc', 'Fx', 'ENOB'],
            [str(SC.qconfig), str(SC.lin), str(SC.dac), f'{Float(SC.fs):.2h}', f'{Float(SC.fc):.1h}', f'{Float(SC.carrier_freq):.1h}', f'{Float(ENOB_M):.3h}']]
        print(tabulate(results_tab))
        
        #t_spice, y_spice = read_spice_bin_file_with_most_recent_timestamp(path)
        #YM = np.zeros([1,y_spice.size])
        #YM[0,:] = y_spice
        #plt.plot(y_spice)
        #print(YM)


if __name__ == "__main__":
    main()
