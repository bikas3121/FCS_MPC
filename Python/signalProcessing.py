#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Filter signal and calcualtes the error variance with respect to the reference siganl
Created on Tue Aug 29 14:35:58 2023
@author: bikashadhikari
"""

from scipy import signal
import statistics


class signalProcessing:    
    def __init__(self, ref, b, a, TRANSOFF):

        """ Filters signal and calculate the variance
        :param ref:  reference signal 
        :param b,a: filter transfer functions (numeration, denominator) 
        :param TRANSFOFF:  Transient length (remove after processing)
        """

        self.ref = ref # reference signal
        self.b =b  # transfer function numberator
        self.a = a # transfer function denominator
        self.TRANSOFF = TRANSOFF
    
    @property
    def referenceFilter(self):
        filteredReference = signal.lfilter(self.b, self.a, self.ref) 
        return filteredReference[self.TRANSOFF:]
    
    def signalFilter(self, sig):
        sig = sig.squeeze()
        sig_len = sig.size 
        filteredSignal = signal.lfilter(self.b, self.a, sig)
        filteredSignal = filteredSignal[self.TRANSOFF:]
        errorWRTreference = self.referenceFilter[0:len(filteredSignal)]-filteredSignal
        varianceError = round(statistics.variance(errorWRTreference.squeeze()),8)
        return [filteredSignal, errorWRTreference, varianceError]
        # return filteredSignal
    
               
 