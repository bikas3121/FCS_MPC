#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:38:25 2023
@author: bikashadhikari
"""

from datetime import datetime
import os
import csv

class writeCustomFilename:   
    def __init__(self, Sig_type, headerlist, datalist):

        """  Save data into file based on the signal type; DAC bits; and carrier, sampling
        and cutoff frequency of the filter
        :param sig_type:  filtered/unfiltered signal signal 
        :param headerlist: File header
        :param datalist: Data to be stored
        """
        self.Sig_type = Sig_type
        self.headerlist = headerlist
        self.datalist = datalist        


    def writeCfile(self, **kwargs):
        
        # INPUT: 
            # Sig_type :  Type of signal to be written; Filetered or Unfiltered
            # nob: number of bits, 
            # fs : signal frequency
            # Fc : cutoff frequency
            # Fs : Sampling frequency 
            
        # OUPUT:
            # write the file 
        current_datetime = datetime.now().strftime("%y%h%d_%H-%M")
        folder_name = str(self.Sig_type)
        kv_dict = kwargs
        key_name = list(kv_dict.keys())
        
        # File naming labels, number of bits (nob), signal frequency (fs), cutoff frequency (Fc) and sampling frequency (Fc)
        req_label = ['nob', 'fs', 'Fc', 'Fs']  
        if all(x in key_name for x in req_label) == False:
            # print('Not enough parameters to generate a filename')
            raise KeyError("Not enough parameters to generate a filename")
        nob = kv_dict.get('nob')
        fs = kv_dict.get('fs')
        Fc = kv_dict.get('Fc')
        Fs = kv_dict.get('Fs')
        folder_to_save_file = folder_name + str("_") + str(int(nob)) + str("_")+ str(int(fs))  +str("_")+ str(int(Fc)) + str("_")+ str(int(Fs))+ str("_")+current_datetime
        str_file_name = str(folder_to_save_file)
        file_name =str_file_name+".csv" 
        
        if not os.path.exists('SimulationData'):
            os.makedirs('SimulationData')
        path = r'SimulationData'
        
        with open(os.path.join(path,file_name),'w') as f1:
            writer = csv.writer(f1, delimiter ='\t')
            writer.writerow(self.headerlist)
            writer.writerows(self.datalist)
        
    
    