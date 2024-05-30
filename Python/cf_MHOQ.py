
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Implementation of explicit MPC: Linearization of digital to analog converter 
Closed form solution of the MPC based on the paper of Goodwin et. al (2003)
@author: Bikash Adhikari
@date: 24.09.2023
@license: BSD 3-Clause
"""

# imports
import numpy as np
import itertools
import tqdm


class cf_MHOQ:
      
    def __init__(self, N, Q, A, B, C, h):
        ''''''''''''''''''''''''''''''''''
        Parameters for MPC:
            N : MPC prediction horizon
            Q : DAC levels (Constraint set)
            A : State matrix 
            B : Input Matrix
            C,D: Output Matrix
            h - impulse response of the filter (low pass)
        '''''''''''''''''''''''''''''''''

        self.N = N  # prediction horizon
        self.Q = Q.squeeze()    # Quantiser levels
        self.A = A  # state matrix of the lowpass filter
        self.B = B # input matrix
        self.C = C  # output matrix of low pass filter
        self.h = h  # impulse response of the low pass filter
        


    # Returns matrix Tau
    @property
    def Tau(self):
        Tau = self.C
        for i in range(1,self.N):
            A_ast = np.linalg.matrix_power(self.A, i)
            # Tau_i = np.matmul(self.C, A_ast)
            Tau_i = self.C@A_ast
            Tau = np.vstack((Tau, Tau_i[0]))
        return Tau

    
    # Returns matrix Psi 
    @property
    def Psi(self):
        Psi = np.zeros(shape = (self.N,self.N))
        for i in range(0,self.N):    
            k = i
            for j in range(0, i+1):
                Psi[i][j] = self.h[k]
                k = k-1
        return Psi
        

    @property
    # Returns the cartesian product of the quantization leves with N x |U|^N matrix for vector quantisation
    def vectorQuantizationLevels(self):
        cartU = list(itertools.product(self.Q,repeat=self.N))
        lenU = len(cartU)

        # list to matrix
        Un = np.zeros((self.N,len(cartU)))
        for i in range(lenU):
            c1 = cartU[i]
            for j in range(self.N):
                Un[j,i] = c1[j] 
        return Un
            

    # Performs vector quantisation
    def vec_quant(self, inp, VQLS):
        """ Vector quantisation of the vector input

        Parameters:
        -----------
        inp     - Vector input
        VQLS    - Vector quantisation levels, obtained from cartesian products of scalar quantisation levels

        Returns:
        --------
        inp_vq  - vector quantised input
        """
        # Constrainted set length
        lpc = VQLS.shape[1] 

        # Store error norm
        err_nrm = []

        for i in range(lpc):
            err_nrm_i = np.linalg.norm(inp.reshape(1,-1) - VQLS[:,i].reshape(1,-1))
            err_nrm.append(err_nrm_i)

        # find index with minimum error norm
        index = np.where(err_nrm == np.min(err_nrm))

        # in case of multiple index, take the first one
        inp_vq = VQLS[:,index[0]]
        return inp_vq


    def get_codes(self, Xcs):
        ''' Moving horizon optimal quantisation using closed form solution

        Parameters:
        ----------
        Xcs     - Reference/test singal
        x0      - Initial condition

        Returns:
        ---------
        C      -  Optimal output
        '''

        Un_tilde = self.Psi@self.vectorQuantizationLevels
        Psi_inv = np.linalg.inv(self.Psi)
        x_init =   np.zeros((self.A.shape[0],1)) 

        Xcs_mhoq = []
        #loop length
        len_mhoq = len(Xcs)-self.N

        for i in tqdm.tqdm(range(0,len_mhoq)):

            a_k = Xcs[i:i+self.N].reshape(-1,1)
            sm = self.Psi@a_k + self.Tau@x_init
            sm = sm.reshape(-1,1) 

            #Vector quantisation
            sm_vq = self.vec_quant(sm, Un_tilde)

            # Reshape
            sm_opt= sm_vq.reshape(-1,1)

            # Optimal value
            Un_opt = Psi_inv@sm_vq

            # Store value
            Xcs_mhoq = np.hstack((Xcs_mhoq, Un_opt[0]))

            # State update
            con = Xcs[i] - Un_opt[0]
            x_init = self.A@x_init + self.B*con
        return np.round(Xcs_mhoq)
