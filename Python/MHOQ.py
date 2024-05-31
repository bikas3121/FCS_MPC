#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Moving horizon implementation of the quantisation 

@author: Bikash Adhikari
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
from scipy import linalg , signal
import sys
import random
import gurobipy as gp
from gurobipy import GRB
import tqdm


class MHOQ:

    def __init__(self, Nb, Qstep, QMODEL,  A, B, C, D):
        """
        Constructor for the Model Predictive Controller.
        :param Nb: Number of bits 
        :param Qstep: Quantizer step size / Least significant bit (LSB) 
        :param QMODEL: Quantiser model, Ideal or Measured 
        :param A, B, C, D: Matrices; state space representation of the reconstruction filter
        """
        self.Nb = Nb
        self.Qstep = abs(Qstep)
        self.QMODEL = QMODEL
        self.A = A
        self.B = B
        self.C = C
        self.D = D
    
        
    def state_prediction(self, st, con):
        """Predict the state for the given initial condition and control
        INPUT
            :state
            :control
        OUTPUT
            :predicted state
        """
        x_iplus1 = self.A @ st + self.B * con
        return x_iplus1

     
    def get_codes(self, N_PRED, Xcs, YQns, MLns ):
        """ Performs the moving horizon implementation and returns DAC codes (non-negative integers) [Optimization problem with integer variables]
        INPUT
            :N_PRED - prediction horizon
            :Xcs    -  reference/carrier signal
            :YQns   - Ideal levels
            :MLns   - Measured levels
        OUTPUT
            :codes
        """

        match self.QMODEL:
            case 1:
                QLS = YQns
            case 2:
                QLS = MLns

        C = []

        # Loop length
        len_MPC = Xcs.size - N_PRED

        # State dimension
        x_dim =  int(self.A.shape[0]) 

        # Initial state
        init_state = np.zeros(x_dim).reshape(-1,1)

        # MPC loop
        for j in tqdm.tqdm(range(len_MPC)):

            m = gp.Model("MPC- INL")
            u = m.addMVar(N_PRED, vtype=GRB.INTEGER, name= "u", lb = 0, ub =  2**self.Nb-1) # control variable
            x = m.addMVar((x_dim*(N_PRED+1),1), vtype= GRB.CONTINUOUS, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "x")  # State varible 


            # Add objective function
            Obj = 0

            # Set initial constraint
            m.addConstr(x[0:x_dim,:] == init_state)
            for i in range(N_PRED):
                k = x_dim * i
                st = x[k:k+x_dim]
                con = u[i] - Xcs[j+i]

                # Objective update
                e_t = self.C @ x[k:k+x_dim] + self.D * con
                Obj = Obj + e_t * e_t

                # Constraints update
                f_value = self.A @ st + self.B * con
                st_next = x[k+x_dim:k+2*x_dim]
                m.addConstr(st_next == f_value)

            # Gurobi model update
            m.update

            # Set Gurobi objective
            m.setObjective(Obj, GRB.MINIMIZE)

            # 0 - Supress log output, 1- Print log outputs
            m.Params.LogToConsole = 0

            # Gurobi setting for precision  
            # m.Params.IntFeasTol = 1e-9
            # m.Params.IntegralityFocus = 1

            # Optimization 
            m.optimize()

            # Extract variable values 
            allvars = m.getVars()
            values = m.getAttr("X",allvars)
            values = np.array(values)

            # Extract only the value of the variable "u", value of the variable "x" are not needed
            C_MPC = values[0:N_PRED]

            # Ideally they should be integral, but gurobi generally return them in floating point values according to the precision tolorence set: m.Params.IntFeasTol
            # Round off to nearest integers
            C_MPC = C_MPC.astype(int)

            # Store only the first value /code
            C.append(C_MPC[0])

            # Get DAC level according to the coe
            U_opt = QLS[0,C_MPC[0]] 

            # State prediction 
            con = U_opt - Xcs[j]
            x0_new = self.state_prediction(init_state, con)

            # State update for subsequent prediction horizon 
            init_state = x0_new

        return np.array(C).reshape(1,-1)




# %% MPC with BINARY FORMULATION (Optimization problem)


class MPC_BIN:
    def __init__(self, Nb, Qstep, QMODEL,  A, B, C, D):

        """
        Constructor for the Model Predictive Controller.
        :param Nb: Number of bits 
        :param Qstep: Quantizer step size / Least significant bit (LSB) 
        :param N_PRED: Prediction horizon | int 
        :param Xcs: Reference/Test signal 
        :param QL: Quantization levels 
        :param A, B, C, D: Matrices; state space representation of the reconstruction filter
        """
        self.Nb = Nb
        self.Qstep = abs(Qstep)
        # self.N_PRED = N_PRED
        # self.Xcs = Xcs
        # self.QL = QL.reshape(1,-1)
        self.QMODEL = QMODEL
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        # self.x0 = x0
    
        
    def state_prediction(self, st, con):
        """
        Predict the state for the given initial condition and control
        """
        x_iplus1 = self.A @ st + self.B * con
        return x_iplus1
    
    # def get_codes(self, Xcs, N_PRED, YQns, MLns)
    def get_codes(self, N_PRED, Xcs, YQns, MLns ):
        """ Performs the moving horizon implementation and returns DAC codes (non-negative integers) [Optimization problem with binary variables]
        INPUT
            :N_PRED - prediction horizon
            :Xcs    -  reference/carrier signal
            :YQns   - Ideal levels
            :MLns   - Measured levels
        OUTPUT
            :codes
        """

        match self.QMODEL:
            case 1:
                QL = YQns
            case 2:
                QL = MLns
        # Storage container for code
        C = []

        # Loop length
        len_MPC = Xcs.size - N_PRED

        # State dimension
        x_dim =  int(self.A.shape[0]) 

        # Initial state
        init_state = np.zeros(x_dim).reshape(-1,1)

        # MPC loop
        for j in tqdm.tqdm(range(len_MPC)):

            m = gp.Model("MPC- INL")
            u = m.addMVar((2**self.Nb, N_PRED), vtype=GRB.BINARY, name= "u") # control variable
            x = m.addMVar((x_dim*(N_PRED+1),1), vtype= GRB.CONTINUOUS, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "x")  # State varible 


            # Add objective function
            Obj = 0

            # Set initial constraint
            m.addConstr(x[0:x_dim,:] == init_state)
            for i in range(N_PRED):
                k = x_dim * i
                st = x[k:k+x_dim]
                bin_con =  QL.reshape(1,-1) @ u[:,i].reshape(-1,1) 
                con = bin_con - Xcs[j+i]

                # Objective update
                e_t = self.C @ x[k:k+x_dim] + self.D * con
                Obj = Obj + e_t * e_t

                # Constraints update
                f_value = self.A @ st + self.B * con
                st_next = x[k+x_dim:k+2*x_dim]
                m.addConstr(st_next == f_value)

                # Binary varialble constraint
                consi = gp.quicksum(u[:,i]) 
                m.addConstr(consi == 1)
        # m.addConstr(consi >= 0.98)
        # m.addConstr(consi <= 1.02
            # Gurobi model update
            m.update

            # Set Gurobi objective
            m.setObjective(Obj, GRB.MINIMIZE)

            # 0 - Supress log output, 1- Print log outputs
            m.Params.LogToConsole = 0

            # Gurobi setting for precision  
            m.Params.IntFeasTol = 1e-9
            m.Params.IntegralityFocus = 1

            # Optimization 
            m.optimize()

            # Extract variable values 
            allvars = m.getVars()
            values = m.getAttr("X",allvars)
            values = np.array(values)

            # Variable dimension
            nr, nc = u.shape
            u_val = values[0:nr*nc]
            u_val = np.reshape(u_val, (2**self.Nb, N_PRED))

            # Extract Code
            C_MPC = []
            for i in range(N_PRED):
                c1 = np.nonzero(u_val[:,i])[0][0]
                c1 = int(c1)
                C_MPC.append(c1)
            C_MPC = np.array(C_MPC)
            C.append(C_MPC[0])

            U_opt = QL[C_MPC[0]] 
            # # Extract only the value of the variable "u", value of the variable "x" are not needed
            # C_MPC = values[0:N_PRED]

            # # Ideally they should be integral, but gurobi generally return them in floating point values according to the precision tolorence set: m.Params.IntFeasTol
            # # Round off to nearest integers
            # C_MPC = C_MPC.astype(int)

            # # Store only the first value /code
            # C.append(C_MPC[0])

            # # Get DAC level according to the coe
            # U_opt = QLS[C_MPC[0]] 

            # State prediction 
            con = U_opt - Xcs[j]
            x0_new = self.state_prediction(init_state, con)

            # State update for subsequent prediction horizon 
            init_state = x0_new

        return np.array(C).reshape(1,-1)
