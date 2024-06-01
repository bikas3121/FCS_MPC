# Linearisation of the Digital to Analog Controller using MPC

1. [Introduction](#introduction)
2. [Moving Horizon Optimal Quantiser](#moving-horizon-optimal-quantiser)
3. [Simulation](#Simulation)

## Introduction

In this repository you find an implementation of the moving horizon optimal quantiser (MHOQ) based on the paper [Linearisation of Digital-to-Analog Converters by Model Predictive Control, Adhikari et. al](https://engrxiv.org/preprint/view/3551). The paper is accepted for publication in [Nonlinear Model Prediction Control NMPC 2024](https://nmpc2024.org/). The details regarding problem formulation, simulation results and test results of the alogrithm can be found in the mentioned paper. The paper is also included in the repository.  


## Moving Horizon Optimal Quantiser (MHOQ)

The MHOQ is based on the Linear Time-Invariant modelling of the Digital-to-Analog Converter (DAC). The problem of finding the optimal quantiser levels is formulated into the error minimization problem where the filtered quantised value tracks reference signal. The MPC formulation translates to well-know Finite Constraint Set MPC.

The algorithm is implemented in both matlab and python. We have mainly used [Gurobi](https://www.gurobi.com/) as the solver in the python. But in the MATLAB, we have used [YALMIP](https://yalmip.github.io/) modelling tool and we also provide the option to choose between the solvers. 

The implementation is based on a small set of libraries mentioned as follows
```
numpy
scipy
matplotlib
statistics
itertools
math    
```

### Simulation
To start the simulation, go to ```main.py```. 


