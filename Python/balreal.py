#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Balanced realisaton of a stable LTI state-space system.

@author: Arnfinn Eielsen
@date: 06.03.2024
@license: BSD 3-Clause
"""

import numpy as np
from scipy import linalg


def balreal(A,B,C,D):
    """
    Straight forward implementation of a Gramian-based balanced realisation
    using SciPy linear algebra library.

    This is for discrete time systems.

    [1] A. Laub, M. Heath, C. Paige, and R. Ward, 
    ‘Computation of System Balancing Transformations and Other Applications of Simultaneous Diagonalization Algorithms’,
    IEEE Transactions on Automatic Control, vol. AC-32, no. 2, pp. 115–122, Feb. 1987.
    """

    Wr = linalg.solve_discrete_lyapunov(A, B@B.T)
    Wo = linalg.solve_discrete_lyapunov(A.T, C.T@C)#, method=None)

    Lr = linalg.cholesky(Wr, lower=True)
    Lo = linalg.cholesky(Wo, lower=True)

    U, s, Vh = linalg.svd(Lo.T@Lr)
    S = linalg.diagsvd(s, A.shape[0], A.shape[1])
    
    T = Lr@Vh.T@linalg.sqrtm(linalg.inv(S))

    A_ = linalg.inv(T)@A@T
    B_ = linalg.inv(T)@B
    C_ = C@T
    D_ = D

    return A_, B_, C_, D_



def balreal_ct(A,B,C,D):
    """
    Straight forward implementation of a Gramian-based balanced realisation
    using SciPy linear algebra library.

    This is for continuous time systems.

    [1] A. Laub, M. Heath, C. Paige, and R. Ward, 
    ‘Computation of System Balancing Transformations and Other Applications of Simultaneous Diagonalization Algorithms’,
    IEEE Transactions on Automatic Control, vol. AC-32, no. 2, pp. 115–122, Feb. 1987.
    """

    Wr = linalg.solve_continuous_lyapunov(A, -B@B.T)
    Wo = linalg.solve_continuous_lyapunov(A.T, -C.T@C)

    Lr = linalg.cholesky(Wr, lower=True)
    Lo = linalg.cholesky(Wo, lower=True)

    U, s, Vh = linalg.svd(Lo.T@Lr)
    S = linalg.diagsvd(s, A.shape[0], A.shape[1])
    
    T = Lr@Vh.T@linalg.sqrtm(linalg.inv(S))

    A_ = linalg.inv(T)@A@T
    B_ = linalg.inv(T)@B
    C_ = C@T
    D_ = D

    return A_, B_, C_, D_



def main():
    """
    Test the method.
    """

    from scipy import signal
    import control as ct
    from matplotlib import pyplot as plt

    match 2:
        case 1:
            if False:
                sys = ct.drss(4, outputs=1, inputs=1)  # random, stable LTI system
            else:
                b, a = signal.butter(3, 0.25, 'lowpass', analog=False)
                Wlp = signal.dlti(b, a, dt=1)  # filter LTI system instance
                sys = Wlp.to_ss()

            A = sys.A
            B = sys.B
            C = sys.C
            D = sys.D

            A_, B_, C_, D_ = balreal(A,B,C,D)

            sys_init = signal.dlti(A, B, C, D, dt=1)
            sys_bal = signal.dlti(A_, B_, C_, D_, dt=1)

            ti, yi = signal.dlti.step(sys_init)
            tb, yb = signal.dlti.step(sys_bal)
            yi = yi[0]
            yb = yb[0]
        case 2:
            if False:
                sys = ct.rss(4, outputs=1, inputs=1)  # random, stable LTI system
                A = sys.A
                B = sys.B
                C = sys.C
                D = sys.D
            else:
                Wn = 2*np.pi*1000e3
                b, a = signal.butter(3, 1, 'lowpass', analog=True)
                Wlp = signal.lti(b, a)  # filter LTI system instance
                sys = Wlp.to_ss()

                A = sys.A
                B = sys.B
                C = sys.C
                D = sys.D
                
                A = Wn*A  # scale to get correct cut-off
                B = Wn*B
                
            A_, B_, C_, D_ = balreal_ct(A,B,C,D)

            sys_init = signal.lti(A, B, C, D)
            sys_bal = signal.lti(A_, B_, C_, D_)

            ti, yi = signal.lti.step(sys_init)
            tb, yb = signal.lti.step(sys_bal)

    plt.plot(ti, yi)
    plt.plot(tb, yb)

    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Compare realisations')
    plt.grid()


if __name__ == "__main__":
    main()
