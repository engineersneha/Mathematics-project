#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    This file is subject to the terms and conditions defined in
    file 'LICENSE.txt', which is part of this source code package.

    Written by Dr. Gianmarco Mengaldo, May 2020.
'''
import __includes__

import sys
import time
import warnings
import numpy as np
from dynamics.ode_solver import OdeIntegration
warnings.filterwarnings("ignore")



def rhs(t, x, b, c):
    theta, omega = x
    dxdt = [omega, -b * omega - c * np.sin(theta)]
    return dxdt

def simple_example():
    '''
        single pendulum subject to gravity and damping
    '''
    x0 = [np.pi - 0.1, 0.0]
    b  = 0.25
    c  = 5.00
    t0 = 0
    tN = 100
    N  = 100000
    t  = np.linspace(t0, tN, N)

    # 3 different methods for ODE solutions
    start = time.time()
    ODE = OdeIntegration(rhs, x0, t, params=(b,c), backend='scipy_odeint')
    t1, sol1 = ODE.solve_ode()
    print('Elapsed time scipy_odeint: ', time.time() - start, 's.')

    start = time.time()
    ODE = OdeIntegration(rhs, x0, t, params=(b,c), backend='scipy_ode', approach='dopri5')
    t2, sol2 = ODE.solve_ode()
    print('Elapsed time scipy_ode: ', time.time() - start, 's.')

    start = time.time()
    ODE = OdeIntegration(rhs, x0, t, params=(b,c), backend='scipy_solve_ivp', approach='LSODA')
    t3, sol3 = ODE.solve_ode()
    print('Elapsed time scipy_solve_ivp: ', time.time() - start, 's.')

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(16,8))
    axs[0].plot(t1, sol1[:, 0], 'b', label='theta(t)')
    axs[0].plot(t1, sol1[:, 1], 'g', label='omega(t)')
    axs[0].legend(loc='best')
    axs[0].set_xlabel('t')
    axs[0].grid()
    axs[1].plot(t2, sol2[:, 0], 'b', label='theta(t)')
    axs[1].plot(t2, sol2[:, 1], 'g', label='omega(t)')
    axs[1].legend(loc='best')
    axs[1].set_xlabel('t')
    axs[1].grid()
    axs[2].plot(t3, sol3[:, 0], 'b', label='theta(t)')
    axs[2].plot(t3, sol3[:, 1], 'g', label='omega(t)')
    axs[2].legend(loc='best')
    axs[2].set_xlabel('t')
    axs[2].grid()
    plt.show()

if __name__ == "__main__":
    simple_example()
