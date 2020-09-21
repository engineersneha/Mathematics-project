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
# warnings.filterwarnings("ignore")

def rhs_lorenz(t, xyz, sigma=10., beta=8/3, rho=28.):
	x, y, z = xyz
	xyz_t = [
		sigma * (y - x),
		x * (rho - z) - y,
		x * y - beta * z
	]
	return xyz_t

def main():
	np.random.seed(123)

	# Parameters for the Lorenz system
	x0 = [0., 1., 20.]
	t0 = 0
	tN = 50
	dt = 0.001
	t  = np.arange(0, tN+dt, dt)
	sigma = 10
	beta  = 8/3
	rho   = 28

	# Solve ODE problem
	start = time.time()
	ODE = OdeIntegration(rhs_lorenz, x0, t, params=(sigma,beta,rho), backend='scipy_odeint')
	t1, sol1 = ODE.solve()
	x1, y1, z1 = sol1[:,0], sol1[:,1], sol1[:,2]
	print('Elapsed time scipy_odeint: ', time.time() - start, 's.')

	start = time.time()
	ODE = OdeIntegration(rhs_lorenz, x0, t, params=(sigma,beta,rho), backend='scipy_ode', approach='dopri5')
	t2, sol2 = ODE.solve()
	x2, y2, z2 = sol2[:,0], sol2[:,1], sol2[:,2]
	print('Elapsed time scipy_ode: ', time.time() - start, 's.')

	start = time.time()
	ODE = OdeIntegration(rhs_lorenz, x0, t, params=(sigma,beta,rho), backend='scipy_solve_ivp', approach='LSODA')
	t3, sol3 = ODE.solve()
	x3, y3, z3 = sol3[:,0], sol3[:,1], sol3[:,2]
	print('Elapsed time scipy_solve_ivp: ', time.time() - start, 's.')

	import matplotlib.pyplot as plt
	fig, axs = plt.subplots(1, 3, constrained_layout=True, subplot_kw={'projection': '3d'})
	axs[0].plot(x1, y1, z1,linewidth=1)
	axs[0].scatter(x0[0],x0[1],x0[2],color='r')
	axs[0].grid()
	axs[0].view_init(18, -113)
	axs[1].plot(x2, y2, z2,linewidth=1)
	axs[1].scatter(x0[0],x0[1],x0[2],color='r')
	axs[1].grid()
	axs[1].view_init(18, -113)
	axs[2].plot(x3, y3, z3,linewidth=1)
	axs[2].scatter(x0[0],x0[1],x0[2],color='r')
	axs[2].grid()
	axs[2].view_init(18, -113)
	plt.show()
	plt.savefig('lorenz.png')

if __name__ == "__main__":
	main()
