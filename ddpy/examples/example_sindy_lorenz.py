import __includes__

import sys
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from dynamics.ode_solver import OdeIntegration
# warnings.filterwarnings("ignore")

def rhs_pendulum(t, x, b, c):
    theta, omega = x
    dxdt = [omega, -b * omega - c * np.sin(theta)]
    return dxdt

def rhs_lorenz(t, xyz, sigma=10., beta=8/3, rho=28.):
	x, y, z = xyz
	xyz_t = [
		sigma * (y - x),
		x * (rho - z) - y,
		x * y - beta * z
	]
	return xyz_t

def poolData(yin, nVars, polyorder):
	n = yin.shape[0]
	yout = np.zeros((n,1))

	# poly order 0
	yout[:,0] = np.ones(n)

	# poly order 1
	for i in range(nVars):
		yout = np.append(yout, yin[:,i].reshape((yin.shape[0],1)), axis=1)

	# poly order 2
	if polyorder >= 2:
		for i in range(nVars):
			for j in range(i,nVars):
				yout = np.append(yout, (yin[:,i]*yin[:,j]).reshape((yin.shape[0],1)), axis=1)

	# poly order 3
	if polyorder >= 3:
		for i in range(nVars):
			for j in range(i,nVars):
				for k in range(j,nVars):
					yout = np.append(yout, (yin[:,i]*yin[:,j]*yin[:,k]).reshape((yin.shape[0],1)), axis=1)

	# poly order 4
	if polyorder >= 4:
		for i in range(nVars):
			for j in range(i,nVars):
				for k in range(j,nVars):
					for m in range(k,nVars):
						yout = np.append(yout, (yin[:,i]*yin[:,j]*yin[:,k]*yin[:,m]).reshape((yin.shape[0],1)), axis=1)

	return yout

def sparsifyDynamics(Theta, dXdt, lamb, n):
	# Initial guess: Least-squares
	Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]
	for k in range(10):
		smallinds = np.abs(Xi) < lamb # Find small coefficients
		Xi[smallinds] = 0                          # and threshold
		for ind in range(n):                       # n is state dimension
			biginds = smallinds[:,ind] == 0
			# Regress dynamics onto remaining terms to find sparse Xi
			Xi[biginds,ind] = np.linalg.lstsq(Theta[:,biginds], dXdt[:,ind], rcond=None)[0]
	return Xi



def main():
	np.random.seed(123)

	# Parameters for the Lorenz system
	x0 = [np.pi, 1.]
	t0 = 0
	tN = 50
	dt = 0.001
	t  = np.arange(0, tN+dt, dt)
	sigma = 10
	beta  = 8/3
	rho   = 28
	b = 1
	c = 5

	# Solve ODE problem
	start = time.time()
	ODE = OdeIntegration(rhs_pendulum, x0, t, params=(b,c), backend='scipy_odeint')
	t1, sol1 = ODE.solve()
	x1, y1 = sol1[:,0], sol1[:,1]
	print('Elapsed time scipy_odeint: ', time.time() - start, 's.')


	# SINDy algorithm
	x = sol1
	dx = np.zeros_like(x)
	for j in range(len(t)):
		dx[j,:] = rhs_pendulum(t, x[j,:], b, c)
	Theta = poolData(x, x.shape[1], 4) # Up to third order polynomials
	lamb  = 0.025            		   # sparsification knob lambda
	Xi = sparsifyDynamics(Theta, dx, lamb, x.shape[1])
	print(Xi)

if __name__ == "__main__":
	main()
