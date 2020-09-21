import __includes__

import os
import sys
import time
import warnings
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from dynamics.ode_solver import OdeIntegration
from dynamics.dmd_solver import DMD_integration
# warnings.filterwarnings("ignore")
np.random.seed(123)

def f1(x,t): return 1. / np.cosh(x + 3) * np.exp(2.3j*t)
def f2(x,t): return 2. / np.cosh(x) * np.tanh(x) * np.exp(2.8j*t)



def main():

	# Define initial data
	x = np.linspace(-5, 5, 65)
	t = np.linspace(0, 4*np.pi, 129)
	xgrid, tgrid = np.meshgrid(x, t)
	X1 = f1(xgrid, tgrid)
	X2 = f2(xgrid, tgrid)
	X = X1 + X2

	# # Plot initial data
	# titles = ['$f_1(x,t)$', '$f_2(x,t)$', '$f$']
	# data = [X1, X2, X]
	# fig = plt.figure(figsize=(18, 6))
	# for n, title, d in zip(range(131, 134), titles, data):
	# 	plt.subplot(n)
	# 	plt.pcolor(xgrid, tgrid, d.real, shading='nearest')
	# 	plt.title(title)
	# plt.colorbar()
	# plt.show()
	# plt.savefig('original_data.png')

	print(X.shape)
	# Perform DMD analysis
	DMD_analysis = DMD_integration(X=X.T, rank=2, approach='mathlab_dmd')
	phi_r, eigen_r, modes_r, atilde, dmd = DMD_analysis.solve()

	for mode in dmd.modes.T:
	    plt.plot(x, mode.real)
	    plt.title('Modes')
	plt.show()

	for dynamic in dmd.dynamics:
	    plt.plot(t, dynamic.real)
	    plt.title('Dynamics')
	plt.show()

	fig = plt.figure(figsize=(17,6))

	for n, mode, dynamic in zip(range(131, 133), dmd.modes.T, dmd.dynamics):
	    plt.subplot(n)
	    plt.pcolor(xgrid, tgrid, (mode.reshape(-1, 1).dot(dynamic.reshape(1, -1))).real.T)

	plt.subplot(133)
	plt.pcolor(xgrid, tgrid, dmd.reconstructed_data.T.real)
	plt.colorbar()
	plt.show()

	plt.pcolor(xgrid, tgrid, (X-dmd.reconstructed_data.T).real)
	fig = plt.colorbar()
	plt.show()

if __name__ == "__main__":
	main()
