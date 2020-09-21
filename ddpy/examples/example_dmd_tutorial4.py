import __includes__

import os
import sys
import time
import warnings
import scipy
import numpy as np
from scipy import io
from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt
from dynamics.ode_solver import OdeIntegration
from dynamics.dmd_solver import DMD_integration
# warnings.filterwarnings("ignore")
np.random.seed(123)



def create_dataset(x_dim, t_dim):
	def f1(x,t):
		return np.exp(-x**2*.2)*np.cos(4*x)*np.exp(2.3j*t)

	def f2(x,t):
		return (1-np.exp(1-x**2/6)) * np.exp(1.3j*t)

	def f3(x, t):
		return (-.02*x**2 + 1) * (1.1j**(-2*t))

	x = np.linspace(-5, 5, x_dim)
	t = np.linspace(0, 4*np.pi, t_dim)

	xgrid, tgrid = np.meshgrid(x,t)

	X1 = f1(xgrid, tgrid)
	X2 = f2(xgrid, tgrid)
	X3 = f3(xgrid, tgrid)
	return xgrid, tgrid, (X1 + X2 + X3)



def main():
	# xgrid, tgrid, snapshots = create_dataset(2560, 1280)
	# plt.figure(figsize=(7, 7))
	# plt.pcolor(xgrid, tgrid, snapshots.real)
	# plt.show()
	#
	# X = snapshots.T
	# random_matrix = np.random.permutation(X.shape[0] * X.shape[1])
	# random_matrix = random_matrix.reshape(X.shape[1],  X.shape[0])
	# compression_matrix = random_matrix / np.linalg.norm(random_matrix)

	# start = time.time()
	# DMD_analysis = DMD_integration(X=X,
	# 							   rank=3,
	# 							   approach='mathlab_cdmd',
	# 							   compression_matrix=compression_matrix)
	# phi_r, eigen_r, modes_r, atilde, dmd_c = DMD_analysis.solve()
	# print('Elapsed time for DMD_basic = ', time.time() - start, 's.')
	#
	# plt.figure(figsize=(16, 8))
	# plt.subplot(1, 2, 1)
	# plt.plot(dmd_c.modes.real)
	# plt.subplot(1, 2, 2)
	# plt.plot(dmd_c.dynamics.T.real)
	# plt.show()
	#
	# start = time.time()
	# DMD_analysis = DMD_integration(X=X,
	# 							   rank=3,
	# 							   approach='mathlab_dmd')
	# phi_r, eigen_r, modes_r, atilde, dmd_basic = DMD_analysis.solve()
	# print('Elapsed time for DMD_c = ', time.time() - start, 's.')
	#
	# dmd_basic_error = np.linalg.norm(X-dmd_basic.reconstructed_data)
	# dmd_c_error     = np.linalg.norm(X-dmd_c    .reconstructed_data)
	# print("DMD_BASIC error: {}".format(dmd_basic_error))
	# print("DMD_C     error: {}".format(dmd_c_error))
	#
	# plt.figure(figsize=(16,8))
	# plt.subplot(1, 3, 1)
	# plt.title('Original snapshots')
	# plt.pcolor(xgrid, tgrid, X.real.T)
	# plt.subplot(1, 3, 2)
	# plt.title('Reconstructed with DMD')
	# plt.pcolor(xgrid, tgrid, dmd_basic.reconstructed_data.real.T)
	# plt.subplot(1, 3, 3)
	# plt.title('Reconstructed with CDMD')
	# plt.pcolor(xgrid, tgrid, dmd_c.reconstructed_data.real.T)
	# plt.show()



	time_dmd = []
	time_cdmd = []
	dim = []

	niter = 4
	ndims = 10 ** np.arange(2, 2+niter)
	nsnaps = [100] * niter
	for nsnap, ndim in zip(nsnaps, ndims):
		snapshots_matrix = create_dataset(ndim, nsnap)[-1].T
		dim.append(snapshots_matrix.shape[0])
		random_matrix = np.random.permutation(snapshots_matrix.shape[0] * snapshots_matrix.shape[1])
		random_matrix = random_matrix.reshape(snapshots_matrix.shape[1], snapshots_matrix.shape[0])

		compression_matrix = random_matrix / np.linalg.norm(random_matrix)

		t0 = time.time()
		DMD_analysis = DMD_integration(X=snapshots_matrix,
									   rank=-1,
									   approach='mathlab_dmd')
		phi_r, eigen_r, modes_r, atilde, dmd_basic = DMD_analysis.solve()
		t1 = time.time()
		time_dmd.append(t1-t0)

		t0 = time.time()
		DMD_analysis = DMD_integration(X=snapshots_matrix,
									   rank=-1,
									   approach='mathlab_cdmd',
									   compression_matrix=compression_matrix)
		phi_r, eigen_r, modes_r, atilde, dmd_c = DMD_analysis.solve()
		t1 = time.time()
		time_cdmd.append(t1-t0)

	plt.figure(figsize=(10,5))
	plt.plot(dim, time_dmd, 'ro--', label='exact dmd')
	plt.plot(dim, time_cdmd, 'bo--', label='compressed dmd')
	plt.legend()
	plt.ylabel('Seconds')
	plt.xlabel('Snapshots dimension')
	plt.show()

if __name__ == "__main__":
	main()
