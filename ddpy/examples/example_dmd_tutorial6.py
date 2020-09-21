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


def myfunc(x):
	return np.cos(x)*np.sin(np.cos(x)) + np.cos(x*.2)



def main():
	x = np.linspace(0, 10, 64)
	y = myfunc(x)
	snapshots = y
	plt.plot(x, snapshots, '.')
	plt.show()

	X = snapshots
	DMD_analysis = DMD_integration(X=X, rank=0, approach='mathlab_hodmd', opt=True, exact=True, d=30)
	phi_r, eigen_r, modes_r, atilde, hodmd_res = DMD_analysis.solve()

	hodmd_res.plot_eigs()

	hodmd_res.original_time['dt']   = hodmd_res.dmd_time['dt'] = x[1] - x[0]
	hodmd_res.original_time['t0']   = hodmd_res.dmd_time['t0'] = x[0]
	hodmd_res.original_time['tend'] = hodmd_res.dmd_time['tend'] = x[-1]
	plt.plot(hodmd_res.original_timesteps, snapshots, '.', label='snapshots')
	plt.plot(hodmd_res.original_timesteps, y, '-', label='original function')
	plt.plot(hodmd_res.dmd_timesteps, hodmd_res.reconstructed_data[0].real,
			 '--', label='DMD output')
	plt.legend()
	plt.show()

	hodmd_res.dmd_time['tend'] = 50
	fig = plt.figure(figsize=(15, 5))
	plt.plot(hodmd_res.original_timesteps, snapshots, '.', label='snapshots')
	plt.plot(np.linspace(0, 50, 128), myfunc(np.linspace(0, 50, 128)), '-', label='original function')
	plt.plot(hodmd_res.dmd_timesteps, hodmd_res.reconstructed_data[0].real, '--', label='DMD output')
	plt.legend()
	plt.show()



	noise_range = [.01, .05, .1, .2]
	fig = plt.figure(figsize=(15, 10))
	future = 20
	for id_plot, i in enumerate(noise_range, start=1):
		X = y + np.random.uniform(-i, i, size=y.shape)
		DMD_analysis = DMD_integration(
			X=X,
			Xp=X,
			rank=0,
			approach='mathlab_hodmd',
			opt=True,
			exact=True,
			d=30)
		phi_r, eigen_r, modes_r, atilde, hodmd_res = DMD_analysis.solve()
		hodmd_res.original_time['dt']   = hodmd_res.dmd_time['dt'] = x[1] - x[0]
		hodmd_res.original_time['t0']   = hodmd_res.dmd_time['t0'] = x[0]
		hodmd_res.original_time['tend'] = hodmd_res.dmd_time['tend'] = x[-1]
		hodmd_res.dmd_time['tend'] = 20

		plt.subplot(2, 2, id_plot)
		plt.plot(hodmd_res.original_timesteps, X, '.', label='snapshots')
		plt.plot(np.linspace(0, future, 128), myfunc(np.linspace(0, future, 128)), '-', label='original function')
		plt.plot(hodmd_res.dmd_timesteps, hodmd_res.reconstructed_data[0].real, '--', label='DMD output')
		plt.legend()
		plt.title('Noise [{} - {}]'.format(-i, i))
	plt.show()


if __name__ == "__main__":
	main()
