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



def main():
	x1 = np.linspace(-3, 3, 80)
	x2 = np.linspace(-3, 3, 80)
	x1grid, x2grid = np.meshgrid(x1, x2)
	time = np.linspace(0, 6, 16)

	data  = [2 / np.cosh(x1grid) / np.cosh(x2grid) * (1.2j**-t) for t in time]
	noise = [np.random.uniform(0., .1, size=x1grid.shape) for t in time]
	data_noised = [d+n for d, n in zip(data, noise)]

	fig = plt.figure(figsize=(12,8))
	for id_subplot, snapshot in enumerate(data_noised, start=1):
		plt.subplot(4, 4, id_subplot)
		plt.pcolor(x1grid, x2grid, snapshot.real, shading='nearest', vmin=-1, vmax=1)
	plt.show()
	fig = plt.figure(figsize=(12,8))
	for id_subplot, snapshot in enumerate(data, start=1):
		plt.subplot(4, 4, id_subplot)
		plt.pcolor(x1grid, x2grid, snapshot.real)
	plt.show()

	X = data_noised
	# print('X = ',X.shape)
	DMD_analysis = DMD_integration(X=X, rank=10, approach='mathlab_dmd', tlsq=5, opt=True, exact=True)
	phi_r, eigen_r, modes_r, atilde, dmd = DMD_analysis.solve()
	fig = plt.plot(scipy.linalg.svdvals(np.array([snapshot.flatten() for snapshot in data_noised]).T), 'o')
	plt.show()

	# # Reconstruct dynamics V1
	# fig = plt.figure(figsize=(12,8))
	# for id_subplot, snapshot in enumerate(dmd.reconstructed_data.T, start=1):
	#     plt.subplot(4, 4, id_subplot)
	#     plt.pcolor(x1grid, x2grid, snapshot.reshape(x1grid.shape).real, vmin=-1, vmax=1)
	# plt.show()

	# Reconstruct dynamics V2
	print(modes_r.shape)
	print(phi_r.shape)
	reconstructed_dynamics = modes_r.dot(phi_r)
	fig = plt.figure(figsize=(12,8))
	for id_subplot, snapshot in enumerate(reconstructed_dynamics.T, start=1):
		plt.subplot(4, 4, id_subplot)
		plt.pcolor(x1grid, x2grid, snapshot.reshape(x1grid.shape).real, vmin=-1, vmax=1)
	plt.show()

	fig = plt.figure()
	dmd_states = [state.reshape(x1grid.shape) for state in dmd.reconstructed_data.T]
	frames = [
		[plt.pcolor(x1grid, x2grid, state.real, vmin=-1, vmax=1)]
		for state in dmd_states
	]
	ani = animation.ArtistAnimation(fig, frames, interval=70, blit=False, repeat=False)
	# Set up formatting for the movie files
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
	ani.save('original.mp4', writer=writer)
	plt.close('all')

	# Reconstruct dynamics V1
	dmd.dmd_time['dt'  ] *= 0.25
	dmd.dmd_time['tend'] *= 4.00
	fig = plt.figure(figsize=(12,8))
	print(dmd.reconstructed_data.T.shape)
	fig = plt.figure()
	dmd_states = [state.reshape(x1grid.shape) for state in dmd.reconstructed_data.T]
	frames = [
		[plt.pcolor(x1grid, x2grid, state.real, vmin=-1, vmax=1)]
		for state in dmd_states
	]
	ani = animation.ArtistAnimation(fig, frames, interval=70, blit=False, repeat=False)
	# Set up formatting for the movie files
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
	ani.save('modified.mp4', writer=writer)
	plt.close('all')

	compute_integral = scipy.integrate.trapz
	original_int = [compute_integral(compute_integral(snapshot)).real for snapshot in data_noised]
	dmd_int = [compute_integral(compute_integral(state)).real for state in dmd_states]
	figure = plt.figure(figsize=(18, 5))
	plt.plot(dmd.original_timesteps, original_int, 'bo', label='original snapshots')
	plt.plot(dmd.dmd_timesteps, dmd_int, 'r.', label='dmd states')
	plt.ylabel('Integral')
	plt.xlabel('Time')
	plt.grid()
	plt.legend()
	plt.show()

if __name__ == "__main__":
	main()
