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



def get_data():
	snapshots = [
	    np.genfromtxt(
			'data_pydmd/velocity0.{}.csv'.format(i),
			delimiter=',',
			skip_header=1)[:, 0]
	    for i in range(20, 40)
	]
	coords = np.genfromtxt('data_pydmd/velocity0.20.csv', delimiter=',', skip_header=1)[:, -3:-1]
	return snapshots, coords



def main():
	snapshots, coords = get_data()


	plt.figure(figsize=(10, 10))
	for i, snapshot in enumerate(snapshots[::5], start=1):
	    plt.subplot(2, 2, i)
	    plt.scatter(coords[:, 0], coords[:, 1], c=snapshot, marker='.')
	plt.show()

	for s in snapshots:
		print(s.shape)
	sys.exit(2)
	dmd = DMD_integration(X=snapshots,rank=10,exact=True,approach='mathlab_dmd')
	phi_r, eigen_r, modes_r, atilde, dmd_res = dmd.solve()
	fbdmd = DMD_integration(X=snapshots,rank=10,exact=True,approach='mathlab_fbdmd')
	phi_r, eigen_r, modes_r, atilde, fbdmd_res = fbdmd.solve()
	print('[DMD  ] Total distance between eigenvalues and unit circle: {}'.format(
	    np.sum(np.abs(dmd_res.eigs.real**2 + dmd_res.eigs.imag**2 - 1))
	))
	print('[FbDMD] Total distance between eigenvalues and unit circle: {}'.format(
	    np.sum(np.abs(fbdmd_res.eigs.real**2 + fbdmd_res.eigs.imag**2 - 1))
	))

	dmd_res.plot_eigs()
	fbdmd_res.plot_eigs()

	dmd_res.dmd_time['dt'] *= .5
	dmd_res.dmd_time['tend'] += 10
	plt.plot(dmd_res.dmd_timesteps, dmd_res.dynamics.T.real)
	plt.show()

	fbdmd_res.dmd_time['dt'] *= .5
	fbdmd_res.dmd_time['tend'] += 10
	plt.plot(fbdmd_res.dmd_timesteps, fbdmd_res.dynamics.T.real)
	plt.show()

	fig = plt.figure(figsize=(12,8))
	t = np.arange(dmd_res.dmd_time['t0'], dmd_res.dmd_time['tend'], dmd_res.dmd_time['dt'])
	dmd_states = [int((state - dmd_res.dmd_time['t0']) / dmd_res.dmd_time['dt']) for state in t]
	print(dmd_states)
	frames = [
		[plt.scatter(coords[:, 0], coords[:, 1],
					 c=dmd_res.reconstructed_data[:, state].real, marker='.')]
		for state in dmd_states
	]
	ani = animation.ArtistAnimation(fig, frames, interval=70, blit=False, repeat=False)
	# Set up formatting for the movie files
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
	ani.save('simulation_dmd.mp4', writer=writer)
	plt.close('all')

	fig = plt.figure(figsize=(12,8))
	t = np.arange(fbdmd_res.dmd_time['t0'], fbdmd_res.dmd_time['tend'], fbdmd_res.dmd_time['dt'])
	dmd_states = [int((state - fbdmd_res.dmd_time['t0']) / fbdmd_res.dmd_time['dt']) for state in t]
	print(dmd_states)
	frames = [
		[plt.scatter(coords[:, 0], coords[:, 1], c=fbdmd_res.reconstructed_data[:, state].real, marker='.')]
		for state in dmd_states
	]
	ani = animation.ArtistAnimation(fig, frames, interval=70, blit=False, repeat=False)
	# Set up formatting for the movie files
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
	ani.save('simulation_fbdmd.mp4', writer=writer)
	plt.close('all')



if __name__ == "__main__":
	main()
