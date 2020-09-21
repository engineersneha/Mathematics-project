import __includes__

import os
import sys
import time
import scipy
import pandas as pd
import numpy  as np
import netCDF4 as nc
from tqdm import tqdm
from matplotlib import animation
import matplotlib.pyplot as plt
from dynamics.ode_solver import OdeIntegration
from dynamics.dmd_solver import DMD_integration



def get_data(var_name, to_csv=False):
	directory = 'slip-potency/data/spr_part_3'
	files_raw = os.listdir(directory)
	files_raw.sort()
	files = [file for file in files_raw if file.endswith(".grd")]
	files.sort()
	k_skip = 1
	k_0    = 0
	k_N    = int(1 * len(files))
	snapshots = np.arange(k_0, k_N, k_skip)
	t = np.empty([len(snapshots),1])

	filename = files[0]
	print(filename)
	fn = os.path.join(directory, filename)
	ds = nc.Dataset(fn)
	print(ds)
	for var in ds.variables.values():
		print(var)
		print(ds[var_name][:])
	tmp = np.array(ds[var_name][:])
	if len(tmp.shape) == 1: dim1 = tmp.shape[0] * 1
	elif len(tmp.shape) == 2: dim1 = tmp.shape[0] * tmp.shape[1]
	elif len(tmp.shape) == 3: dim1 = tmp.shape[0] * tmp.shape[1] * tmp.shape[2]
	elif len(tmp.shape) == 4: dim1 = tmp.shape[0] * tmp.shape[1] * tmp.shape[2] * tmp.shape[3]
	else: ValueError('Support up to 4D tensors supported only')
	dim2 = len(t)
	data = np.empty([dim1, dim2])
	for i in tqdm(snapshots,desc='extracting_data from ' + str(directory)):
		filename = files[i]
		if filename.endswith(".grd"):
			fn = os.path.join(directory, filename)
			ds = nc.Dataset(fn)
			tmp = np.array(ds[var_name][:])
			if len(tmp.shape) == 1: tmp = tmp.reshape(tmp.shape[0],)
			elif len(tmp.shape) == 2: tmp = tmp.reshape(tmp.shape[0] * tmp.shape[1],)
			elif len(tmp.shape) == 3: tmp = tmp.reshape(tmp.shape[0] * tmp.shape[1] * tmp.shape[2],)
			elif len(tmp.shape) == 4: tmp = tmp.reshape(tmp.shape[0] * tmp.shape[1] * tmp.shape[2] * tmp.shape[3],)
			else: ValueError('Support up to 4D tensors supported only')
			data[:,i] = tmp
	if to_csv:
		df = pd.DataFrame(data[:,:])
		df = df.add_prefix('time__')
		df.index.name = var_name
		df.to_csv(var_name+'_data.csv')

	return data




def main():

	# Parameters
	movie = False
	data_retrieval = False
	basic = True
	fb    = False
	ho    = False
	c     = False
	all   = False

	# Data retrieval
	if data_retrieval:
		X = get_data(var_name='z', to_csv=True)

	# Reading retrieved data and transform accordingly
	t0 = time.time()
	print('reading data ...')
	x_coord = pd.read_csv('x_data.csv',index_col=0)
	y_coord = pd.read_csv('y_data.csv',index_col=0)
	x_coord = x_coord.values
	y_coord = y_coord.values
	xgrid, ygrid = np.meshgrid(x_coord, y_coord)
	xgrid = xgrid / 1000
	ygrid = ygrid / 1000 - np.mean(ygrid) / 1000
	X = pd.read_csv('z_data_spr_part_1__12000_22000.csv',index_col=0)
	X = X.values
	X_movie = X.reshape((xgrid.shape[0],xgrid.shape[1],X.shape[1]))
	print('done. Elapsed time: ',time.time() - t0)

	# Generate movie
	if movie:
		t_skip = 10
		t = np.arange(0, X_movie.shape[2], t_skip)
		fig = plt.figure(figsize=(18, 6))
		frames = [
			[plt.pcolor(xgrid, ygrid, X_movie[:,:,state], shading='nearest')]
			for state in t
		]
		a = animation.ArtistAnimation(fig, frames, interval=70, blit=False, repeat=False)
		Writer = animation.writers['ffmpeg']
		writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
		a.save('earthquake__12000_22000.mp4', writer=writer)
		plt.close('all')

	# Generate random matrix
	random_matrix = np.random.permutation(X.shape[0] * X.shape[1])
	random_matrix = random_matrix.reshape(X.shape[1],  X.shape[0])
	compression_matrix = random_matrix / np.linalg.norm(random_matrix)

	print(X.shape)
	t0 = time.time()
	print('solving DMD ...')
	if basic:
		dmd_basic = DMD_integration(X=X, rank=-1, exact=True, approach='mathlab_dmd')
		_, _, _, _, sol_dmd_basic = dmd_basic.solve()
		sol_dmd_basic.plot_eigs()
		plt.plot(sol_dmd_basic.dmd_timesteps, sol_dmd_basic.dynamics.T.real)
		plt.show()

		X_reconstructed_movie = sol_dmd_basic.reconstructed_data.real.reshape(
			xgrid.shape[0],xgrid.shape[1],sol_dmd_basic.reconstructed_data.shape[1])
		t = np.arange(sol_dmd_basic.dmd_time['t0'], sol_dmd_basic.dmd_time['tend'], sol_dmd_basic.dmd_time['dt'])
		print(t)
		print(sol_dmd_basic.dmd_time['dt'])
		print(sol_dmd_basic.dmd_time['t0'])
		dmd_states = [int((state - sol_dmd_basic.dmd_time['t0']) / sol_dmd_basic.dmd_time['dt']) for state in t]
		print(dmd_states)
		fig = plt.figure(figsize=(18,6))
		frames = [
			[plt.pcolor(xgrid, ygrid, X_reconstructed_movie[:,:,state].real)]
			for state in dmd_states
		]
		a = animation.ArtistAnimation(fig, frames, interval=70, blit=False, repeat=False)
		Writer = animation.writers['ffmpeg']
		writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
		a.save('earthquake__12000_22000__dmd_basic_reconstruction.mp4', writer=writer)
		plt.close('all')
	if fb:
		dmd_fb = DMD_integration(X=X, rank=-1, exact=True, approach='mathlab_fbdmd')
		_, _, _, _, sol_dmd_fb = dmd_fb.solve()
		sol_dmd_fb.plot_eigs()
	if ho:
		dmd_ho = DMD_integration(X=X, rank=-1, approach='mathlab_hodmd', opt=True, exact=True, d=30)
		_, _, _, _, sol_dmd_ho = dmd_ho.solve()
		sol_dmd_ho.plot_eigs()
	if c:
		dmd_c = DMD_integration(X=X, rank=-1, approach='mathlab_cdmd', compression_matrix=compression_matrix)
		_, _, _, _, sol_dmd_c = dmd_c.solve()
		sol_dmd_c.plot_eigs()
	if all:
		dmd_basic = DMD_integration(X=X, rank=-1, approach='mathlab_dmd'  , exact=True)
		dmd_fb    = DMD_integration(X=X, rank=-1, approach='mathlab_fbdmd', exact=True)
		dmd_ho    = DMD_integration(X=X, rank=-1, approach='mathlab_hodmd', exact=True, opt=True, d=30)
		dmd_c     = DMD_integration(X=X, rank=-1, approach='mathlab_cdmd' , compression_matrix=compression_matrix)
		_, _, _, _, sol_dmd_basic = dmd_basic.solve()
		_, _, _, _, sol_dmd_fb    = dmd_fb   .solve()
		_, _, _, _, sol_dmd_ho    = dmd_ho   .solve()
		_, _, _, _, sol_dmd_c     = dmd_c    .solve()
		sol_dmd_basic.plot_eigs()
		sol_dmd_fb   .plot_eigs()
		sol_dmd_ho   .plot_eigs()
		sol_dmd_c    .plot_eigs()
	print('done. Elapsed time: ',time.time() - t0)






if __name__ == "__main__":
	main()
