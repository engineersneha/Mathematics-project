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


def create_sample_data():
    x = np.linspace(-10, 10, 80)
    t = np.linspace(0, 20, 1600)
    Xm, Tm = np.meshgrid(x, t)

    D = np.exp(-np.power(Xm/2, 2)) * np.exp(0.8j * Tm)
    D += np.sin(0.9 * Xm) * np.exp(1j * Tm)
    D += np.cos(1.1 * Xm) * np.exp(2j * Tm)
    D += 0.6 * np.sin(1.2 * Xm) * np.exp(3j * Tm)
    D += 0.6 * np.cos(1.3 * Xm) * np.exp(4j * Tm)
    D += 0.2 * np.sin(2.0 * Xm) * np.exp(6j * Tm)
    D += 0.2 * np.cos(2.1 * Xm) * np.exp(8j * Tm)
    D += 0.1 * np.sin(5.7 * Xm) * np.exp(10j * Tm)
    D += 0.1 * np.cos(5.9 * Xm) * np.exp(12j * Tm)
    D += 0.1 * np.random.randn(*Xm.shape)
    D += 0.03 * np.random.randn(*Xm.shape)
    D += 5 * np.exp(-np.power((Xm+5)/5, 2)) * np.exp(-np.power((Tm-5)/5, 2))
    D[:800,40:] += 2
    D[200:600,50:70] -= 3
    D[800:,:40] -= 2
    D[1000:1400,10:30] += 3
    D[1000:1080,50:70] += 2
    D[1160:1240,50:70] += 2
    D[1320:1400,50:70] += 2
    return D.T

def make_plot(X, x=None, y=None, figsize=(12, 8), title=''):
    """
    Plot of the data X
    """
    plt.figure(figsize=figsize)
    plt.title(title)
    X = np.real(X)
    CS = plt.pcolor(x, y, X)
    cbar = plt.colorbar(CS)
    plt.xlabel('Space')
    plt.ylabel('Time')
    plt.show()

def main():
	sample_data = create_sample_data()
	x = np.linspace(-10, 10, 80)
	t = np.linspace(0, 20, 1600)
	make_plot(sample_data.T, x=x, y=t)

	X = sample_data
	DMD_analysis = DMD_integration(X=X, rank=-1, approach='mathlab_dmd', opt=True, exact=True)
	phi_r, eigen_r, modes_r, atilde, dmd = DMD_analysis.solve()
	fig = plt.plot(scipy.linalg.svdvals(np.array([snapshot.flatten() for snapshot in sample_data]).T), 'o')
	plt.show()
	make_plot(dmd.reconstructed_data.T, x=x, y=t)

	X = sample_data
	DMD_analysis = DMD_integration(X=X, rank=1000, approach='mathlab_mrdmd',
								   max_level=2, max_cycles=1)
	phi_r, eigen_r, modes_r, atilde, dmd = DMD_analysis.solve()
	make_plot(dmd.reconstructed_data.T, x=x, y=t)
	print('The number of eigenvalues is {}'.format(dmd.eigs.shape[0]))
	dmd.plot_eigs(show_axes=True, show_unit_circle=True, figsize=(8, 8))
	dmd.plot_eigs(show_axes=True, show_unit_circle=True, figsize=(8, 8), level=3, node=0)

	fig = plt.figure(figsize=(12,8))
	n_subplots = 6
	for id_subplot in range(n_subplots):
		pmodes = dmd.partial_modes(level=id_subplot)
		pdyna  = dmd.partial_dynamics(level=id_subplot)
		plt.subplot(2, n_subplots, id_subplot+1)
		fig = plt.plot(t, pdyna.real.T)
		plt.subplot(2, n_subplots, id_subplot+n_subplots+1)
		fig = plt.plot(x, pmodes.real)
	plt.show()

	pdata = dmd.partial_reconstructed_data(level=0)
	make_plot(pdata.T, x=x, y=t, title='level 0', figsize=(7.5, 5))
	for i in range(1, 6):
	    pdata += dmd.partial_reconstructed_data(level=i)
	    make_plot(pdata.T, x=x, y=t, title='levels 0-' + str(i), figsize=(7.5, 5))






if __name__ == "__main__":
	main()
