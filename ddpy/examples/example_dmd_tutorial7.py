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


def create_system(n, m):
    A = scipy.linalg.helmert(n, True)
    B = np.random.rand(n, n)-.5
    x0 = np.array([0.25]*n)
    u = np.random.rand(n, m-1)-.5
    snapshots = [x0]
    for i in range(m-1):
        snapshots.append(A.dot(snapshots[i])+B.dot(u[:, i]))
    snapshots = np.array(snapshots).T
    return {'snapshots': snapshots, 'u': u, 'B': B, 'A': A}



def main():
	s = create_system(25, 10)
	print(s['snapshots'].shape)

	DMD_analysis = DMD_integration(X=s['snapshots'], u=s['u'], rank=-1, approach='mathlab_dmdc', opt=True, exact=True)
	phi_r, eigen_r, modes_r, atilde, dmd_c = DMD_analysis.solve()

	plt.figure(figsize=(16,6))
	plt.subplot(121)
	plt.title('Original system')
	plt.pcolor(s['snapshots'].real)
	plt.colorbar()
	plt.subplot(122)
	plt.title('Reconstructed system')
	plt.pcolor(dmd_c.reconstructed_data().real)
	plt.colorbar()
	plt.show()

	new_u = np.exp(s['u'])
	plt.figure(figsize=(8,6))
	plt.pcolor(dmd_c.reconstructed_data(new_u).real)
	plt.colorbar()
	plt.show()

	dmd_c.dmd_time['dt'] = .5
	new_u = np.random.rand(s['u'].shape[0], dmd_c.dynamics.shape[1]-1)

	plt.figure(figsize=(8,6))
	plt.pcolor(dmd_c.reconstructed_data(new_u).real)
	plt.colorbar()
	plt.show()

if __name__ == "__main__":
	main()
