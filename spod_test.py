#Import libraries
import xarray as xr
import numpy  as np
import pysindy as ps
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import os
import pickle
import sys
import time
import h5py
import warnings
from pathlib import Path
CWD = os.getcwd()
sys.path.append(os.path.join(CWD,"U:\ anaconda3"))
# Import library specific modules
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming
import pyspod.weights as weights
#Get data and store
ds2 = xr.open_dataset('spr_part_2.nc')
variables = ['slip_potency']
t = np.array(ds2['time'])
x1 = np.array(ds2['x'])
x2 = np.array(ds2['z'])
X = np.array(ds2[variables[0]]).T
X_train = np.load ('modes1to0003_freq0010.npy')
print(X_train)
X_train = np.array(X_train).T
print(X_train.shape)
X_train2d = np.reshape(X_train,(2,-1)).T
print(X_train2d.shape)
print('t.shape  = ', t.shape)
print('x1.shape = ', x1.shape)
print('x2.shape = ', x2.shape)
print('X.shape  = ', X.shape)
# define required and optional parameters
params = dict()
params['dt'          ] = 1
params['nt'          ] = len(t)
params['xdim'        ] = 2
params['nv'          ] = len(variables)
params['n_FFT'       ] = np.ceil(64)
params['n_freq'      ] = params['n_FFT'] / 2 + 1
params['n_overlap'   ] = np.ceil(params['n_FFT'] * 10 / 100)
params['savefreqs'   ] = np.arange(0,params['n_freq'])
params['conf_level'  ] = 0.95
params['n_vars'      ] = 1
params['n_modes_save'] = 3
params['normvar'     ] = False
params['savedir'     ] = os.path.join(CWD, 'results', Path('spr_part_2.nc').stem)
params['weights'     ] = np.ones([len(x1) * len(x2) * params['nv'],1])
params['mean'        ] = 'longtime'
# Perform SPOD analysis using low storage module
SPOD_analysis = SPOD_streaming(X=X, params=params, data_handler=False, variables=variables)
spod = SPOD_analysis.fit()
# Show results
T_approx = 10 # approximate period
freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=spod.freq)
modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
spod.plot_eigs()
freq = spod.freq
spod.plot_eigs_vs_frequency(freq=freq)
spod.plot_eigs_vs_period   (freq=freq, xticks=[1, 0.5, 0.2, 0.1, 0.05, 0.02])
spod.plot_2D_modes_at_frequency(
    freq_required=freq_found,
    freq=freq,
    x1=x1,
    x2=x2,
    modes_idx=[0,1,2],
    vars_idx=[0])

#Set parameters for SINDy object
poly_order = 2
threshold = 0.0001
order = 1
poly_library = ps.PolynomialLibrary(degree = poly_order, include_interaction=False)
#Create SINDy object
model = ps.SINDy(
    #Define numerical differentiation method to compute X' from X
    differentiation_method =ps.SINDyDerivative (order=order),
    #Provides a set of sparse regression solvers for determining  Ξ (contains sparse vectors)
    optimizer=ps.STLSQ(threshold=threshold),
    #Constructs set of library functions and handles formation of Θ(X)
    feature_library=poly_library,
    )
X_train2d = np.float64(X_train2d.real)
model.fit(X_train2d, multiple_trajectories=False, quiet=True)
model.print()
#Plot results
fig = plt.figure(figsize=(10, 4.5))
ax = fig.add_subplot(121, projection="3d")
ax.plot(X_train2d[:, 0], X_train2d[:, 1])
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
plt.title("Full Simulation")
plt.show()
