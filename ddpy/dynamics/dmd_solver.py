#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
	This file is subject to the terms and conditions defined in
	file 'LICENSE.txt', which is part of this source code package.

	Written by Dr. Gianmarco Mengaldo, May 2020.
'''
import numpy as np
import pydmd
from pydmd import DMD
from pydmd import CDMD
from pydmd import MrDMD
from pydmd import FbDMD
from pydmd import HODMD
from pydmd import DMDc
print(pydmd.__file__)


class DMD_integration:

	def __init__(self, X, rank, u=None, approach='basic', **kwargs):
		self.X  = X
		self.rank = rank
		self.u = u
		self.approach = approach
		self.params = dict()
		for key in kwargs:
			self.params[key] = kwargs[key]

	def solve(self):
		if self.approach.lower() == 'basic': self.basic_dmd()
		elif self.approach.lower() == 'mathlab_dmd'  : self.mathlab_dmd  ()
		elif self.approach.lower() == 'mathlab_mrdmd': self.mathlab_mrdmd()
		elif self.approach.lower() == 'mathlab_cdmd' : self.mathlab_cdmd ()
		elif self.approach.lower() == 'mathlab_fbdmd': self.mathlab_fbdmd()
		elif self.approach.lower() == 'mathlab_hodmd': self.mathlab_hodmd()
		elif self.approach.lower() == 'mathlab_dmdc' : self.mathlab_dmdc ()

		else:
			raise ValueError(self.approach, 'not implemented.')
		return self.phi_r, self.eigen_r, self.modes_r, self.atilde, self.dmd

	# def basic_dmd(self):
	#
	# 	# Step 1
	# 	U, Sigma, VT = np.linalg.svd(self.X, full_matrices=0)
	# 	U_r     = U[:,:self.r]
	# 	Sigma_r = np.diag(Sigma[:self.r])
	# 	VT_r    = VT[:self.r,:]
	#
	# 	# Step 2
	# 	self.atilde = np.linalg.solve(Sigma_r.T,(U_r.T @ self.Xp @ VT_r.T).T).T
	#
	# 	# Step 3
	# 	self.eigen_r, W = np.linalg.eig(self.atilde)
	# 	self.eigen_r = np.diag(self.eigen_r)
	#
	# 	# Step 4
	# 	self.phi_r = self.Xp @ np.linalg.solve(Sigma_r.T, VT_r).T @ W
	# 	self.modes_r = np.linalg.solve(W @ self.eigen_r, Sigma_r @ VT_r[:,0])

	def mathlab_dmd(self):
		self.tlsq  = 0
		self.exact = False
		self.opt   = False
		for key in self.params:
			if key == 'tlsq' : self.tlsq  = self.params[key]
			if key == 'opt'  : self.opt   = self.params[key]
			if key == 'exact': self.exact = self.params[key]
		dmd = DMD(svd_rank=self.rank, tlsq_rank=self.tlsq, exact=self.exact, opt=self.opt)
		dmd.fit(self.X)
		self.eigen_r = dmd.eigs
		self.modes_r = dmd.modes
		self.phi_r   = dmd.dynamics
		self.atilde  = dmd.atilde
		self.dmd = dmd

	def mathlab_mrdmd(self):
		self.tlsq  = 0
		self.exact = False
		self.opt   = False
		self.max_level  = 1
		self.max_cycles = 1
		for key in self.params:
			if key == 'tlsq' : self.tlsq  = self.params[key]
			if key == 'opt'  : self.opt   = self.params[key]
			if key == 'exact': self.exact = self.params[key]
			if key == 'max_level' : self.max_level  = self.params[key]
			if key == 'max_cycles': self.max_cycles = self.params[key]
		dmd = MrDMD(svd_rank=self.rank, tlsq_rank=self.tlsq, exact=self.exact,
				    max_level=self.max_level, max_cycles=self.max_cycles)
		dmd.fit(self.X)
		self.eigen_r = dmd.eigs
		self.modes_r = dmd.modes
		self.phi_r   = dmd.dynamics
		self.atilde  = dmd.atilde
		self.dmd = dmd

	def mathlab_cdmd(self):
		self.tlsq  = 0
		self.exact = False
		self.opt   = False
		self.compression_matrix = None
		for key in self.params:
			if key == 'tlsq' : self.tlsq  = self.params[key]
			if key == 'opt'  : self.opt   = self.params[key]
			if key == 'exact': self.exact = self.params[key]
			if key == 'compression_matrix' : self.compression_matrix  = self.params[key]
		dmd = CDMD(svd_rank=self.rank, compression_matrix=self.compression_matrix)
		dmd.fit(self.X)
		self.eigen_r = dmd.eigs
		self.modes_r = dmd.modes
		self.phi_r   = dmd.dynamics
		self.atilde  = dmd.atilde
		self.dmd = dmd

	def mathlab_fbdmd(self):
		self.tlsq  = 0
		self.exact = False
		self.opt   = False
		self.compression_matrix = None
		for key in self.params:
			if key == 'tlsq' : self.tlsq  = self.params[key]
			if key == 'opt'  : self.opt   = self.params[key]
			if key == 'exact': self.exact = self.params[key]
		dmd = FbDMD(svd_rank=self.rank, exact=self.exact)
		dmd.fit(self.X)
		self.eigen_r = dmd.eigs
		self.modes_r = dmd.modes
		self.phi_r   = dmd.dynamics
		self.atilde  = dmd.atilde
		self.dmd = dmd

	def mathlab_hodmd(self):
		self.tlsq  = 0
		self.exact = False
		self.opt   = False
		self.d     = 30
		for key in self.params:
			if key == 'tlsq' : self.tlsq  = self.params[key]
			if key == 'opt'  : self.opt   = self.params[key]
			if key == 'exact': self.exact = self.params[key]
			if key == 'd'    : self.d     = self.params[key]
		dmd = HODMD(svd_rank=self.rank, tlsq_rank=self.tlsq, exact=self.exact, opt=self.opt, d=self.d)
		dmd.fit(self.X)
		self.eigen_r = dmd.eigs
		self.modes_r = dmd.modes
		self.phi_r   = dmd.dynamics
		self.atilde  = dmd.atilde
		self.dmd = dmd

	def mathlab_dmdc(self):
		self.tlsq = 0
		self.opt  = False
		if not isinstance(self.u,(np.ndarray,list)):
			if self.u == None: raise ValueError('DMDc requires control input `u`')
		for key in self.params:
			if key == 'tlsq' : self.tlsq  = self.params[key]
			if key == 'opt'  : self.opt   = self.params[key]
		dmd = DMDc(svd_rank=self.rank, tlsq_rank=self.tlsq, opt=self.opt)
		dmd.fit(self.X, self.u)
		self.eigen_r = dmd.eigs
		self.modes_r = dmd.modes
		self.phi_r   = dmd.dynamics
		self.atilde  = dmd.atilde
		self.dmd = dmd
