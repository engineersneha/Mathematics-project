#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    This file is subject to the terms and conditions defined in
    file 'LICENSE.txt', which is part of this source code package.

    Written by Dr. Gianmarco Mengaldo, May 2020.
'''
import numpy as np
from utils.helpers import prettify
from scipy import integrate



class OdeIntegration:

    def __init__(self,rhs,x0,t,params,backend='scipy_solve_ivp',atol=1e-8,rtol=1e-8,approach=None):
        self.rhs = rhs
        self.x0 = x0
        self.t = t
        self.atol = atol
        self.rtol = rtol
        self.backend = backend
        self.approach = approach
        self.params = params

    def solve_ode(self):
        if   self.backend.lower() == 'scipy_ode': self.scipy_solve_ode()
        elif self.backend.lower() == 'scipy_odeint': self.scipy_solve_odeint()
        elif self.backend.lower() == 'scipy_solve_ivp': self.scipy_solve_ivp()
        else:
            raise ValueError(self.backend, 'not implemented.')
        return self.t,self.sol

    def scipy_solve_ode(self):
        # Set integration method
        r = integrate.ode(self.rhs)
        if self.approach == None: self.approach = 'vode'
        if self.approach.lower() == 'vode':
            raise ValueError('`vode` not supported, as scipy ode.integrate has a possible bug')
        elif self.approach.lower() == 'zvode':
            raise ValueError('`zvode` not supported, as scipy ode.integrate has a possible bug')
        elif self.approach.lower() == 'lsoda':
            raise ValueError('`lsoda` not supported, as scipy ode.integrate has a possible bug')
        elif self.approach.lower() == 'dopri5':
            r.set_integrator(self.approach, atol=self.atol, rtol=self.rtol)
        elif self.approach.lower() == 'dop853':
            r.set_integrator(self.approach, atol=self.atol, rtol=self.rtol)
        else:
            raise ValueError(self.approach, 'integration method not recognized.')
        # Set initial conditions
        r.set_initial_value(self.x0)
        # Set coefficients
        if len(self.params) == 1:
            r.set_f_params(self.params[0])
        elif len(self.params) == 2:
            r.set_f_params(self.params[0],self.params[1])
        elif len(self.params) == 3:
            r.set_f_params(self.params[0],self.params[1],self.params[2])
        elif len(self.params) == 4:
            r.set_f_params(self.params[0],self.params[1],self.params[2],self.params[3])
        elif len(self.params) == 5:
            r.set_f_params(self.params[0],self.params[1],self.params[2],self.params[3],self.params[4])
        elif len(self.params) == 6:
            r.set_f_params(self.params[0],self.params[1],self.params[2],self.params[3],self.params[4],self.params[5])
        else:
            raise ValueError('maximum number of coefficients available is 6.')
        # Solve ODEs
        self.sol = np.empty([len(self.t),len(self.x0)])
        self.sol[0,:] = r.integrate(r.t)
        for k in range(1,len(self.t)):
            dt = self.t[k] - self.t[k-1]
            self.sol[k,:] = r.integrate(r.t+dt)

    def scipy_solve_odeint(self):
        self.sol = integrate.odeint(
            self.rhs, self.x0, self.t, args=self.params, tfirst=True, atol=self.atol, rtol=self.rtol)

    def scipy_solve_ivp(self):
        t0 = self.t[0]
        tN = self.t[-1]
        self.sol = integrate.solve_ivp(
            self.rhs, t_span=(t0,tN), y0=self.x0, t_eval=self.t,
            args=self.params, method=self.approach, atol=self.atol, rtol=self.rtol)
        self.t = self.sol.t
        self.sol = self.sol.y.T

    def __repr__(self):
        return '\n'.join([
            'ODE object',
            prettify('backend',self.backend),
            prettify('approach',self.approach),
            prettify('rhs',self.rhs),
            prettify('x0',self.x0),
            prettify('params',self.params),
            prettify('atol',self.atol),
            prettify('rtol',self.rtol),
            prettify('t', self.t),
            prettify('sol', self.sol),
            ])
