#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    This file is subject to the terms and conditions defined in
    file 'LICENSE.txt', which is part of this source code package.

    Written by Dr. Gianmarco Mengaldo, May 2020.
'''
import __includes__

import numpy as np
from dynamics.ode_solver import OdeIntegration



def rhs(t, x, b, c):
    theta, omega = x
    dxdt = [omega, -b * omega - c * np.sin(theta)]
    return dxdt

def test_odeint():
    x0 = [np.pi - 0.1, 0.0]
    b  = 0.25
    c  = 5.00
    t0 = 0
    tN = 10
    N  = 100
    t  = np.linspace(t0, tN, N)
    ODE = OdeIntegration(rhs, x0, t, params=(b,c), backend='scipy_odeint',atol=1e-12,rtol=1e-14)
    t, sol = ODE.solve_ode()
    tol = 1e-10
    assert((sol[ 2,0] <  3.0314059274697405 +tol) & (sol[2,0] >  3.0314059274697405 -tol))
    assert((sol[-1,0] <  0.02001153087847384+tol) & (sol[2,0] >  0.02001153087847384-tol))
    assert((sol[ 2,1] < -0.10169816769173402+tol) & (sol[2,0] > -0.10169816769173402-tol))
    assert((sol[-1,1] <  1.5678182626646398 +tol) & (sol[2,0] >  1.5678182626646398 -tol))

def test_ode():
    x0 = [np.pi - 0.1, 0.0]
    b  = 0.25
    c  = 5.00
    t0 = 0
    tN = 10
    N  = 100
    t  = np.linspace(t0, tN, N)
    ODE_dopri5 = OdeIntegration(rhs, x0, t, params=(b,c), backend='scipy_ode', approach='dopri5', atol=1e-10, rtol=1e-12)
    ODE_dop853 = OdeIntegration(rhs, x0, t, params=(b,c), backend='scipy_ode', approach='dop853', atol=1e-10, rtol=1e-12)
    t_dopri5, sol_dopri5 = ODE_dopri5.solve_ode()
    t_dop853, sol_dop853 = ODE_dop853.solve_ode()
    tol = 1e-10
    # dopri5
    assert((sol_dopri5[ 2,0] <  3.0314059274681604  +tol) & (sol_dopri5[2,0] >  3.0314059274681604  -tol))
    assert((sol_dopri5[-1,0] <  0.02001153103560903 +tol) & (sol_dopri5[2,0] >  0.02001153103560903 -tol))
    assert((sol_dopri5[ 2,1] < -0.10169816768921475 +tol) & (sol_dopri5[2,0] > -0.10169816768921475 -tol))
    assert((sol_dopri5[-1,1] <  1.5678182623726236  +tol) & (sol_dopri5[2,0] >  1.5678182623726236  -tol))
    # dop853
    assert((sol_dop853[ 2,0] <  3.0314059274701837  +tol) & (sol_dop853[2,0] >  3.0314059274701837  -tol))
    assert((sol_dop853[-1,0] <  0.020011530870941874+tol) & (sol_dop853[2,0] >  0.020011530870941874-tol))
    assert((sol_dop853[ 2,1] < -0.1016981676890163  +tol) & (sol_dop853[2,0] > -0.1016981676890163  -tol))
    assert((sol_dop853[-1,1] <  1.5678182626491397  +tol) & (sol_dop853[2,0] >  1.5678182626491397  -tol))

def test_solve_ivp():

    x0 = [np.pi - 0.1, 0.0]
    b  = 0.25
    c  = 5.00
    t0 = 0
    tN = 10
    N  = 100
    t  = np.linspace(t0, tN, N)
    ODE_rk45   = OdeIntegration(rhs, x0, t, params=(b,c), backend='scipy_solve_ivp', approach='RK45'  , atol=1e-10, rtol=1e-12)
    ODE_rk23   = OdeIntegration(rhs, x0, t, params=(b,c), backend='scipy_solve_ivp', approach='RK23'  , atol=1e-10, rtol=1e-12)
    ODE_dop853 = OdeIntegration(rhs, x0, t, params=(b,c), backend='scipy_solve_ivp', approach='DOP853', atol=1e-10, rtol=1e-12)
    ODE_radau  = OdeIntegration(rhs, x0, t, params=(b,c), backend='scipy_solve_ivp', approach='Radau' , atol=1e-10, rtol=1e-12)
    ODE_bdf    = OdeIntegration(rhs, x0, t, params=(b,c), backend='scipy_solve_ivp', approach='BDF'   , atol=1e-10, rtol=1e-12)
    ODE_lsoda  = OdeIntegration(rhs, x0, t, params=(b,c), backend='scipy_solve_ivp', approach='LSODA' , atol=1e-10, rtol=1e-12)
    t_rk45  , sol_rk45   = ODE_rk45  .solve_ode()
    t_rk23  , sol_rk23   = ODE_rk23  .solve_ode()
    t_dop853, sol_dop853 = ODE_dop853.solve_ode()
    t_radau , sol_radau  = ODE_radau .solve_ode()
    t_bdf   , sol_bdf    = ODE_bdf   .solve_ode()
    t_lsoda , sol_lsoda  = ODE_lsoda .solve_ode()
    tol = 1e-10
    # RK45
    assert((sol_rk45[ 2,0] <  3.0314059274638745  +tol) & (sol_rk45[2,0] >  3.0314059274638745  -tol))
    assert((sol_rk45[-1,0] <  0.020011531247159438+tol) & (sol_rk45[2,0] >  0.020011531247159438-tol))
    assert((sol_rk45[ 2,1] < -0.10169816770133835 +tol) & (sol_rk45[2,0] > -0.10169816770133835 -tol))
    assert((sol_rk45[-1,1] <  1.5678182620108059  +tol) & (sol_rk45[2,0] >  1.5678182620108059  -tol))
    # RK23
    assert((sol_rk23[ 2,0] <  3.031405927512424   +tol) & (sol_rk23[2,0] >  3.031405927512424   -tol))
    assert((sol_rk23[-1,0] <  0.020011530409670374+tol) & (sol_rk23[2,0] >  0.020011530409670374-tol))
    assert((sol_rk23[ 2,1] < -0.1016981676684984  +tol) & (sol_rk23[2,0] > -0.1016981676684984  -tol))
    assert((sol_rk23[-1,1] <  1.5678182606997646  +tol) & (sol_rk23[2,0] >  1.5678182606997646  -tol))
    # DOP853
    assert((sol_dop853[ 2,0] <  3.0314059275594376  +tol) & (sol_dop853[2,0] >  3.0314059275594376  -tol))
    assert((sol_dop853[-1,0] <  0.020011530866711202+tol) & (sol_dop853[2,0] >  0.020011530866711202-tol))
    assert((sol_dop853[ 2,1] < -0.1016981677472088  +tol) & (sol_dop853[2,0] > -0.1016981677472088  -tol))
    assert((sol_dop853[-1,1] <  1.5678182626547512  +tol) & (sol_dop853[2,0] >  1.5678182626547512  -tol))
    # Radau
    assert((sol_radau[ 2,0] <  3.0314059274706118 +tol) & (sol_radau[2,0] >  3.0314059274706118 -tol))
    assert((sol_radau[-1,0] <  0.02001153087835398+tol) & (sol_radau[2,0] >  0.02001153087835398-tol))
    assert((sol_radau[ 2,1] < -0.1016981676888477 +tol) & (sol_radau[2,0] > -0.1016981676888477 -tol))
    assert((sol_radau[-1,1] <  1.567818262657624  +tol) & (sol_radau[2,0] >  1.567818262657624  -tol))
    # BDF
    assert((sol_bdf[ 2,0] <  3.031405926858494   +tol) & (sol_bdf[2,0] >  3.031405926858494  -tol))
    assert((sol_bdf[-1,0] <  0.020011534531785065+tol) & (sol_bdf[2,0] >  0.020011534531785065-tol))
    assert((sol_bdf[ 2,1] < -0.10169816850775981 +tol) & (sol_bdf[2,0] > -0.10169816850775981  -tol))
    assert((sol_bdf[-1,1] <  1.5678182631820434  +tol) & (sol_bdf[2,0] >  1.5678182631820434  -tol))
    # LSODA
    assert((sol_lsoda[ 2,0] <  3.031405927405019   +tol) & (sol_lsoda[2,0] >  3.031405927405019   -tol))
    assert((sol_lsoda[-1,0] <  0.020011531078441416+tol) & (sol_lsoda[2,0] >  0.020011531078441416-tol))
    assert((sol_lsoda[ 2,1] < -0.10169816787340426 +tol) & (sol_lsoda[2,0] > -0.10169816787340426 -tol))
    assert((sol_lsoda[-1,1] <  1.5678182627386206  +tol) & (sol_lsoda[2,0] >  1.5678182627386206  -tol))


if __name__ == "__main__":
    test_odeint()
    test_ode()
    test_solve_ivp()
