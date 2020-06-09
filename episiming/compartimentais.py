#!/anaconda3/envs/py38/bin/python
# -*- coding: utf-8 -*-
"""
Módulo para simulação de modelos epidemiológicos do tipo compartimental.
"""

import numpy as np

from os import path
from collections import namedtuple

from scipy.integrate import solve_ivp

def evolucao_SIR(pop_0, beta, gamma, tempos, plot=False):
    """
    Implementação do modelo SIR compartimental
    """

    SIR_Compartimental = namedtuple('SIR_Compartimental', 
                                 [
                                     'pop_0',
                                     'num_pop',
                                     'beta',
                                     'gamma',
                                     'tempos',
                                     'S',
                                     'I', 
                                     'R',
                                     'X'
                                 ])        
          
    def diferencial(t, X, N, beta, gamma):
        S, I = X
        dXdt = [- beta*I*S/N, beta*I*S/N - gamma*I]
        return dXdt        
    
    num_pop = sum(pop_0)
    y0 = pop_0[0:2]
    sol = solve_ivp(diferencial, t_span=[tempos[0],tempos[-1]], 
                    y0 = y0, t_eval = tempos,
                    args=(num_pop, beta, gamma))
    
    resultado = SIR_Compartimental(pop_0, num_pop, beta, gamma, tempos,
                                   sol.y[0], sol.y[1], 
                                   num_pop - sol.y[0] - sol.y[1], 
                                   sol)
    return resultado

def evolucao_SEIRMQDA(pop_0, p, lamb, beta, theta, sigma, rho, gammaA,
                      epsA, gammaI, epsI, dI, gammaD, dD,tempos,
                      plot=False):
    """
    Implementação do modelo SEIRMQDA compartimental.
    """

    SEIRMQDA_Compartimental = namedtuple('SEIRMQDA_Compartimental', 
                                 [
                                     'pop_0',
                                     'num_pop',
                                     'p',
                                     'lamb',
                                     'beta',
                                     'theta',
                                     'sigma',
                                     'rho',
                                     'gammaA',
                                     'epsA',
                                     'gammaI',
                                     'epsI',
                                     'dI',
                                     'gammaD',
                                     'dD',
                                     'tempos',
                                     'S',
                                     'E',
                                     'I', 
                                     'R',
                                     'M',
                                     'Q',
                                     'D',
                                     'A',
                                     'X'
                                 ])        
          
    def diferencial(t, X, N, p, lamb, beta, theta, sigma, rho, gammaA,
                    epsA, gammaI, epsI, dI, gammaD, dD):
        S, E, I, R, M, Q, D, A = X
        dXdt = [- beta*I*S/N -p*S + lamb*Q - theta*A*S/N, 
                beta*I*S/N - sigma*E + theta*A*S/N, 
                sigma*rho*E - gammaI*I - dI*I - epsI*I,
                gammaI*I + gammaA*A, 
                dD*D + dI*I, 
                p*S - lamb*Q, 
                epsI*I + epsA*A - gammaD*D - dD*D ,
                sigma*(1-rho)*E - gammaA*A - epsA*A]
        return dXdt        
    
    num_pop = sum(pop_0)
    y0 = [num_pop-I_0, 0, I_0, 0, 0, 0, 0, 0]
    sol = solve_ivp(diferencial, t_span=[tempos[0],tempos[-1]], 
                    y0 = y0, t_eval = tempos,
                    args=(num_pop, p, lamb, beta, theta, sigma, rho, 
                          gammaA, epsA, gammaI, epsI, dI, gammaD, dD))
    
    resultado = SEIRMQDA_Compartimental(
        pop_0, num_pop, p, lamb, beta, theta, sigma,rho, gammaA, epsA, 
        gammaI, epsI, dI, gammaD, dD, tempos,
        sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4], 
        sol.y[5], sol.y[6], sol.y[7], sol)
    return resultado