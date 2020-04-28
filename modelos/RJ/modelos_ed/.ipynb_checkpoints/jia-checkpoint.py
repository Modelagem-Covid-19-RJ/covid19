import numpy as np
from scipy.integrate import solve_ivp

def F(t, v, beta, theta, p, lamb, sigma, rho, epsA, gammaA, epsI, gammaI, deathI, gammaD, deathD):
    """Differential equation for SEIR-QAD model"""
    S,Q,E,A,I,D,R = v
    dS = - beta*S*(I + theta*A) - p*S + lamb*Q
    dQ = p*S - lamb*Q
    dE = beta*S*(I + theta*A) - sigma*E
    dA = sigma*(1 - rho)*E - epsA*A - gammaA*A
    dI = sigma*rho*E - epsI*I - gammaI*I - deathI*I
    dD = epsA*A + epsI*I - gammaD*D - deathD*D
    dR = gammaA*A + gammaI*I + gammaD*D
    return np.array([dS, dQ, dE, dA, dI, dD, dR])

def multi_regime(CI, t0, t_par_list):
    tmp_results = [np.array(CI).reshape((-1,1))]
    for (tj, par) in t_par_list:
        f = lambda t,v: F(t,v, *par)
        res = solve_ivp(f, (t0,tj), CI, t_eval=np.arange(t0+1,tj+1))
        t0 = tj
        CI = res.y[:,-1]
        tmp_results.append(res.y)
    return np.hstack(tmp_results)

def new_diagn(res, params):
    epsA, epsI = params[6], params[8]
    return res[3,:]*epsA + res[4,:]*epsI
def new_deaths(res, params):
    dI, dD = params[10], params[12]
    return res[4,:]*dI + res[5,:]*dD
def tot_deaths(res):
    return np.sum(res[:,0]) - np.sum(res, axis=0)
