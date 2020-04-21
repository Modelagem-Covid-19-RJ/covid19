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

def F_Quarentena(t, v, betaS, theta, p, lamb, sigma, rho, epsA, gammaA, epsI, gammaI, deathI, gammaD, deathD):
    """Differential equation for SEIR-QAD model"""
    S,Q,E,A,I,D,R = v
    dS = - betaS*S*(I + theta*A) - p*S + lamb*Q
    dQ = p*S - lamb*Q -betaQ*Q*(I+theta*A)
    dE = betaS*S*(I + theta*A) - sigma*E + betaQ*Q*(I+theta*A)
    dA = sigma*(1 - rho)*E - epsA*A - gammaA*A
    dI = sigma*rho*E - epsI*I - gammaI*I - deathI*I
    dD = epsA*A + epsI*I - gammaD*D - deathD*D
    dR = gammaA*A + gammaI*I + gammaD*D
    return np.array([dS, dQ, dE, dA, dI, dD, dR])


def G(t, v, pars):
    """Differential equation for SEIR-QAD 2 City model"""
    S = [v[0], v[len(v)//2]]
    Q = [v[1], v[1 + len(v)//2]]
    E = [v[2], v[2 + len(v)//2]]
    A = [v[3], v[3 + len(v)//2]]
    I = [v[4], v[4 + len(v)//2]]
    D = [v[5], v[5 + len(v)//2]]
    R = [v[6], v[6 + len(v)//2]]
    
    beta, theta, p, lamb, sigma, rho, epsA, gammaA, epsI, gammaI, deathI, gammaD, deathD, mu, delta = pars
    
    dS = [-beta[0]*S[0]*(I[0] + theta[0]*A[0]) - p[0]*S[0] + lamb[0]*Q[0] - mu[0]*S[0] + mu[1]*S[1],-beta[1]*S[1]*(I[1] + theta[1]*A[1]) - p[1]*S[1] + lamb[1]*Q[1] - mu[1]*S[1] + mu[0]*S[0]]
    dQ = [p[0]*S[0] - lamb[0]*Q[0], p[1]*S[1] - lamb[1]*Q[1]]
    dE = [beta[0]*S[0]*(I[0] + theta[0]*A[0]) - sigma[0]*E[0] + mu[0]*E[0] - mu[1]*E[1], beta[1]*S[1]*(I[1] + theta[1]*A[1]) - sigma[1]*E[1] - mu[0]*E[0] + mu[1]*E[1]]
    dA = [sigma[0]*(1 - rho[0])*E[0] - epsA[0]*A[0]- gammaA[0]*A[0] - mu[0]*A[0] + mu[1]*A[1], sigma[1]*(1 - rho[1])*E[1] - epsA[1]*A[1]- gammaA[1]*A[1] + mu[0]*A[0] - mu[1]*A[1]]
    dI = [sigma[0]*rho[0]*E[0] - epsI[0]*I[0] - gammaI[0]*I[0] - deathI[0]*I[0] - delta[0]*I[0] + delta[1]*I[1], sigma[1]*rho[1]*E[1] - epsI[1]*I[1] - gammaI[1]*I[1] - deathI[1]*I[1] + delta[0]*I[0] - delta[1]*I[1]]
    dD = [epsA[0]*A[0] + epsI[0]*I[0] - gammaD[0]*D[0] - deathD[0]*D[0], epsA[1]*A[1] + epsI[1]*I[1] - gammaD[1]*D[1] - deathD[1]*D[1]]
    dR = [gammaA[0]*A[0] + gammaI[0]*I[0] + gammaD[0]*D[0] + mu[1]*R[1] - mu[0]*R[0], gammaA[1]*A[1] + gammaI[1]*I[1] + gammaD[1]*D[1] - mu[1]*R[1] + mu[0]*R[0]]
    return np.array([dS[0], dQ[0], dE[0], dA[0], dI[0], dD[0], dR[0], dS[1], dQ[1], dE[1], dA[1], dI[1], dD[1], dR[1]])

def multi_regime(CI, t0, t_par_list):
    tmp_results = [np.array(CI).reshape((-1,1))]
    for (tj, par) in t_par_list:
        f = lambda t,v: F(t,v, *par)
        res = solve_ivp(f, (t0,tj), CI, t_eval=np.arange(t0+1,tj+1))
        t0 = tj
        CI = res.y[:,-1]
        tmp_results.append(res.y)
    return np.hstack(tmp_results)

def multi_regime_cidades(CI, t0, t_par_list):
    tmp_results = [np.array(CI).reshape((-1,1))]
    for (tj, par) in t_par_list:
        g = lambda t,v: G(t,v, par)
        res = solve_ivp(g, (t0,tj), CI, t_eval=np.arange(t0+1,tj+1))
        t0 = tj
        CI = res.y[:,-1]
        tmp_results.append(res.y)
    return np.hstack(tmp_results)

def new_diagn(res, params):
    epsA, epsI = params[6], params[8]
    return [res[3,:]*epsA[0] + res[4,:]*epsI[0],res[3+len(res)//2,:]*epsA[1] + res[4+len(res)//2,:]*epsI[1]]
def new_deaths(res, params):
    dI, dD = params[10], params[12]
    return [res[4,:]*dI[0] + res[5,:]*dD[0], res[4+len(res)//2,:]*dI[1] + res[5+len(res)//2,:]*dD[1]]
def tot_deaths(res):
    return np.sum(res[:,0]) - np.sum(res, axis=0)
