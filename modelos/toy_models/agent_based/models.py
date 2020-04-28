import numpy as np
import pandas as pd
from sympy.combinatorics.permutations import Permutation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def evolui(modelo, pop, params):
    
    if modelo == 'SIR':
        
        indices = [i for i in range(len(pop))]
        p = np.random.permutation(indices)
        ciclos = Permutation(p).cyclic_form
        new_pop = pop.copy()
        not_in_ciclos = list(set(indices)-set([c for ciclo in ciclos for c in ciclo]))
        
        for ciclo in ciclos:
            
            for a in ciclo:
                
                n = ciclo.index(a)
                
                if pop[a] == 2 and np.random.rand() <= params[1]:
                    
                    new_pop[a] = 3
                    
                    continue
                    
                else:
                    
                    if pop[a] == 2 and pop[ciclo[(n-1)%len(ciclo)]] == 1 and np.random.rand() <= params[0]:
                        
                        new_pop[ciclo[(n-1)%len(ciclo)]] = 2
                        
                    if pop[a] == 2 and pop[ciclo[(n+1)%len(ciclo)]] == 1 and np.random.rand() <= params[0]:
                        
                        new_pop[ciclo[(n+1)%len(ciclo)]] = 2

        for x in not_in_ciclos:
            
            if pop[x] == 2 and np.random.rand() <= params[1]:
                new_pop[x] = 3
                
        return np.array(new_pop)
    
    elif modelo == 'SEIR':
        
        indices = [i for i in range(len(pop))]
        p = np.random.permutation(indices)
        ciclos = Permutation(p).cyclic_form
        new_pop = pop.copy()
        not_in_ciclos = list(set(indices)-set([c for ciclo in ciclos for c in ciclo]))
        
        for ciclo in ciclos:
            
            for a in ciclo:
                
                n = ciclo.index(a)
                
                if pop[a] == 2 and np.random.rand() <= params[1]:
                    
                    new_pop[a] = 3
                    
                if pop[a] == 3 and np.random.rand() <= params[2]:
                    
                    new_pop[a] = 4
                    
                else:
                    
                    if pop[a] == 3 and pop[ciclo[(n-1)%len(ciclo)]] == 1 and np.random.rand() <= params[0]:
                        
                        new_pop[ciclo[(n-1)%len(ciclo)]] = 2
                        
                    if pop[a] == 3 and pop[ciclo[(n+1)%len(ciclo)]] == 1 and np.random.rand() <= params[0]:
                        
                        new_pop[ciclo[(n+1)%len(ciclo)]] = 2

        for x in not_in_ciclos:
            
            if pop[x] == 3 and np.random.rand() <= params[2]:
                
                new_pop[x] = 4
                continue
                
            if pop[x] == 2 and np.random.rand() <= params[1]:
                
                new_pop[x] = 3
                continue
                
        return np.array(new_pop)
    
    elif modelo == 'SIS':
        
        indices = [i for i in range(len(pop))]
        fator_de_transmissao ,fator_de_recuperacao = params[0], params[1]
        p = np.random.permutation(indices)
        ciclos = Permutation(p).cyclic_form
        new_pop = pop.copy()
        not_in_ciclos = list(set(indices)-set([c for ciclo in ciclos for c in ciclo]))
        
        for ciclo in ciclos:
            
            for a in ciclo:
                
                n = ciclo.index(a)
                
                if pop[a] == 2 and np.random.rand() <= fator_de_recuperacao:
                    
                    new_pop[a] = 1
                    
                else:
                    
                    if pop[a] == 2 and pop[ciclo[(n-1)%len(ciclo)]] == 1 and np.random.rand() <= fator_de_transmissao:
                        
                        new_pop[ciclo[(n-1)%len(ciclo)]] = 2
                        
                    if pop[a] == 2 and pop[ciclo[(n+1)%len(ciclo)]] == 1 and np.random.rand() <= fator_de_transmissao:
                        
                        new_pop[ciclo[(n+1)%len(ciclo)]] = 2

        for x in not_in_ciclos:
            
            if pop[x] == 2 and np.random.rand() <= fator_de_recuperacao:
                
                new_pop[x] = 1
                continue
                
        return np.array(new_pop)



def simulador(modelo, CI, params, tmax, simuls, alpha):
    
    if modelo == 'SIR':
        
       
        populacoes = [[CI[0]][0] for i in range(simuls)]
        num_suscetiveis = [np.array([CI[1]]) for i in range(simuls)]
        num_infectados = [np.array([CI[2]]) for i in range(simuls)]
        num_recuperados = [np.array([CI[3]]) for i in range(simuls)]
        suscetiveis_mean = np.zeros_like(np.arange(tmax))
        infectados_mean = np.zeros_like(np.arange(tmax))
        recuperados_mean = np.zeros_like(np.arange(tmax))
        
        for t in range(1, tmax):
            
            for i in range(simuls):
                
                populacoes[i] = evolui('SIR', populacoes[i], params)
                num_suscetiveis[i] = np.hstack([num_suscetiveis[i], np.count_nonzero(populacoes[i] == 1)])       
                num_infectados[i] = np.hstack([num_infectados[i], np.count_nonzero(populacoes[i] == 2)])        
                num_recuperados[i] = np.hstack([num_recuperados[i], np.count_nonzero(populacoes[i] == 3)])
                
        data = []
        
        for i in range(simuls):
            
            data += [go.Scatter(x = np.arange(len(num_suscetiveis[i])), y = np.array(num_suscetiveis[i]), mode = 'lines', 
                                showlegend = False, legendgroup = 'Suscetiveis', name = 'Suscetiveis', 
                                marker = dict(color = 'rgba(0,0,252,'+str(alpha)+')')),
                     go.Scatter(x = np.arange(len(num_suscetiveis[i])), y = np.array(num_infectados[i]), mode = 'lines', 
                                showlegend = False, legendgroup = 'Infectados', name = 'Infectados', 
                                marker = dict(color = 'rgba(252,0,0,'+str(alpha)+')')),
                     go.Scatter(x = np.arange(len(num_suscetiveis[i])), y = np.array(num_recuperados[i]), mode = 'lines', 
                                showlegend = False, legendgroup = 'Recuperados', name = 'Recuperados', 
                                marker = dict(color = 'rgba(0,252,0,'+str(alpha)+')'))]
            
            suscetiveis_mean += np.array(num_suscetiveis[i])
            infectados_mean += np.array(num_infectados[i])
            recuperados_mean += np.array(num_recuperados[i])
            
        suscetiveis_mean = 1/simuls * suscetiveis_mean
        infectados_mean = 1/simuls * infectados_mean
        recuperados_mean = 1/simuls * recuperados_mean
        
        data += [go.Scatter(x = np.arange(len(suscetiveis_mean)), y = np.array(suscetiveis_mean), mode = 'lines', 
                            showlegend = True, legendgroup = 'Suscetiveis', name = 'Suscetiveis', 
                            marker = dict(color = 'blue')),
                 go.Scatter(x = np.arange(len(suscetiveis_mean)), y = np.array(infectados_mean), mode = 'lines', 
                            showlegend = True, legendgroup = 'Infectados', name = 'Infectados', 
                            marker = dict(color = 'red')),
                 go.Scatter(x = np.arange(len(suscetiveis_mean)), y = np.array(recuperados_mean), mode = 'lines', 
                            showlegend = True, legendgroup = 'Recuperados', name = 'Recuperados', 
                            marker = dict(color = 'green'))
                ]
        
        return data
    
    elif modelo == 'SEIR':
        
       
        populacoes = [[CI[0]][0].copy() for i in range(simuls)]
        num_suscetiveis = [np.array([CI[1]]) for i in range(simuls)]
        num_expostos = [np.array([CI[2]]) for i in range(simuls)]
        num_infectados = [np.array([CI[3]]) for i in range(simuls)]
        num_recuperados = [np.array([CI[4]]) for i in range(simuls)]
        suscetiveis_mean = np.zeros_like(np.arange(tmax))
        expostos_mean = np.zeros_like(np.arange(tmax))
        infectados_mean = np.zeros_like(np.arange(tmax))
        recuperados_mean = np.zeros_like(np.arange(tmax))
        
        for t in range(1, tmax):
            
            for i in range(simuls):
                
                populacoes[i] = evolui('SEIR', populacoes[i], params)
                num_suscetiveis[i] = np.hstack([num_suscetiveis[i], np.count_nonzero(populacoes[i] == 1)])
                num_expostos[i] = np.hstack([num_expostos[i], np.count_nonzero(populacoes[i] == 2)])
                num_infectados[i] = np.hstack([num_infectados[i], np.count_nonzero(populacoes[i] == 3)])        
                num_recuperados[i] = np.hstack([num_recuperados[i], np.count_nonzero(populacoes[i] == 4)])
                
        data = []
        
        for i in range(simuls):
            
            data += [go.Scatter(x = np.arange(len(num_suscetiveis[i])), y = np.array(num_suscetiveis[i]), mode = 'lines', 
                                showlegend = False, legendgroup = 'Suscetiveis', name = 'Suscetiveis', 
                                marker = dict(color = 'rgba(0,0,252,'+str(alpha)+')')),
                     go.Scatter(x = np.arange(len(num_suscetiveis[i])), y = np.array(num_expostos[i]), mode = 'lines', 
                                showlegend = False, legendgroup = 'Expostos', name = 'Expostos', 
                                marker = dict(color = 'rgba(249,166,2,'+str(alpha)+')')),
                     go.Scatter(x = np.arange(len(num_suscetiveis[i])), y = np.array(num_infectados[i]), mode = 'lines', 
                                showlegend = False, legendgroup = 'Infectados', name = 'Infectados', 
                                marker = dict(color = 'rgba(252,0,0,'+str(alpha)+')')),
                     go.Scatter(x = np.arange(len(num_suscetiveis[i])), y = np.array(num_recuperados[i]), mode = 'lines', 
                                showlegend = False, legendgroup = 'Recuperados', name = 'Recuperados', 
                                marker = dict(color = 'rgba(0,252,0,'+str(alpha)+')'))]
            
            suscetiveis_mean += np.array(num_suscetiveis[i])
            expostos_mean += np.array(num_expostos[i])
            infectados_mean += np.array(num_infectados[i])
            recuperados_mean += np.array(num_recuperados[i])
            
        suscetiveis_mean = 1/simuls * suscetiveis_mean
        expostos_mean = 1/simuls * expostos_mean
        infectados_mean = 1/simuls * infectados_mean
        recuperados_mean = 1/simuls * recuperados_mean
        
        data += [go.Scatter(x = np.arange(len(suscetiveis_mean)), y = np.array(suscetiveis_mean), mode = 'lines', 
                            showlegend = True, legendgroup = 'Suscetiveis', name = 'Suscetiveis', 
                            marker = dict(color = 'blue')),
                 go.Scatter(x = np.arange(len(suscetiveis_mean)), y = np.array(expostos_mean), mode = 'lines', 
                            showlegend = True, legendgroup = 'Expostos', name = 'Expostos', 
                            marker = dict(color = 'rgba(249,166,2,1)')),
                 go.Scatter(x = np.arange(len(suscetiveis_mean)), y = np.array(infectados_mean), mode = 'lines', 
                            showlegend = True, legendgroup = 'Infectados', name = 'Infectados', 
                            marker = dict(color = 'red')),
                 go.Scatter(x = np.arange(len(suscetiveis_mean)), y = np.array(recuperados_mean), mode = 'lines', 
                            showlegend = True, legendgroup = 'Recuperados', name = 'Recuperados', 
                            marker = dict(color = 'green'))
                ]
        
        return data
    
    elif modelo == 'SIS':
        
       
        populacoes = [[CI[0]][0] for i in range(simuls)]
        num_suscetiveis = [np.array([CI[1]]) for i in range(simuls)]
        num_infectados = [np.array([CI[2]]) for i in range(simuls)]
        suscetiveis_mean = np.zeros_like(np.arange(tmax))
        infectados_mean = np.zeros_like(np.arange(tmax))
        
        for t in range(1, tmax):
            
            for i in range(simuls):
                
                populacoes[i] = evolui('SIS', populacoes[i], params)
                num_suscetiveis[i] = np.hstack([num_suscetiveis[i], np.count_nonzero(populacoes[i] == 1)])       
                num_infectados[i] = np.hstack([num_infectados[i], np.count_nonzero(populacoes[i] == 2)])
                
        data = []
        
        for i in range(simuls):
            
            data += [go.Scatter(x = np.arange(len(num_suscetiveis[i])), y = np.array(num_suscetiveis[i]), mode = 'lines', 
                                showlegend = False, legendgroup = 'Suscetiveis', name = 'Suscetiveis', 
                                marker = dict(color = 'rgba(0,0,252,'+str(alpha)+')')),
                     go.Scatter(x = np.arange(len(num_suscetiveis[i])), y = np.array(num_infectados[i]), mode = 'lines', 
                                showlegend = False, legendgroup = 'Infectados', name = 'Infectados', 
                                marker = dict(color = 'rgba(252,0,0,'+str(alpha)+')'))]
            
            suscetiveis_mean += np.array(num_suscetiveis[i])
            infectados_mean += np.array(num_infectados[i])
            
        suscetiveis_mean = 1/simuls * suscetiveis_mean
        infectados_mean = 1/simuls * infectados_mean
        
        data += [go.Scatter(x = np.arange(len(suscetiveis_mean)), y = np.array(suscetiveis_mean), mode = 'lines', 
                            showlegend = True, legendgroup = 'Suscetiveis', name = 'Suscetiveis', 
                            marker = dict(color = 'blue')),
                 go.Scatter(x = np.arange(len(suscetiveis_mean)), y = np.array(infectados_mean), mode = 'lines', 
                            showlegend = True, legendgroup = 'Infectados', name = 'Infectados', 
                            marker = dict(color = 'red'))
                ]
        
        return data
    
    
    
def get_CI(modelo, num_individuos, num_infectados0):
    
    if modelo == 'SIR':
    
        populacao0 = np.ones(num_individuos)
        infectados0 = np.random.choice(num_individuos, num_infectados0)
        populacao0[infectados0] = 2*np.ones(num_infectados0)
        CI = [populacao0, num_individuos-num_infectados0, num_infectados0, 0]

        return CI

    elif modelo == 'SEIR':
    
        populacao0 = np.ones(num_individuos)
        infectados0 = np.random.choice(num_individuos, num_infectados0)
        populacao0[infectados0] = 3*np.ones(num_infectados0)
        CI = [populacao0, num_individuos-num_infectados0, 0, num_infectados0, 0]

        return CI

    elif modelo == 'SIS':
        
        populacao0 = np.ones(num_individuos)
        infectados0 = np.random.choice(num_individuos, num_infectados0)
        populacao0[infectados0] = 2*np.ones(num_infectados0)
        CI = [populacao0, num_individuos-num_infectados0, num_infectados0]

        return CI