#!/anaconda3/envs/py38/bin/python
# -*- coding: utf-8 -*-
"""
Módulo para simulação de modelos epidemiológicos baseados em indivíduos.
"""

import random

import numpy as np
import networkx as nx

from numba import njit
from numba.typed import List

from os import path
from sys import getsizeof
from collections import namedtuple
from functools import partial

from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as c

bernoulli = False

def color_to_rgba(cor, alpha):
    cor_rgba = tuple(np.array(c.to_rgba(cor, alpha = alpha/255))*255)
    return 'rgba'+ str(cor_rgba)

def passo_vetorial(pop_estado, redes, redes_tx_transmissao,
                   pop_fator_tx_transmissao_c, prob_nao_recuperacao,
                   pop_posicoes, f_kernel, dt):

    #
    # calcula número de indivíduos
    #
    num_pop = len(pop_estado)

    #
    # separa os suscetíveis, criando um vetor de 1's e 0's, se for, ou não, suscetível
    #
    pop_suscetiveis = np.select([pop_estado==1], [pop_estado])

    #
    # separa os infectados, criando um vetor de 1's e 0's, se for, ou não, infectado/contagioso
    #
    pop_infectados = np.select([pop_estado==2], [pop_estado])/2

    #
    # cria lista de grafos orientados para contatos com risco de contágio em cada rede
    # copia os vértices de cada rede
    # copia apenas as arestas que ligam um infectado (k) a um suscetível (i)
    #
    
    contatos_de_risco_rs = np.zeros((len(redes_tx_transmissao), num_pop))

    for j in range(len(redes_tx_transmissao)):
        for (i,k) in redes[j].edges:
            if pop_infectados[k] and pop_suscetiveis[i]:
                contatos_de_risco_rs[j][i] += 1
            elif pop_infectados[i] and pop_suscetiveis[k]:
                contatos_de_risco_rs[j][k] += 1
               
    contatos_de_risco_c = np.array(
            [np.dot(pop_infectados,
                    f_kernel(np.linalg.norm(pop_posicoes - pop_posicoes[i], 
                             axis=1)
                            )
                    )
             for i in range(num_pop)
            ]
        )

    
    lambda_rate = ((redes_tx_transmissao * contatos_de_risco_rs).sum(axis=0) +
                   pop_fator_tx_transmissao_c * contatos_de_risco_c)
    
    prob_nao_contagio = np.exp(-dt*lambda_rate)                             

    # `sorteio` será usado independentemente n a seguir, por isso
    # é calculado aqui já para os dois.
    sorteio = np.random.rand(num_pop)

    pop_novos_infectados = np.select([sorteio > prob_nao_contagio], [np.ones(num_pop)])

    pop_novos_recuperados = np.select([pop_infectados * sorteio > prob_nao_recuperacao], 
                                      [np.ones(num_pop)])
    
    # retorna a população atualizada adicionando '1' aos que avançaram de estágio

    return pop_estado + pop_novos_infectados + pop_novos_recuperados

def evolucao_vetorial(pop_estado_0, pop_posicoes, redes, redes_tx_transmissao,
                      pop_fator_tx_transmissao_c, gamma, f_kernel,
                      dados_temporais, num_sim, show=''):
    """Evolução temporal da epidemia em um grafo estruturado.

    Entrada:
        pop_0: numpy.array
            state of the population, with 
                1: suscetível
                2: infectado
                3: recuperado ou removido

        G: numpy.array
            grafo de conexões, com atributo `taxa de transmissao`

        kernel: function
            função 'kernel' decaindo com distância entre a posição dos indivíduos

        dados_temporais: list
            [t_0, dt, num_dt]

        num_sim: int
            número de simulações

        show: str
            indica se é para exibir um gráfico e de que tipo:
                - 'nuvem': exibe uma nuvem com todas as simulações e o valor médio em destaque
                - 'sd': exibe o valor médio com uma faixa dada pelo desvio padrão
                - 'sdc': exibe o valor médio com uma faixa dada pelo desvio padrão corrigido
                - 'medio': exibe apenas o valor médio
                - '': não exibe gráfico algum.

    Saída
        X: class.SIR_Individual
            Uma instância da classe `SIR_Individual` com os seguintes atributos:
                tempos:
                num_sim:
                S_medio:
                I_medio: 
                R_medio:
                I_sigma:
                R_sigma:
                S_sigma:
    """  

    # confere se escolha para `show` é válida    
    if show:
        assert(show in ('nuvem', 'sd', 'sdc', 'media')), 'Valor inválido para argumento `show`.'

    # atributos de saída
    SIR_Individual = namedtuple('SIR_Individual', 
                                [
                                    'tempos',
                                    'S_medio',
                                    'I_medio', 
                                    'R_medio',
                                    'S_sigma',
                                    'I_sigma', 
                                    'R_sigma'
                                ])
    
    # número de indivíduos da população
    num_pop = len(pop_estado_0)
    num_inf_0 = np.count_nonzero(pop_estado_0==2)        

    # número de instantes no tempo e passos de tempo
    t_0 = dados_temporais[0]
    dt = dados_temporais[1]
    num_dt = dados_temporais[2]
    tempos = np.linspace(t_0, num_dt*dt, num_dt+1)
    
    # ajusta dados da rede dependendo se há mais de uma ou não
    #
    if type(redes) == list or type(redes_tx_transmissao) == list:
        assert(len(redes) == len(redes_tx_transmissao)), 'redes e taxas de transmissão \
        \ devem ter o mesmo número de elementos.'
    else:
        redes = [redes]
        redes_tx_transmissao = [redes_tx_transmissao]
    
    # calcula propabilidade de não recuperação
    prob_nao_recuperacao = np.exp(-dt*gamma)

    # inicializa variáveis para o cálculo da média
    S_medio = np.zeros(num_dt+1)
    I_medio = np.zeros(num_dt+1)
    R_medio = np.zeros(num_dt+1)

    # inicializa variáveis para o cálculo do desvio padrão
    S_sigma = np.zeros(num_dt+1)
    I_sigma = np.zeros(num_dt+1)
    R_sigma = np.zeros(num_dt+1)

    # prepara gráfico se necessário
    if show:    
        # inicializa figura e define eixo vertical
        plt.figure(figsize=(12,6))
        plt.ylim(0, num_pop)
        plt.xlim(tempos[0], tempos[-1])
    
    if show == 'nuvem':
        # alpha para a nuvem de gráficos das diversas simulaçõe
        alpha_nuvem = min(0.2, 5/num_sim)  

    # simulações
    for j in range(num_sim):

        # inicializa população de cada simulação
        pop_estado = np.copy(pop_estado_0)
        S = np.array([num_pop - num_inf_0])
        I = np.array([num_inf_0])
        R = np.array([0])
     
        
        # evolui o dia e armazena as novas contagens
        for j in range(1,num_dt+1):

            pop_estado = passo_vetorial(pop_estado, redes, redes_tx_transmissao,
                                        pop_fator_tx_transmissao_c, 
                                        prob_nao_recuperacao,
                                        pop_posicoes, f_kernel, dt)

            S = np.hstack([S, np.count_nonzero(pop_estado==1)])
            I = np.hstack([I, np.count_nonzero(pop_estado==2)])
            R = np.hstack([R, np.count_nonzero(pop_estado==3)])

        # atualiza a média
        S_medio = S_medio + (S - S_medio) / j
        I_medio = I_medio + (I - I_medio) / j
        R_medio = R_medio + (R - R_medio) / j

        # atualiza a média dos quadrados, para o cálculo do desvio padrão
        S_sigma = S_sigma + (S**2 - S_sigma) / j
        I_sigma = I_sigma + (I**2 - I_sigma) / j
        R_sigma = R_sigma + (R**2 - R_sigma) / j

        if show == 'nuvem':
            # exibe os gráficos dos dados de cada simulação
            plt.plot(tempos, S, '-', color='tab:green', alpha=alpha_nuvem)
            plt.plot(tempos, I, color='tab:red', alpha=alpha_nuvem)
            plt.plot(tempos, R, '-', color='tab:blue', alpha=alpha_nuvem)
            plt.plot(tempos, num_pop - S, '-', color='tab:gray', alpha=alpha_nuvem)

    # calcula desvio padrão
    S_sigma = ( S_sigma - S_medio**2 )**.5
    I_sigma = ( I_sigma - I_medio**2 )**.5
    R_sigma = ( R_sigma - R_medio**2 )**.5

    # exibe gráficos
    if show == 'sd':
        plt.fill_between(tempos, S_medio - S_sigma, S_medio + S_sigma, facecolor='tab:green', alpha = 0.2)
        plt.fill_between(tempos, I_medio - I_sigma, I_medio + I_sigma, facecolor='tab:red', alpha = 0.2)
        plt.fill_between(tempos, R_medio - R_sigma, R_medio + R_sigma, facecolor='tab:blue', alpha = 0.2)
        plt.fill_between(tempos, num_pop - S_medio - S_sigma, num_pop - S_medio + S_sigma, facecolor='tab:gray', alpha = 0.2)

    if show:
        plt.plot(tempos, S_medio, '-', color='tab:green', label='suscetíveis')
        plt.plot(tempos, I_medio, '-', color='tab:red', label='infectados')
        plt.plot(tempos, R_medio, '-', color='tab:blue', label='recuperados')
        plt.plot(tempos, num_pop - S_medio, '-', color='tab:gray', label='inf.+ rec.')
        
    if show == 'nuvem':
        plt.title('Evolução do conjunto de simulações e da média', fontsize=16)
    elif show == 'sd':
        plt.title('Evolução da média, com o desvio padrão', fontsize=16)
    elif show == 'media':
        plt.title('Evolução da média das simulações', fontsize=16)
        
    # informações para o gráfico
    if show:
        plt.xlabel('tempo', fontsize=14)
        plt.ylabel('número de indivíduos', fontsize=14)
        plt.legend(fontsize=12)
        plt.show() 

    resultado = SIR_Individual(
        tempos,
        S_medio,
        I_medio, 
        R_medio,
        I_sigma,
        R_sigma,
        S_sigma
    )

    return resultado

@njit
def get_estado_jit(pop_estado, estado):
    return np.array([1 if e == estado else 0 for e in pop_estado])

@njit
def get_contatos_de_risco_jit(num_pop, conexoes,
                              pop_suscetiveis, pop_infectados):
    contatos_de_risco = np.zeros(num_pop)
    for i, k in conexoes:
            if pop_infectados[k] and pop_suscetiveis[i]:
                contatos_de_risco[i] += 1
            elif pop_infectados[i] and pop_suscetiveis[k]:
                contatos_de_risco[k] += 1
    return contatos_de_risco

@njit
def dist2_jit(x,y):
    return (abs(x[0] - y[0])**2 + abs(x[1]-y[1])**2)**.5

@njit
def f_kernel_jit(d):
    return 1.0/(1.0 + (d/1.0)**1.5)

@njit
def get_contatos_de_risco_c_jit_old(num_pop, pop_infectados, 
                                pop_suscetiveis, pop_posicoes):

    ret = []
    for i in range(num_pop):
        produto = 0
        if pop_suscetiveis[i] == 1:
            for j in range(num_pop):
                if pop_infectados[j] == 1:
                    produto += f_kernel_jit(dist2_jit(pop_posicoes[j], pop_posicoes[i])) 

        ret.append(produto)
    return np.array(ret)

@njit
def get_contatos_de_risco_c_jit(num_pop, pop_infectados, 
                                pop_suscetiveis, pop_posicoes):

    ret = np.zeros(num_pop)
    for j in range(num_pop):
        produto = 0
        if pop_infectados[j] == 1:
            for i in range(num_pop):
                if pop_suscetiveis[i] == 1:
                    ret[i] += f_kernel_jit(dist2_jit(pop_posicoes[j], pop_posicoes[i])) 

    return ret

@njit
def get_novos_infectados_jit(num_pop, sorteio, prob_nao_contagio):
    return np.array([1 if sorteio[i]> prob_nao_contagio[i] else 0 for i in range(num_pop)])

@njit
def get_novos_recuperados_jit(num_pop, sorteio, pop_infectados,
                              prob_nao_recuperacao):
    return np.array([1 if pop_infectados[i]*sorteio[i]> prob_nao_recuperacao else 0 for i in range(num_pop)])

@njit
def passo_vetorial_jit(pop_estado, conexoes, tx_transmissao,
                       pop_fator_tx_transmissao_c, prob_nao_recuperacao,
                       pop_posicoes, dt):
    num_pop = len(pop_estado)

    pop_suscetiveis = get_estado_jit(pop_estado, 1)

    pop_infectados = get_estado_jit(pop_estado, 2)
    
    aux = len(tx_transmissao)
    
    contatos_de_risco_rs = np.zeros((len(tx_transmissao), num_pop))
    
    for j in range(len(conexoes)):
        contatos_de_risco_rs[j] = \
            get_contatos_de_risco_jit(num_pop, conexoes[j],
                                      pop_suscetiveis, pop_infectados)
        
    contatos_de_risco_c \
        = get_contatos_de_risco_c_jit(num_pop, pop_infectados, 
                                      pop_suscetiveis, pop_posicoes)

    lambda_rate = np.random.rand(num_pop)
    
    lambda_rate = tx_transmissao[0] * contatos_de_risco_rs[0]
    for j in range(1, len(tx_transmissao)):
        lambda_rate += tx_transmissao[j] * contatos_de_risco_rs[j]
    lambda_rate += pop_fator_tx_transmissao_c * contatos_de_risco_c

    prob_nao_contagio = np.exp(-dt*lambda_rate)
    
    sorteio = np.random.rand(num_pop)
    
    pop_novos_infectados = np.select([sorteio > prob_nao_contagio], [np.ones(num_pop)])
    
    pop_novos_recuperados = np.select([pop_infectados * sorteio > prob_nao_recuperacao], 
                                      [np.ones(num_pop)])
    
    return pop_estado + pop_novos_infectados + pop_novos_recuperados

@njit
def hstack_jit(vetor, valor):
    return np.append(vetor, valor)

@njit
def simulacao(pop_estado, conexoes, tx_transmissao,
              pop_fator_tx_transmissao_c,
              prob_nao_recuperacao,
              pop_posicoes, dt, num_dt, S, I, R):
    for j in range(1,num_dt+1):

        pop_estado = \
            passo_vetorial_jit(pop_estado, conexoes, tx_transmissao,
                               pop_fator_tx_transmissao_c,
                               prob_nao_recuperacao,
                               pop_posicoes, dt)

        S = hstack_jit(S, np.count_nonzero(pop_estado==1))
        I = hstack_jit(I, np.count_nonzero(pop_estado==2))
        R = hstack_jit(R, np.count_nonzero(pop_estado==3))

    return S, I, R

def evolucao_vetorial_jit(pop_estado_0, pop_posicoes, redes,
                          redes_tx_transmissao, pop_fator_tx_transmissao_c,
                          gamma, f_kernel,
                          dados_temporais, num_sim, show=''):
    """Evolução temporal da epidemia em um grafo estruturado.

    Entrada:
        pop_0: numpy.array
            state of the population, with 
                1: suscetível
                2: infectado
                3: recuperado ou removido

        G: numpy.array
            grafo de conexões, com atributo `taxa de transmissao`

        kernel: function
            função 'kernel' decaindo com distância entre a posição dos indivíduos

        dados_temporais: list
            [t_0, dt, num_dt]

        num_sim: int
            número de simulações

        show: str
            indica se é para exibir um gráfico e de que tipo:
                - 'nuvem': exibe uma nuvem com todas as simulações e o valor médio em destaque
                - 'sd': exibe o valor médio com uma faixa dada pelo desvio padrão
                - 'sdc': exibe o valor médio com uma faixa dada pelo desvio padrão corrigido
                - 'medio': exibe apenas o valor médio
                - '': não exibe gráfico algum.

    Saída
        X: class.SIR_Individual
            Uma instância da classe `SIR_Individual` com os seguintes atributos:
                tempos:
                num_sim:
                S_medio:
                I_medio: 
                R_medio:
                I_sigma:
                R_sigma:
                S_sigma:
    """  

    # confere se escolha para `show` é válida    
    if show:
        assert(show in ('nuvem', 'sd', 'sdc', 'media')), 'Valor inválido para argumento `show`.'

    # atributos de saída
    SIR_Individual = namedtuple('SIR_Individual', 
                                [
                                    'tempos',
                                    'S_medio',
                                    'I_medio', 
                                    'R_medio',
                                    'S_sigma',
                                    'I_sigma', 
                                    'R_sigma'
                                ])
    
    # número de indivíduos da população
    num_pop = len(pop_estado_0)
    num_inf_0 = np.count_nonzero(pop_estado_0==2)        

    # número de instantes no tempo e passos de tempo
    t_0 = dados_temporais[0]
    dt = dados_temporais[1]
    num_dt = dados_temporais[2]
    tempos = np.linspace(t_0, num_dt*dt, num_dt+1)
    
    # ajusta dados da rede dependendo se há mais de uma ou não
    #
    if type(redes) == list or type(redes_tx_transmissao) == list:
        assert(len(redes) == len(redes_tx_transmissao)), 'redes e taxas de transmissão \
        \ devem ter o mesmo número de elementos.'
    else:
        redes = [redes]
        redes_tx_transmissao = [redes_tx_transmissao]

    conexoes = List()
    for rede in redes:
        conexoes_aux = list(rede.edges)
        typed_conexoes = List()
        for c in conexoes_aux:
            typed_conexoes.append(c)
        conexoes.append(typed_conexoes)
        
    tx_transmissao = List()
    for rede_tx in redes_tx_transmissao:
        tx_transmissao.append(rede_tx)

    # calcula propabilidade de não recuperação
    prob_nao_recuperacao = np.exp(-dt*gamma)

    # inicializa variáveis para o cálculo da média
    S_medio = np.zeros(num_dt+1)
    I_medio = np.zeros(num_dt+1)
    R_medio = np.zeros(num_dt+1)

    # inicializa variáveis para o cálculo do desvio padrão
    S_sigma = np.zeros(num_dt+1)
    I_sigma = np.zeros(num_dt+1)
    R_sigma = np.zeros(num_dt+1)

    # prepara gráfico se necessário
    if show:    
        # inicializa figura e define eixo vertical
        plt.figure(figsize=(12,6))
        plt.ylim(0, num_pop)
        plt.xlim(tempos[0], tempos[-1])
    
    if show == 'nuvem':
        # alpha para a nuvem de gráficos das diversas simulaçõe
        alpha_nuvem = min(0.2, 5/num_sim)  

    # simulações
    for j in range(1, num_sim + 1):

        # inicializa população de cada simulação
        pop_estado = np.copy(pop_estado_0)
        S = np.array([num_pop - num_inf_0])
        I = np.array([num_inf_0])
        R = np.array([0])
     
        
        # faz uma simulação e armazena as novas contagens
        S, I, R = simulacao(pop_estado, conexoes, tx_transmissao,
                            pop_fator_tx_transmissao_c,
                            prob_nao_recuperacao,
                            pop_posicoes, dt, num_dt, S, I, R)

        # atualiza a média
        S_medio = S_medio + (S - S_medio) / j
        I_medio = I_medio + (I - I_medio) / j
        R_medio = R_medio + (R - R_medio) / j

        # atualiza a média dos quadrados, para o cálculo do desvio padrão
        S_sigma = S_sigma + (S**2 - S_sigma) / j
        I_sigma = I_sigma + (I**2 - I_sigma) / j
        R_sigma = R_sigma + (R**2 - R_sigma) / j

        if show == 'nuvem':
            # exibe os gráficos dos dados de cada simulação
            plt.plot(tempos, S, '-', color='tab:green', alpha=alpha_nuvem)
            plt.plot(tempos, I, color='tab:red', alpha=alpha_nuvem)
            plt.plot(tempos, R, '-', color='tab:blue', alpha=alpha_nuvem)
            plt.plot(tempos, num_pop - S, '-', color='tab:gray', alpha=alpha_nuvem)

    # calcula desvio padrão
    S_sigma = ( S_sigma - S_medio**2 )**.5
    I_sigma = ( I_sigma - I_medio**2 )**.5
    R_sigma = ( R_sigma - R_medio**2 )**.5

    # exibe gráficos
    if show == 'sd':
        plt.fill_between(tempos, S_medio - S_sigma, S_medio + S_sigma, facecolor='tab:green', alpha = 0.2)
        plt.fill_between(tempos, I_medio - I_sigma, I_medio + I_sigma, facecolor='tab:red', alpha = 0.2)
        plt.fill_between(tempos, R_medio - R_sigma, R_medio + R_sigma, facecolor='tab:blue', alpha = 0.2)
        plt.fill_between(tempos, num_pop - S_medio - S_sigma, num_pop - S_medio + S_sigma, facecolor='tab:gray', alpha = 0.2)

    if show:
        plt.plot(tempos, S_medio, '-', color='tab:green', label='suscetíveis')
        plt.plot(tempos, I_medio, '-', color='tab:red', label='infectados')
        plt.plot(tempos, R_medio, '-', color='tab:blue', label='recuperados')
        plt.plot(tempos, num_pop - S_medio, '-', color='tab:gray', label='inf.+ rec.')
        
    if show == 'nuvem':
        plt.title('Evolução do conjunto de simulações e da média', fontsize=16)
    elif show == 'sd':
        plt.title('Evolução da média, com o desvio padrão', fontsize=16)
    elif show == 'media':
        plt.title('Evolução da média das simulações', fontsize=16)

        
    # informações para o gráfico
    if show:
        plt.xlabel('tempo', fontsize=14)
        plt.ylabel('número de indivíduos', fontsize=14)
        plt.legend(fontsize=12)
        plt.show() 

    resultado = SIR_Individual(
        tempos,
        S_medio,
        I_medio, 
        R_medio,
        I_sigma,
        R_sigma,
        S_sigma
    )

    return resultado

def passo_matricial(num_pop, populacao, T_prob_nao_infeccao,
                    prob_nao_recuperacao):
        
        # gera uma matriz cheia aleatória (números em [0.0,1.0))
        A_random = np.random.rand(num_pop, num_pop)
        
        # separa os suscetíveis, criando um vetor de 1's e 0's, se for, ou não, suscetível
        pop_suscetiveis = np.select([populacao==1], [populacao])
        
        # separa os infectados, criando um vetor de 1's e 0's, se for, ou não, infectado/contagioso
        pop_infectados = np.select([populacao==2], [populacao])/2
        
        # cria uma matriz aleatória de risco, mantendo apenas as conexões que envolvem infectados
        A_risco_random = np.multiply(np.tile(pop_infectados, (num_pop, 1)), A_random)
        
        # filtra a matriz aleatória mantendo apenas os contatos entre um suscetível e um infectado
        A_contatos = np.multiply(np.tile(pop_suscetiveis, (num_pop, 1)).transpose(), A_risco_random)

        # cria uma matriz de 1's e 0's, indicando se houve, ou não, contágio
        A_infectados = np.select([A_contatos > T_prob_nao_infeccao], [np.ones([num_pop, num_pop])])
        
        # obtém novos infectados
        pop_novos_infectados = np.select([np.sum(A_infectados, axis=1)>0], [np.ones(num_pop)])
        
        if bernoulli:
            # filtra matriz aleatória com a diagonal
            pop_recuperando = pop_infectados @ np.multiply(np.eye(num_pop), A_random)
            
            # obtém novos recuperados
            pop_novos_recuperados = np.select([pop_recuperando > prob_nao_recuperacao], [np.ones(num_pop)])
            
            # atualiza população adicionando um aos que avançaram de estágio
            populacao_nova = populacao + pop_novos_infectados + pop_novos_recuperados
        
        else:
            populacao_nova = populacao + pop_novos_infectados
                    
        # Observe que cada elemento da matriz aleatória é usado apenas uma vez, garantindo
        # a independência desses eventos aleatórios (tanto quanto se leve em consideração
        # que os números gerados são pseudo-aleatórios)

        return populacao_nova

def evolucao_matricial(pop_0, G, gamma, kappa, infectados_contador_0, tempos, num_sim, show=''):
    """Evolução temporal da epidemia em um grafo estruturado.


    Entrada:
        pop_0: numpy.array
            state of the population, with 
                1: suscetível
                2: infectado
                3: recuperado ou removido

        G: numpy.array
            grafo de conexões, com atributo `taxa de transmissao`

        gamma: float
            taxa de recuperação por unidade de tempo

        tempos: numpy.array
            instantes de tempo

        num_sim: int
            número de simulações

        show: str
            indica se é para exibir um gráfico e de que tipo:
                - 'nuvem': exibe uma nuvem com todas as simulações e o valor médio em destaque
                - 'sd': exibe o valor médio com uma faixa dada pelo desvio padrão
                - 'sdc': exibe o valor médio com uma faixa dada pelo desvio padrão corrigido
                - 'medio': exibe apenas o valor médio
                - '': não exibe gráfico algum.

    Saída
        X: class.SIR_Individual
            Uma instância da classe `SIR_Individual` com os seguintes atributos:
                pop_0:
                num_pop:
                tau:
                gamma:
                tempos:
                num_sim:
                S_medio:
                I_medio: 
                R_medio:
                I_sigma:
                R_sigma:
                S_sigma:
    """

    

    
    # confere se escolha para `show` é válida
    if show:
        assert(show in ('nuvem', 'sd','media')), 'Valor inválido para argumento `show`.'

    # atributos de saída
    SIR_Individual = namedtuple('SIR_Individual', 
                                [
                                    'pop_0',
                                    'num_pop',
                                    'gamma',
                                    'tempos',
                                    'num_sim',
                                    'S_medio',
                                    'I_medio', 
                                    'R_medio',
                                    'S_sigma',
                                    'I_sigma', 
                                    'R_sigma'
                                ])
    
    # número de indivíduos da população
    num_pop = len(pop_0)
    I_0 = np.count_nonzero(pop_0==2)        

    # número de instantes no tempo e passos de tempo 
    num_t = len(tempos)
    passos_de_tempo = tempos[1:] - tempos[:-1]

    # inicializa variáveis para o cálculo da média
    S_medio = np.zeros(num_t)
    I_medio = np.zeros(num_t)
    R_medio = np.zeros(num_t)

    # inicializa variáveis para o cálculo do desvio padrão
    S_sigma = np.zeros(num_t)
    I_sigma = np.zeros(num_t)
    R_sigma = np.zeros(num_t)
    
    # obtém matriz de adjacências e o número médio de vizinhos do grafo
    T_adj = nx.to_numpy_matrix(G, weight = 'taxa de transmissao')

    # prepara gráfico se necessário
    if show:    
        # inicializa figura e define eixo vertical
        plt.figure(figsize=(12,6))
        plt.ylim(0, num_pop)
        plt.xlim(tempos[0], tempos[-1])
    
    if show == 'nuvem':
        # alpha para a nuvem de gráficos das diversas simulaçõe
        alpha = min(0.2, 5/num_sim)    

    # simulações
    for j in range(num_sim):

        # inicializa população de cada simulação
        populacao = np.copy(pop_0)
        S = np.array([num_pop - I_0])
        I = np.array([I_0])
        R = np.array([0])
        infectados_contador = np.copy(infectados_contador_0)
        tempo_decorrido = 0
        # evolui o dia e armazena as novas contagens
        for dt in passos_de_tempo:
            A_contador = np.tile(infectados_contador, [num_pop, 1])
            # calcula probabilidades
            T_prob_nao_infeccao = np.exp(-dt * np.multiply(T_adj, kappa(A_contador))) 
            prob_nao_recuperacao = np.exp(-gamma*dt)
            populacao = passo_matricial(num_pop, populacao,
                                                T_prob_nao_infeccao, prob_nao_recuperacao)
            # a = np.logical_and(infectados_contador > 10, kappa(infectados_contador) == 0)
            populacao[infectados_contador > 11] = 3
            S = np.hstack([S, np.count_nonzero(populacao==1)])
            I = np.hstack([I, np.count_nonzero(populacao==2)])
            R = np.hstack([R, np.count_nonzero(populacao==3)])
            infectados_contador[populacao==2] += dt
            tempo_decorrido += dt
        # adiciona as contagens dessa simulação para o cálculo final da média
        S_medio += S
        I_medio += I
        R_medio += R

        # adiciona as contagens dessa simulação para o cálculo final do desvio padrão
        S_sigma += S ** 2
        I_sigma += I ** 2
        R_sigma += R ** 2

        if show == 'nuvem':
            # exibe os gráficos dos dados de cada simulação
            plt.plot(tempos, S, '-', color='tab:green', alpha=alpha)
            plt.plot(tempos, I, color='tab:red', alpha=alpha)
            plt.plot(tempos, R, '-', color='tab:blue', alpha=alpha)
            plt.plot(tempos, num_pop - S, '-', color='tab:gray', alpha=alpha)

    # divide pelo número de evoluções para obter a média
    S_medio /= num_sim
    I_medio /= num_sim
    R_medio /= num_sim

    # ajusta o calcula do desvio padrão
    S_sigma = ( S_sigma / num_sim - S_medio**2 )**.5
    I_sigma = ( I_sigma / num_sim - I_medio**2 )**.5
    R_sigma = ( R_sigma / num_sim - R_medio**2 )**.5
    
    S_sigma_cor = ( num_sim * S_sigma**2 / (num_sim - 1) )**.5
    I_sigma_cor = ( num_sim * I_sigma**2 / (num_sim - 1) )**.5
    R_sigma_cor = ( num_sim * R_sigma**2 / (num_sim - 1) )**.5

    # exibe gráficos
    if show == 'sd':
        plt.fill_between(tempos, S_medio - S_sigma, S_medio + S_sigma, facecolor='tab:green', alpha = 0.2)
        plt.fill_between(tempos, I_medio - I_sigma, I_medio + I_sigma, facecolor='tab:red', alpha = 0.2)
        plt.fill_between(tempos, R_medio - R_sigma, R_medio + R_sigma, facecolor='tab:blue', alpha = 0.2)
        plt.fill_between(tempos, num_pop - S_medio - S_sigma, num_pop - S_medio + S_sigma, facecolor='tab:gray', alpha = 0.2)

    if show:
        plt.plot(tempos, S_medio, '-', color='tab:green', label='suscetíveis')
        plt.plot(tempos, I_medio, '-', color='tab:red', label='infectados')
        plt.plot(tempos, R_medio, '-', color='tab:blue', label='recuperados')
        plt.plot(tempos, num_pop - S_medio, '-', color='tab:gray', label='inf.+ rec.')
        
    if show == 'nuvem':
        plt.title('Evolução do conjunto de simulações e da média', fontsize=16)
    elif show == 'sd':
        plt.title('Evolução da média, com o desvio padrão', fontsize=16)
    elif show == 'media':
        plt.title('Evolução da média das simulações', fontsize=16)

        
    # informações para o gráfico
    if show:
        plt.xlabel('tempo', fontsize=14)
        plt.ylabel('número de indivíduos', fontsize=14)
        plt.legend(fontsize=12)
        plt.show() 

    resultado = SIR_Individual(
        pop_0,
        num_pop,
        gamma,
        tempos,
        num_sim,
        S_medio,
        I_medio, 
        R_medio,
        I_sigma,
        R_sigma,
        S_sigma
    )

    return resultado


def evolucao_grafo_estruturado(pop_0, p, lamb, mu, theta, sigma, rho, gammaA, epsA, gammaI, epsI, dI, gammaD, dD, G, tempos, num_sim, show='', toprint = False):
    """Evolução temporal da epidemia em um grafo estruturado.


    Entrada:
        pop_0: numpy.array
            state of the population, with 
                1: suscetível
                2: exposto
                3: infectado
                4: recuperado
                5: decesso
                6: diagnosticado
                7: quarentenado
                8: infectado assintomático


        G: numpy.array
            grafo de conexões

        tempos: numpy.array
            instantes de tempo

        num_sim: int
            número de simulações

        show: str
            indica se é para exibir um gráfico e de que tipo:
                - 'nuvem': exibe uma nuvem com todas as simulações e o valor médio em destaque
                - 'sd': exibe o valor médio com um intervalo de confiança dado pelo desvio padrão
                - 'medio': exibe apenas o valor médio
                - '': não exibe gráfico algum.

    Saída
        X: class.SEIRMQDA_Individual
            Uma instância da classe `SEIRMQDA_Individual` com os seguintes atributos:
                pop_0:
                num_pop:
                .
                parâmetros:
                .
                G:
                tempos:
                num_sim:
                S_mean:
                E_mean: 
                I_mean: 
                R_mean:
                M_mean:
                D_mean:
                Q_mean:
                A_mean:
                S_sigma:
                E_sigma: 
                I_sigma: 
                R_sigma:
                M_sigma:
                D_sigma:
                Q_sigma:
                A_sigma:
    """
    def passo_grafo_estruturado(num_pop, populacao, A_adj, prob_entrar_quar, prob_sair_quar,prob_infeccao, prob_infeccao_assint, prob_incubacao, proporcao_inf, prob_recuperacao_assint, prob_diagnosticar_assint, prob_recuperacao_inf, prob_diagnosticar_inf, prob_morte_inf, prob_recuperacao_diagnost, prob_morte_diagnost):
        
        
        # gera uma matriz cheia aleatória (números em [0.0,1.0)) (para usarmos nos infectados sintomaticos)
        A_random = np.random.rand(num_pop, num_pop)*2
        
        # filtra de acordo com as conexões
        A_graph_random = np.multiply(A_adj, A_random)
        
        # copia a população (acho que é desnecessário)
        pop_aux = np.copy(populacao)
        
        # separa os suscetíveis, criando um vetor de 1's e 0's, se for, ou não, suscetível
        pop_suscetiveis = np.select([pop_aux==1], [pop_aux])
        
        # separa os infectados, criando um vetor de 1's e 0's, se for, ou não, exposto
        pop_expostos = np.select([pop_aux==2], [pop_aux])/2
        
        # separa os infectados, criando um vetor de 1's e 0's, se for, ou não, infectado
        pop_infectados = np.select([pop_aux==3], [pop_aux])/3
        
        # separa os assintomaticos, criando um vetor de 1's e 0's, se for, ou não, assintomatico
        pop_assintomaticos = np.select([pop_aux==8], [pop_aux])/8
        
        # separa os quarentenados, criando um vetor de 1's e 0's, se for, ou não, quarentenados
        pop_quarentenados = np.select([pop_aux==6], [pop_aux])/6
        
        # separa os diagnosticados, criando um vetor de 1's e 0's, se for, ou não, diagnosticados
        pop_diagnosticados = np.select([pop_aux==7], [pop_aux])/7
        
        # cria uma matriz de risco, mantendo apenas as conexões que envolvem um infectado
        A_risco = np.multiply(np.tile(pop_infectados, (num_pop, 1)), A_graph_random)
        
        # cria uma matriz de risco, mantendo apenas as conexões que envolvem um assintomatico
        B_risco = np.multiply(np.tile(pop_assintomaticos, (num_pop, 1)), A_graph_random)
        
        # cria matriz de contatos entre um suscetível e um infectado
        A_contatos = np.multiply(np.tile(pop_suscetiveis, (num_pop, 1)).transpose(), A_risco)
        
        # cria matriz de contatos entre um suscetível e um assintomatico
        B_contatos = np.multiply(np.tile(pop_suscetiveis, (num_pop, 1)).transpose(), B_risco)
        
        # cria uma matriz de 1's e 0's, indicando se houve, ou não, contágio
        A_expostos = np.select([np.logical_or(np.logical_and(A_contatos<1,A_contatos > 1-prob_infeccao), np.logical_and(B_contatos <1,
        B_contatos > 1-prob_infeccao_assint))], 
                               [np.ones([num_pop, num_pop])])

        # obtém novos expostos
        pop_novos_expostos = np.select([np.sum(A_expostos, axis=1)>0], [np.ones(num_pop)])
         
        # filtra matriz aleatória com a diagonal
        pop_quarentenando = pop_suscetiveis @ np.multiply(np.eye(num_pop), A_random)
        pop_saindo_quar = pop_quarentenados @ np.multiply(np.eye(num_pop), A_random)/2
        pop_incubando = pop_expostos @ np.multiply(np.eye(num_pop), A_random)/2
        # pop_inf_mudanca = pop_infectados * np.random.rand(num_pop)/2
        # pop_assint_mudanca = pop_assintomaticos * np.random.rand(num_pop)/2
        pop_inf_mudanca = pop_infectados @ np.multiply(np.eye(num_pop), A_random)/2
        pop_assint_mudanca = pop_assintomaticos @ np.multiply(np.eye(num_pop), A_random)/2
        # pop_inf_mudanca = pop_infectados * np.random.rand(num_pop)/2
        # pop_diagnost_mudanca = pop_diagnosticados * np.random.rand(num_pop)/2
        pop_diagnost_mudanca = pop_diagnosticados @ np.multiply(np.eye(num_pop), A_random)/2
        
        # obtém novos quarentenados
        pop_novos_quarentenados = np.select([pop_quarentenando > 2-prob_entrar_quar], 
                                            [np.ones(num_pop)])
        

        # obtém novos suscetiveis
        pop_novos_suscetiveis = np.select([pop_saindo_quar > 1-prob_sair_quar], ##np.logical_and(pop_saindo_quar > 1-prob_sair_quar, pop_novos_expostos == 0)
                                            [np.ones(num_pop)])
        
        # obtém vetor de valores pós incubação
        pop_fim_incubacao = np.select([pop_incubando > 1-prob_incubacao], [np.ones(num_pop)]) * np.random.rand(num_pop)
        
        #obtém novos assintomáticos
        pop_novos_assintomaticos = np.select([pop_fim_incubacao > proporcao_inf], [np.ones(num_pop)])
        
        #obtém novos infectados
        pop_novos_infectados = np.select([np.logical_and(pop_fim_incubacao <=  proporcao_inf, pop_fim_incubacao > 0)], 
                                                             [np.ones(num_pop)])
    
        
        # obtém novos recuperados
        pop_novos_recuperados = np.select([np.logical_or.reduce([
            np.logical_and(3*pop_inf_mudanca > 1-  prob_recuperacao_inf,3*pop_inf_mudanca < 1),
            np.logical_and(2*pop_diagnost_mudanca > 1-  prob_recuperacao_diagnost,2*pop_diagnost_mudanca < 1),  
            np.logical_and(2*pop_assint_mudanca > 1-  prob_recuperacao_assint,2*pop_assint_mudanca < 1)])],
            [np.ones(num_pop)])

        # pop_novos_recuperados = np.select([np.logical_or.reduce([pop_inf_mudanca > 1-  prob_recuperacao_inf, 
        #                                                  pop_diagnost_mudanca > 1 - prob_recuperacao_diagnost,
        #                                                 pop_assint_mudanca > 1 - prob_recuperacao_assint])], 
        #                                                 [np.ones(num_pop)])
        
        # obtém novos mortos
        pop_novos_mortos = np.select([np.logical_or.reduce([
            np.logical_and(3*pop_inf_mudanca > 2-  prob_morte_inf,3*pop_inf_mudanca < 2),
            np.logical_and(2*pop_diagnost_mudanca > 2-  prob_morte_diagnost,2*pop_diagnost_mudanca < 2)])],
            [np.ones(num_pop)])

        # pop_novos_mortos = np.select([np.logical_or(np.logical_and(pop_inf_mudanca > 1-prob_morte_inf, 
        #                                                            pop_novos_recuperados == 0), 
        #                                             np.logical_and(pop_diagnost_mudanca > 1 - prob_morte_diagnost, 
        #                                                            pop_novos_recuperados == 0))], [np.ones(num_pop)])
        
        # obtém novos diagnosticados
        pop_novos_diagnosticados = np.select([np.logical_or.reduce([
            np.logical_and(3*pop_inf_mudanca > 3-  prob_diagnosticar_inf,3*pop_inf_mudanca < 3),
            np.logical_and(2*pop_assint_mudanca > 2-  prob_diagnosticar_assint,2*pop_assint_mudanca < 2)])],
            [np.ones(num_pop)])

        # pop_novos_diagnosticados = np.select([np.logical_or(np.logical_and.reduce([pop_inf_mudanca > 1-prob_diagnosticar_inf, 
        #                                                       pop_novos_recuperados == 0, pop_novos_mortos == 0]),
        #                                                       np.logical_and(pop_assint_mudanca > 1-prob_diagnosticar_assint, 
        #                                                       pop_novos_recuperados == 0))],
        #                                                       [np.ones(num_pop)])
        
        #obtém suscetives
        suscetiveis = np.select([np.logical_or(np.logical_and.reduce([pop_suscetiveis == 1, pop_novos_expostos == 0, 
                                                       pop_novos_quarentenados == 0]), pop_novos_suscetiveis == 1)], [np.ones(num_pop)])

        #obtém expostos
        expostos = np.select([np.logical_or(np.logical_and.reduce([pop_expostos == 1, 
                                                                    pop_novos_infectados == 0, pop_novos_assintomaticos == 0]), ##, pop_novos_mortos == 0 
                                                                    pop_novos_expostos == 1)], [np.ones(num_pop)])
 
        
        #obtém infectados
        infectados = np.select([np.logical_or(np.logical_and.reduce([pop_infectados == 1, 
                                                                    pop_novos_recuperados == 0, pop_novos_mortos == 0, 
                                                                    pop_novos_diagnosticados == 0]), 
                                                                    pop_novos_infectados == 1)], [np.ones(num_pop)])
        #obtém assintomaticos
        assintomaticos = np.select([np.logical_or(np.logical_and.reduce([pop_assintomaticos == 1, 
                                                                    pop_novos_recuperados == 0, pop_novos_diagnosticados == 0]), 
                                                                    pop_novos_assintomaticos == 1)], [np.ones(num_pop)])
        
        
        
        #obtém recuperados
        pop_recuperados = np.select([pop_aux == 4], [pop_aux])/4
        recuperados = np.select([np.logical_or(pop_recuperados == 1, pop_novos_recuperados == 1)], [np.ones(num_pop)])
        
        #obtém recuperados
        pop_mortos = np.select([pop_aux == 5], [pop_aux])/5
        mortos = np.select([np.logical_or(pop_mortos == 1, pop_novos_mortos == 1)], [np.ones(num_pop)])
        
        #obtém quarentenados
        quarentenados = np.select([np.logical_or(np.logical_and.reduce([pop_quarentenados == 1, 
                                                       pop_novos_suscetiveis == 0]), pop_novos_quarentenados == 1)], 
                                  [np.ones(num_pop)])
        
        #obtém diagnosticados
        diagnosticados = np.select([np.logical_or(np.logical_and.reduce([pop_diagnosticados == 1, pop_novos_recuperados == 0,
                                                                        pop_novos_mortos == 0]), pop_novos_diagnosticados == 1)], 
                                  [np.ones(num_pop)])
        
        # atualiza população adicionando um aos que avançaram de estágio
        populacao_nova = suscetiveis + 2*expostos + 3*infectados + 4*recuperados + 5*mortos + 6*quarentenados + 7*diagnosticados + 8*assintomaticos

        # Observe que cada elemento da matriz aleatória é usado apenas uma vez, garantindo
        # a independência desses eventos aleatórios (tanto quanto se leve em consideração
        # que os números gerados são pseudo-aleatórios)
        
        
        return populacao_nova

    
    # confere se escolha para `show` é válida
    if show:
        assert(show in ('nuvem', 'sd', 'media')), 'Valor inválido para argumento `show`.'

    # atributos de saída
    SEIRMQDA_Individual = namedtuple('SEIRMQDA_Individual', 
                                [
                                    'pop_0',
                                    'num_pop',
                                    'p',
                                    'lamb',
                                    'mu',
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
                                    'G',
                                    'tempos',
                                    'num_sim',
                                    'S_mean',
                                    'E_mean',
                                    'I_mean', 
                                    'R_mean',
                                    'M_mean',
                                    'Q_mean',
                                    'D_mean',
                                    'A_mean',
                                    'S_sigma',
                                    'E_sigma',
                                    'I_sigma', 
                                    'R_sigma',
                                    'M_sigma',
                                    'Q_sigma',
                                    'D_sigma',
                                    'A_sigma',
                                    'data_plot'
                                ])
    
    # número de indivíduos da população
    num_pop = len(pop_0)
    I_0 = np.count_nonzero(pop_0==3)        

    # número de instantes no tempo e passos de tempo 
    num_t = len(tempos)
    passos_de_tempo = tempos[1:] - tempos[:-1]

    # inicializa variáveis para o cálculo da média
    S_mean = np.zeros(num_t)
    E_mean = np.zeros(num_t)
    I_mean = np.zeros(num_t)
    R_mean = np.zeros(num_t)
    M_mean = np.zeros(num_t)
    Q_mean = np.zeros(num_t)
    D_mean = np.zeros(num_t)
    A_mean = np.zeros(num_t)

    # inicializa variáveis para o cálculo do desvio padrão
    S_sigma = np.zeros(num_t)
    E_sigma = np.zeros(num_t)
    I_sigma = np.zeros(num_t)
    R_sigma = np.zeros(num_t)
    M_sigma = np.zeros(num_t)
    Q_sigma = np.zeros(num_t)
    D_sigma = np.zeros(num_t)
    A_sigma = np.zeros(num_t)

    
    # obtém matriz de adjacências e o número médio de vizinhos do grafo
    A_adj = nx.to_numpy_matrix(G)

    # prepara gráfico se necessário
    if show in ('nuvem', 'sd', 'media'):    
        # inicializa figura e define eixo vertical
        fig = go.Figure()
        fig.update_layout(height=650, width=989, xaxis = dict(range = [tempos[0], tempos[-1]]), yaxis = dict(range = [0,num_pop]))
        colors = ['blue', 'purple', 'red', 'green', 'black', 'deeppink', 'brown', 'orange', 'grey']
    
    if show == 'nuvem':
        # alpha para a nuvem de gráficos das diversas simulaçõe
        alpha = min(0.2, 5/num_sim) 
        new_colors = [color_to_rgba(c, alpha) for c in colors]
        
    # simulações
    for j in range(num_sim):
        if toprint:
            print(j)

        # inicializa população de cada simulação
        populacao = np.copy(pop_0)
        S = np.array([num_pop - I_0])
        E = np.array([0])
        I = np.array([I_0])
        R = np.array([0])
        M = np.array([0])
        Q = np.array([0])
        D = np.array([0])
        A = np.array([0])
     
        
        # evolui o dia e armazena as novas contagens
        for dt in passos_de_tempo:

            # calcula probabilidades
            prob_entrar_quar =  1 - np.exp(-p*dt)
            prob_sair_quar =  1 - np.exp(-lamb*dt)
            prob_infeccao = 1 - np.exp(-mu*dt)
            prob_infeccao_assint = 1 - np.exp(-theta*dt)
            prob_incubacao = 1 - np.exp(-sigma*dt)
            proporcao_inf = rho
            prob_recuperacao_assint = 1 - np.exp(-gammaA*dt)
            prob_diagnosticar_assint = 1 - np.exp(-epsA*dt)
            prob_recuperacao_inf = 1 - np.exp(-gammaI*dt)
            prob_diagnosticar_inf = 1 - np.exp(-epsI*dt)
            prob_morte_inf = 1 - np.exp(-dI*dt)
            prob_recuperacao_diagnost = 1 - np.exp(-gammaD*dt)
            prob_morte_diagnost = 1 - np.exp(-dD*dt)

            populacao = passo_grafo_estruturado(num_pop, populacao, A_adj,prob_entrar_quar, 
                                                prob_sair_quar,prob_infeccao, prob_infeccao_assint, 
                                                prob_incubacao, proporcao_inf, prob_recuperacao_assint, 
                                                prob_diagnosticar_assint, prob_recuperacao_inf, prob_diagnosticar_inf, 
                                                prob_morte_inf, prob_recuperacao_diagnost, prob_morte_diagnost)
            
            S = np.hstack([S, np.count_nonzero(populacao==1)])
            E = np.hstack([E, np.count_nonzero(populacao==2)])
            I = np.hstack([I, np.count_nonzero(populacao==3)])
            R = np.hstack([R, np.count_nonzero(populacao==4)])
            M = np.hstack([M, np.count_nonzero(populacao==5)])
            Q = np.hstack([Q, np.count_nonzero(populacao==6)])
            D = np.hstack([D, np.count_nonzero(populacao==7)])
            A = np.hstack([A, np.count_nonzero(populacao==8)])
            
            
        # adiciona as contagens dessa simulação para o cálculo final da média
        S_mean += S
        E_mean += E
        I_mean += I
        R_mean += R
        M_mean += M
        Q_mean += Q
        D_mean += D
        A_mean += A

        # adiciona as contagens dessa simulação para o cálculo final do desvio padrão
        S_sigma += S ** 2
        E_sigma += E ** 2
        I_sigma += I ** 2
        R_sigma += R ** 2
        M_sigma += M ** 2
        Q_sigma += Q ** 2
        D_sigma += D ** 2
        A_sigma += A ** 2

        if show == 'nuvem':
            # exibe os gráficos dos dados de cada simulação
            
            fig.add_trace(go.Scatter(x = tempos, y = S, marker = dict(color = new_colors[0]), name = 'Suscetíveis (simulações)', legendgroup = 'Suscetíveis', showlegend = False))
            fig.add_trace(go.Scatter(x = tempos, y = E, marker = dict(color = new_colors[1]), name = 'Expostos (simulações)', legendgroup = 'Expostos', showlegend = False))
            fig.add_trace(go.Scatter(x = tempos, y = I, marker = dict(color = new_colors[2]), name = 'Infectados (simulações)', legendgroup = 'Infectados', showlegend = False))
            fig.add_trace(go.Scatter(x = tempos, y = R, marker = dict(color = new_colors[3]), name = 'Recuperados (simulações)', legendgroup = 'Recuperados', showlegend = False))
            fig.add_trace(go.Scatter(x = tempos, y = M, marker = dict(color = new_colors[4]), name = 'Óbitos (simulações)', legendgroup = 'Óbitos', showlegend = False))
            fig.add_trace(go.Scatter(x = tempos, y = Q, marker = dict(color = new_colors[5]), name = 'Quarentenados (simulações)', legendgroup = 'Quarentenados', showlegend = False))
            fig.add_trace(go.Scatter(x = tempos, y = D, marker = dict(color = new_colors[6]), name = 'Diagnosticados (simulações)', legendgroup = 'Diagnosticados', showlegend = False))
            fig.add_trace(go.Scatter(x = tempos, y = A, marker = dict(color = new_colors[7]), name = 'Assintomáticos (simulações)', legendgroup = 'Assintomáticos', showlegend = False))
            fig.add_trace(go.Scatter(x = tempos, y = num_pop - S - Q, marker = dict(color = new_colors[8]), name = 'Afestados pela doença (simulações)', legendgroup = 'Afetados', showlegend = False))
    # divide pelo número de evoluções para obter a média
    S_mean /= num_sim
    E_mean /= num_sim
    I_mean /= num_sim
    R_mean /= num_sim
    M_mean /= num_sim
    Q_mean /= num_sim
    D_mean /= num_sim
    A_mean /= num_sim

    # ajusta o calcula do desvio padrão
    S_sigma = ( S_sigma / num_sim - S_mean**2 )**.5
    E_sigma = ( E_sigma / num_sim - E_mean**2 )**.5
    I_sigma = ( I_sigma / num_sim - I_mean**2 )**.5
    R_sigma = ( R_sigma / num_sim - R_mean**2 )**.5
    M_sigma = ( M_sigma / num_sim - M_mean**2 )**.5
    Q_sigma = ( Q_sigma / num_sim - Q_mean**2 )**.5
    D_sigma = ( D_sigma / num_sim - D_mean**2 )**.5
    A_sigma = ( A_sigma / num_sim - A_mean**2 )**.5

    # exibe os gráficos das médias
    if show in ('nuvem', 'sd', 'media'):
        new_colors = [color_to_rgba(c, alpha = 0.2) for c in colors]
        compartimentos = [S_mean, E_mean, I_mean, R_mean, M_mean, Q_mean, D_mean, A_mean]
        sigmas = [S_sigma, E_sigma, I_sigma, R_sigma, M_sigma, Q_sigma, D_sigma, A_sigma]
        nomes = ['Suscetíveis', 'Expostos', 'Infectados', 'Recuperados', 'Óbitos', 'Quarentenados', 'Diagnosticados', 'Assintomáticos', 'Afetados pela doença']
        for i, comp in enumerate(compartimentos):
            fig.add_trace(go.Scatter(x = tempos, y = compartimentos[i], marker = dict(color = colors[i]), name = nomes[i], legendgroup = nomes[i], showlegend = True))
        fig.add_trace(go.Scatter(x = tempos, y = num_pop - S_mean - Q_mean, marker = dict(color = colors[-1]), name = nomes[-1], legendgroup = nomes[-1], showlegend = True))
 
    if show == 'sd':
        for i, comp in enumerate(compartimentos):
            fig.add_trace(go.Scatter(x = tempos, y =  compartimentos[i]- sigmas[i], marker = dict(color = colors[i]), name = nomes[i], legendgroup = nomes[i], showlegend = False, line = dict(width = 0)))
            fig.add_trace(go.Scatter(x = tempos, y =  compartimentos[i] + sigmas[i], marker = dict(color = colors[i]), name = nomes[i], legendgroup = nomes[i], showlegend = False, line = dict(width = 0), fill = 'tonexty', fillcolor = new_colors[i]))
            
        fig.add_trace(go.Scatter(x = tempos, y = num_pop - S_mean - Q_mean - S_sigma - Q_sigma , marker = dict(color = colors[-1]), name = nomes[-1], legendgroup = nomes[-1], showlegend = False, line = dict(width = 0)))
        fig.add_trace(go.Scatter(x = tempos, y = num_pop - S_mean - Q_mean + S_sigma + Q_sigma , marker = dict(color = colors[-1]), name = nomes[-1], legendgroup = nomes[-1], showlegend = False, line = dict(width = 0), fill = 'tonexty', fillcolor = new_colors[-1]))
 
    if show == 'nuvem':
        fig.update_layout(title = dict(text = 'Evolução do conjunto de simulações e da média', x = 0.5))
    elif show == 'sd':
        fig.update_layout(title = dict(text = 'Evolução da média, com o desvio padrão', x = 0.5))
    elif show == 'media':
        fig.update_layout(title = dict(text = 'Evolução da média das simulações', x = 0.5))

        
    # informações para o gráfico
    if show in ('nuvem', 'sd', 'media'):
        fig.update_xaxes(title_text = 'tempo')
        fig.update_yaxes(title_text = 'número de indivíduos')
        fig.show() 

    resultado = SEIRMQDA_Individual(
        pop_0,
        num_pop,
        p,
        lamb,
        mu,
        theta,
        sigma,
        rho,
        gammaA,
        epsA,
        gammaI,
        epsI,
        dI,
        gammaD,
        dD,
        G,
        tempos,
        num_sim,
        S_mean,
        E_mean,
        I_mean, 
        R_mean,
        M_mean,
        Q_mean,
        D_mean,
        A_mean,
        S_sigma,
        E_sigma,
        I_sigma,
        R_sigma,
        M_sigma,
        Q_sigma,
        D_sigma,
        A_sigma,
        fig.data
    )

    return resultado
