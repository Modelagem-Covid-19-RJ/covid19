#!/anaconda3/envs/py38/bin/python
# -*- coding: utf-8 -*-
"""
Módulo com cenários para simulações de modelos epidemiológicos.
"""

from collections import namedtuple

from functools import partial

import random

import numpy as np
import pandas as pd

import networkx as nx

from episiming import redes, individuais, rede_escolar

def power_decay(a, b, x):
    return 1.0/(1.0 + (x/a)**b)

def distribui_sobra_bloco(distrib_res_bloco, sobra, modo = 's'):
    '''
    Distribui a sobra de indivíduos por tamanho de residência.
    
    A sobra é distribuida seguindo uma ordem definida pelo argumento `modo`.
    
        Se `modo == 'c'` ("crescente"), a distribuição é do menor 
        para o maior tamanho de residência.

        Se `modo == 'd'` ("decrescente"), a distribuição é do maior 
        para o menor tamanho de residência.
        
        Se `modo == 's'` ("sorteado"), a distribuição é em ordem aleatória.
    '''
    
    if modo == 'c':
        k_indices = range(1, len(distrib_res_bloco)+1)
    elif modo == 'd':
        k_indices = range(len(distrib_res_bloco), 0, -1)
    else:
        k_aux = list(range(1, len(distrib_res_bloco)+1))
        np.random.shuffle(k_aux)
        k_indices = iter(k_aux)

    for k in k_indices:
        if sobra >= k:
            distrib_res_bloco[k-1] += 1
            sobra -= k
    distrib_pop_bloco = [distrib_res_bloco[k]*(k+1) 
                        for k in range(len(distrib_res_bloco))]
    if sobra > 0:
        distrib_res_bloco, distrib_pop_bloco, sobra \
            = distribui_sobra_bloco(distrib_res_bloco, sobra, modo)
    return distrib_res_bloco, distrib_pop_bloco, sobra

def distribui_residencias_bloco(num_pop_bloco, censo_residencial, modo = 's'):
    '''
    Distribui indivíduos por tamanho de residência, seguindo uma lista 
    com o censo de distribuição.
    
    A sobra é distribuida seguindo uma ordem definida pelo argumento `modo`.
    
        Se `modo == 'c'` ("crescente"), a distribuição é do menor 
        para o maior tamanho de residência.

        Se `modo == 'd'` ("decrescente"), a distribuição é do maior 
        para o menor tamanho de residência.
        
        Se `modo == 's'` ("sorteado"), a distribuição é em ordem aleatória.
    '''
    distrib_res_bloco = [int(num_pop_bloco*censo_residencial[k]/(k+1)) 
                                      for k in range(len(censo_residencial))]
    distrib_pop_bloco = [distrib_res_bloco[k]*(k+1) 
                              for k in range(len(censo_residencial))]
    total_bloco = sum(distrib_pop_bloco)
    sobra = num_pop_bloco -  total_bloco
    if sobra > 0:
        distrib_res_bloco, distrib_pop_bloco, sobra \
            = distribui_sobra_bloco(distrib_res_bloco, sobra, modo)
    return distrib_res_bloco, distrib_pop_bloco, sobra

def distribui_residencias_e_individuos(regiao, censo_residencial, modo = 's'):
    N, M = regiao.shape
    distrib_res_regiao = []
    distrib_pop_regiao = []
    for i in range(N):
        aux_res = []
        aux_pop = []
        for j in range(M):
            distrib_res_bloco, distrib_pop_bloco, sobra \
                = distribui_residencias_bloco(regiao[i, j], censo_residencial, modo)
            aux_res.append(distrib_res_bloco)
            aux_pop.append(distrib_pop_bloco)
            assert(sobra == 0), f'Não foi possível alocar toda a população do bloco ({i}, {j})'
        distrib_res_regiao += aux_res
        distrib_pop_regiao += aux_pop
    return distrib_res_regiao, distrib_pop_regiao

def aloca_residencias_bloco(distrib_res):
    '''
    Aloca as residências por tamanho e as posiciona relativamente ao bloco
    '''
    num_residencias = sum(distrib_res)
    sorteio_lista = list(range(num_residencias**2))
    np.random.shuffle(sorteio_lista) # embaralha "in place"
    sorteio = sorteio_lista[:num_residencias]
    pos_residencias = [( i // num_residencias / num_residencias \
                             + 1/2/num_residencias, 
                         i % num_residencias / num_residencias \
                             + 1/2/num_residencias )
                       for i in sorteio]
    return pos_residencias

def aloca_individuos_bloco(distrib_res, pos_residencias):
    '''
    Aloca e posiciona os indivíduos em residências
    '''
    num_residencias = sum(distrib_res)
    pos_individuos = []
    res_individuos = []
    m = 0
    n = 0
    for k in range(len(distrib_res)):
        for l in range(1, distrib_res[k]+1):
            res_individuos_l = []
            for i in range(k+1):
                if k == 0:
                    x = pos_residencias[m][0]
                    y = pos_residencias[m][1]
                else:
                    x = pos_residencias[m][0] \
                        + np.cos(i*2*np.pi/(k+1))/3/num_residencias
                    y = pos_residencias[m][1] \
                        + np.sin(i*2*np.pi/(k+1))/3/num_residencias
                pos_individuos.append((x, y))
                res_individuos_l.append(n)
                n += 1
            res_individuos.append(res_individuos_l)
            m += 1
    return pos_individuos, res_individuos

def aloca_residencias_e_individuos(regiao, censo_residencial):
    '''
    Aloca as residências e os residentes de cada residência.

    Lê a matriz `regiao` com a quantidade de indivíduos em cada "bloco"
    e aloca as residências e os indivíduos nos respectivos blocos,
    buscando ter uma distribuição de indivíduos por tamanho de residência
    próxima a da indicada pela lista `censo_residencial`.

    Entradas:
    ---------
        regiao: numpy.ndarray bidimensional
            Uma matriz cujos coeficientes indicam o número de residentes
            no bloco correspondente.

        censo_residencial: list of floats
            Uma lista onde cada elemento, digamos na posição k, é um 
            float entre 0 e 1 indicando a fração da população em residências
            com k+1 residentes.

    Saídas:
    -------
        pos_residencias: list of tuples of two floats
            Os elementos da lista são as coordenadas de cada residência.

        pos_individuos: list of tuples of two floats
            Os elementos da lista são as coordenadas de cada indivíduo.

        res_individuos: list of lists of integers
            Os elementos da lista são lista com os índices dos indivíduos
            de cada residência.

        pop_blocos_indices: list of integers
            É uma lista de tamanho igual ao número de blocos. Cada 
            elemento da lista indica o índice do primeiro indivíduo 
            do bloco correspondente.
    '''
     
    distrib_res_regiao, distrib_pop_regiao \
        = distribui_residencias_e_individuos(regiao, censo_residencial)

    pop_blocos_indices = list()
    soma = 0
    for distrib in distrib_pop_regiao:
        soma += sum(distrib)
        pop_blocos_indices.append(soma)   

    pos_residencias = []
    pos_individuos = []
    res_individuos = []
    n = 0
    N, M = regiao.shape
    for i in range(N):
        for j in range(M):
            pos_residencias_bloco = aloca_residencias_bloco(distrib_res_regiao[i*M + j])

            pos_individuos_bloco, res_individuos_bloco \
                = aloca_individuos_bloco(distrib_res_regiao[i*M + j], pos_residencias_bloco)
            
            pos_residencias_translated \
                = [(j + pos_residencias_bloco[k][0],
                    N - 1 - i +  pos_residencias_bloco[k][1])
                   for k in range(len(pos_residencias_bloco))]
            pos_individuos_translated \
                = [(j + pos_individuos_bloco[k][0],
                    N - 1 - i + pos_individuos_bloco[k][1])
                   for k in range(len(pos_individuos_bloco))]
            res_individuos_translated \
                = [[n + l for l in r] for r in res_individuos_bloco]
            
            pos_individuos += pos_individuos_translated
            pos_residencias += pos_residencias_translated
            res_individuos += res_individuos_translated
    
            n += len(pos_individuos_bloco)
    
    return pos_residencias, pos_individuos, res_individuos, pop_blocos_indices

def obtem_idades_fracoes(idades_grupos, idades_fracoes_grupos, idade_max=100):

    # interpola/extrapola pirâmide populacional
    idades_fracoes = list()
    for j in range(len(idades_grupos)-1):
        idades_fracoes += (idades_grupos[j+1] - idades_grupos[j]) \
            * [idades_fracoes_grupos[j]/(idades_grupos[j+1]-idades_grupos[j])]

    idades_fracoes += (idade_max - idades_grupos[-1]) \
            * [idades_fracoes_grupos[-1]/(idade_max-idades_grupos[-1])]

    idades_fracoes = np.array(idades_fracoes)

    return idades_fracoes

def gera_idades(num_pop, num_tam_res, res_individuos, idades_fracoes):

    # separa as residências por tamanho
    res = (num_tam_res+1)*[[]]
    for j in range(1,num_tam_res+1):
        # seleciona residências com j indivíduos (res[0]=[])
        res[j] = [r for r in res_individuos if len(r) == j]
    # separa as residências com um adulto e um menor:
    res_2b = random.sample(res[2], k=int(0.1*len(res[2])))
    # separa as residências com dois adultos:
    res_2a = [r for r in res[2] if r not in res_2b]
    # agrega as residências com três ou mais indivíduos:
    res_3mais = []
    for res_k in res[3:]:
        res_3mais += res_k

    # separa a pirâmide populacional
    idades = list(range(len(idades_fracoes)))

    distrib_idades_adultos = num_pop*idades_fracoes
    distrib_idades_adultos[:20] = 0

    distrib_idades_menores = num_pop*idades_fracoes
    distrib_idades_menores[20:] = 0

    # inicializa lista de idades
    pop_idades = np.zeros(num_pop).astype(int)

    # define a idade dos adultos morando sozinhos
    ind_idades = random.choices(idades, distrib_idades_adultos, k=len(res[1]))
    for j in range(len(res[1])):
        pop_idades[res[1][j]] = ind_idades[j]
        distrib_idades_adultos[ind_idades[j]] -= 1

    # define a idade do único adulto em residências com um adulto e um menor
    ind_idades = random.choices(idades, distrib_idades_adultos, k=len(res_2b))
    for j in range(len(res_2b)):
        pop_idades[res_2b[j][0]] = ind_idades[j]
        distrib_idades_adultos[ind_idades[j]] -= 1

    # define a idade de dois adultos nas outras residências com dois 
    # indivíduos
    len_res_2a = len(res_2a)
    ind_idades = random.choices(idades, distrib_idades_adultos,
                                k=2*len_res_2a)
    for j in range(len_res_2a):
        pop_idades[res_2a[j][0]] = ind_idades[j]
        pop_idades[res_2a[j][1]] = ind_idades[len_res_2a + j]
        distrib_idades_adultos[ind_idades[j]] -= 1
        distrib_idades_adultos[ind_idades[len_res_2a + j]] -= 1

    # define a idade de dois adultos nas residências com três ou mais 
    # indivíduos
    len_res_3mais = len(res_3mais)
    ind_idades = random.choices(idades, distrib_idades_adultos,
                                k=2*len_res_3mais)
    for j in range(len_res_3mais):
        pop_idades[res_3mais[j][0]] = ind_idades[j]
        pop_idades[res_3mais[j][1]] = ind_idades[len_res_3mais + j]
        distrib_idades_adultos[ind_idades[j]] -= 1
        distrib_idades_adultos[ind_idades[len_res_3mais + j]] -= 1

    # define a idade dos menores de idade em residências com um adulto 
    # e um menor
    len_res_2b = len(res_2b)
    ind_idades = random.choices(idades, distrib_idades_menores, k=len_res_2b)
    for j in range(len_res_2b):
        pop_idades[res_2b[j][1]] = ind_idades[j]
        distrib_idades_menores[ind_idades[j]] -= 1
    
    # calcula a distribuição restante de idades
    distrib_idades_left = np.array(
        [distrib_idades_menores[j] + distrib_idades_adultos[j] 
        for j in range(len(idades_fracoes))]
        )

    # define a idade do restante dos invidívuos em residências de três 
    # ou mais indivíduos
    for k in range(3,num_tam_res+1):
        ind_idades = random.choices(idades, distrib_idades_left,
                                    k=(k-2)*len(res[k]))
        for j in range(len(res[k])):
            for l in range(2, k):
                pop_idades[res[k][j][l]] = ind_idades[(l-2)*len(res[k]) + j]
                distrib_idades_left[ind_idades[(l-2)*len(res[k]) + j]] -= 1

    return pop_idades
def zipf3_acum(a, c, k_max, k):
    '''
    retorna a fração de indivíduos em empresas de tamanho maior que k.
    '''
    return (((1.0 + k_max/a)/(1.0 + k/a))**c - 1)/((1 + k_max/a)**c - 1.0)

def zipf3(a, c, k_max, k):
    '''
    retorna a fração de individuos em empresas de tamanho igual a k.
    '''
    return zipf3_acum(a, c, k_max, k-1) - zipf3_acum(a, c, k_max, k)

def zipf3e(a, c, k_max, k):
    '''
    retorna a fração de empresas de tamanho igual a k.
    '''
    return zipf3(a, c, k_max, k)/k

def quantifica_empresas_por_tamanho(num_pop, num_pea, tam_min, tam_max,
                                    z3_a, z3_c, verbose=False):

    emp_tam_z3 = np.arange(tam_min, 2*tam_max)
    emp_num_z3 = (num_pea*zipf3e(z3_a, z3_c, 2*tam_max, emp_tam_z3)).astype(int)
    emp_num_z3 = emp_num_z3[emp_num_z3>0]
    emp_tam_z3 = np.array(list(range(tam_min, tam_min + len(emp_num_z3))))
    emp_pop_z3 = np.array([(tam_min + k)*emp_num_z3[k] for k in range(len(emp_num_z3))])

    if not len(emp_tam_z3):
        print('Não foi possível distribuir as empresas, tente com outros parâmetros')
    elif verbose:
        print(f'Total da população: {num_pop}')
        print(f'Total da força de trabalho (PEA): {num_pea}')
        print(f'Número de tamanhos de empresas: {len(emp_num_z3)}')
        print(f'Número de empresas: {emp_num_z3.sum()}')
        print(f'Tamanhos de empresas: de {emp_tam_z3.min()} a {emp_tam_z3.max()}')
        print(f'Número de indivíduos nas empresas (PEA ocupados): {emp_pop_z3.sum()}')
        print(f'Média de indivíduos por empresa: {emp_pop_z3.sum()/emp_num_z3.sum()}')
        print('Porcentagem de indivíduos da força de trabalho nas empresas: '
              + f'{100*emp_pop_z3.sum()/num_pea:.1f}%')
        print(f'Distribuição do número de empresas por tamanho: \n{emp_num_z3}')
        print(f'Distribuição do número de indivíduos por tamanho de empresa: \n{emp_pop_z3}')
    
    return emp_tam_z3, emp_num_z3, emp_pop_z3

def aloca_empresas(pop_por_bloco, emp_num_z3, tam_min):
    pop_por_bloco_flat = pop_por_bloco.flatten()
    emp_loc = random.choices(list(range(len(pop_por_bloco_flat))),
                             pop_por_bloco_flat, k=emp_num_z3.sum())

    emp_por_bloco = np.zeros_like(pop_por_bloco)

    emp_bloco_pos = list()
    emp_tam = list()
    k_nivel = 0
    for k in range(len(emp_loc)):
        if k >= emp_num_z3[:k_nivel+1].sum():
            k_nivel += 1
        emp_tam.append(tam_min + k_nivel)
        loc = emp_loc[k]
        emp_bloco_pos.append((loc // 83, loc % 83))
        emp_por_bloco[loc // 83, loc % 83] += 1

    return emp_bloco_pos, emp_por_bloco, emp_tam

def aloca_emp_membros_blocos(pop_por_bloco, emp_por_bloco, emp_bloco_pos, 
                             tam_min, emp_num_z3, a_dist, c_dist):
    '''
    Aloca os blocos de localização para cada indivíduo de cada empresa.
    '''
    i = np.arange(0.5, 0.5 + emp_por_bloco.shape[0])
    j = np.arange(0.5, 0.5 + emp_por_bloco.shape[1])
    jj, ii = np.meshgrid(j,i)

    emp_membros_blocos = list()

    for k in range(len(emp_num_z3)):
        for j in range(emp_num_z3[k]):
            dist =  np.sqrt((jj - emp_bloco_pos[k+j][1])**2 
                            + (ii - emp_bloco_pos[k+j][0])**2)
#            k_dist = f_dist(dist)*pop_por_bloco
            k_dist = power_decay(a_dist, c_dist, dist)*pop_por_bloco
            emp_membros_blocos.append(
                random.choices(
                    list(range(emp_por_bloco.shape[0]*emp_por_bloco.shape[1])),
                    k_dist.flatten(),
                    k = tam_min + k
                )
            )

    return emp_membros_blocos

def aloca_emp_individuos(num_pop, pop_idades, pop_blocos_indices, 
                         pea_fracoes, emp_pop_z3, emp_tam,
                         emp_membros_blocos):
    '''
    Aloca os indivíduos em cada empresa.
    '''
    
    indices = np.arange(num_pop)
#    indices = np.array(range(num_pop))
    pop_pia_indices = indices[pop_idades >= 16]    
    
    # Define os pesos de cada individuo segundo a sua idade e os pesos para cada idade
    pesos = pea_fracoes[pop_idades[pop_pia_indices]]
    pesos /= pesos.sum() # probabities must add up to 1
    pop_pia_livres = np.random.choice(pop_pia_indices,
                                      size=emp_pop_z3.sum(),
                                      replace=False,
                                      p=pesos)
    
    # Escolhe aleatoriamete um indivíduo em cada bloco alocado
    emp_membros = list()

    for j in range(len(emp_tam)):
        membros_j = list()
        for l in emp_membros_blocos[j]:
            aux = pop_pia_livres[pop_pia_livres >= pop_blocos_indices[l]]
            candidatos = aux[aux < pop_blocos_indices[l+1]]
            if len(candidatos):
                individuo = random.choice(candidatos)
                membros_j.append(individuo)
                pop_pia_livres = pop_pia_livres[pop_pia_livres != individuo]
        emp_membros.append(membros_j)

    # Alguns blocos podem não ter mais indivíduos economicamente 
    # ativos disponíveis, então completamos com indivíduos 
    # de outros blocos quaisquer, 
    # portanto, sem peso segundo a distância.
    for j in range(len(emp_tam)):
        faltam = emp_tam[j] - len(emp_membros[j])
        if faltam > 0:
            membros_j = list(np.random.choice(pop_pia_livres, size=faltam,
                                              replace=False))
            emp_membros[j] += membros_j
            for individuo in membros_j:
                pop_pia_livres = pop_pia_livres[pop_pia_livres != individuo]
    
    return emp_membros

def gera_empresas(pop_por_bloco, pop_idades, pop_blocos_indices,
                  num_pea, pea_fracoes, 
                  tam_min, tam_max, z3_a, z3_c, a_dist, c_dist):
    '''
    Gera as empresas, com as suas localizações e os seus indivíduos.
    '''

    num_pop = len(pop_idades)

    emp_tam_z3, emp_num_z3, emp_pop_z3 \
        = quantifica_empresas_por_tamanho(num_pop, num_pea,
                                          tam_min, tam_max,
                                          z3_a, z3_c)

    emp_bloco_pos, emp_por_bloco, emp_tam \
        = aloca_empresas(pop_por_bloco, emp_num_z3, tam_min)

    emp_membros_blocos \
        = aloca_emp_membros_blocos(pop_por_bloco, emp_por_bloco,
                                   emp_bloco_pos, tam_min, emp_num_z3,
                                   a_dist, c_dist)

    emp_membros = aloca_emp_individuos(num_pop, pop_idades,
                                       pop_blocos_indices,
                                       pea_fracoes,
                                       emp_pop_z3, emp_tam,
                                       emp_membros_blocos)

    return emp_bloco_pos, emp_por_bloco, emp_tam, emp_membros

class Cenario:
    def __init__(self, num_pop, num_infectados_0, beta, gamma):
        self.nome = 'Base'
        self.define_parametros()
        self.cria_redes()
        self.inicializa_pop_estado()

    def define_parametros(self):
        self.num_pop = 1
        self.beta = 1
        self.gamma = 1
        self.attr_pos = {0: [0.0, 0.0]}
        self.pop_posicoes = np.array([[0.0, 0.0]])
        self.f_kernel = partial(power_decay, 1.0, 1.0)

    def cria_redes(self):
        """
        Gera uma rede regular completa.

        Deve ser sobre-escrito para cenários diferentes.
        """
        self.redes = []
        self.redes_tx_transmissao = []
        self.pop_fator_tx_transmissao_c = np.array([0])
        

    def inicializa_pop_estado(self, num_infectados_0):
        """
        Distribui aleatoriamente um certo número de infectados.
        """
        self.pop_estado_0 = np.array([1])
        self.attr_estado_0 = {0: {'estado': 1}}


    def exibe_redes(self, info=True, node_size=100, hist = True):
        for G in self.redes:
            redes.analise_rede(G, info=info, node_size=node_size,
                               pos=self.attr_pos, hist=hist)

    def evolui(self, dados_temporais, num_sim, show=''):
        X = individuais.evolucao_vetorial(
                self.pop_estado_0, 
                self.pop_posicoes, 
                self.redes, 
                self.redes_tx_transmissao,
                self.pop_fator_tx_transmissao_c,
                self.gamma,
                self.f_kernel,
                dados_temporais,
                num_sim,
                show
            )
        return X

    def evolui_jit(self, dados_temporais, num_sim, show=''):
        X = individuais.evolucao_vetorial_jit(
                self.pop_estado_0, 
                self.pop_posicoes, 
                self.redes, 
                self.redes_tx_transmissao,
                self.pop_fator_tx_transmissao_c,
                self.gamma,
                self.f_kernel,
                dados_temporais,
                num_sim,
                show
            )
        return X

    def evolui_matricial(self, dados_temporais, num_sim, show=''):
        tempos = np.linspace(
            dados_temporais[0],
            dados_temporais[1]*dados_temporais[2],
            dados_temporais[2] + 1
        )

        G_preps = list()
        for r in range(len(self.redes)):
            G_preps.append(self.redes[r].copy())
            attr_transmissao_edge = {
                (i, j): {'taxa de transmissao': self.redes_tx_transmissao[r][i]}
                       for (i,j) in self.redes[r].edges()
            }
            nx.set_edge_attributes(G_preps[-1], attr_transmissao_edge)

        G_c = nx.random_geometric_graph(self.num_pop, 0, pos=self.attr_pos)
        distancia = lambda x, y: sum(abs(a - b)**2 for a, b in zip(x, y))**0.5
        attr_kernel_dist = [(i, j, 
                     {'weight': self.f_kernel(
                         distancia(self.attr_pos[i], self.attr_pos[j]))}) 
                    for i in range(self.num_pop)
                    for j in range(self.num_pop) if j != i]

        G_c.add_edges_from(attr_kernel_dist)

        attr_transmissao = {(i, j): 
                            {'taxa de transmissao': self.beta_c * G_c.edges[i,j]['weight']/ G_c.degree(i, weight='weight')}
                            for (i,j) in G_c.edges()
                        }

        nx.set_edge_attributes(G_c, attr_transmissao)

        G_preps.append(G_c)
        G = nx.compose_all(G_preps)

        for (u, v, w) in G_c.edges.data('taxa de transmissao', default=0):
            G.edges[u,v]['taxa de transmissao'] = \
                G_c.edges[u,v]['taxa de transmissao']
            for G_aux in G_preps:
                if (u, v) in G_aux.edges:
                    G.edges[u, v]['taxa de transmissao'] += \
                        G_aux.edges[u,v]['taxa de transmissao']

        X = individuais.evolucao_matricial(
                self.pop_estado_0,
                G,
                self.gamma, 
                tempos,
                num_sim,
                show)
        return X

class RedeCompleta(Cenario):
    def __init__(self, num_pop, num_infectados_0, beta, gamma):
        self.nome = 'rede completa'
        self.define_parametros(num_pop, beta, gamma)
        self.inicializa_pop_estado(num_infectados_0)
        self.cria_redes()

    def define_parametros(self, num_pop, beta, gamma):
        self.num_pop = num_pop
        self.beta = beta

        self.beta_r = 0.0
        self.beta_s = beta
        self.beta_c = 0.0

        self.gamma = gamma

        self.pop_rho = np.ones(num_pop)

        self.f_kernel = partial(power_decay, 1.0, 1.5)

        self.attr_pos = dict()
        k = 0
        for i in range(self.num_pop):
            self.attr_pos.update(
                {k: [np.random.rand(), np.random.rand()]})
            k += 1
        self.pop_posicoes = np.array(list(self.attr_pos.values()))

    def inicializa_pop_estado(self, num_infectados_0):
        """
        Distribui aleatoriamente um certo número de infectados.
        """
        self.num_infectados_0 = num_infectados_0
        np.random.seed(seed = 342)
        #self.pop_estado_0 = np.ones(num_pop, dtype=np.uint8)
        self.pop_estado_0 = np.ones(self.num_pop)
        infectados_0 = np.random.choice(self.num_pop,
                                        self.num_infectados_0, 
                                        replace=False)
        #self.pop_estado_0[infectados_0] = \
        #    2*np.ones(num_infectados_0, dtype=np.uint8)
        self.pop_estado_0[infectados_0] = 2*np.ones(self.num_infectados_0)
        self.attr_estado_0 = dict([(i, {'estado': int(self.pop_estado_0[i])}) 
                                   for i in range(self.num_pop)])

    def cria_redes(self):
        """
        Gera uma rede completa.
        """
        G_reg = nx.random_regular_graph(d=self.num_pop-1,
                                        n=self.num_pop)

        nx.set_node_attributes(
            G_reg, 
            dict([(i, {'estado': int(self.pop_estado_0[i])})
                  for i in range(self.num_pop)])
            )

        nx.set_node_attributes(
            G_reg,
            dict([(i, {'rho': self.pop_rho[i]}) 
                  for i in range(self.num_pop)])
            )

        tx_transmissao_reg = np.array(
                [self.beta / (1+G_reg.degree(i)) 
                 for i in G_reg.nodes]
            )

        attr_transmissao = \
            dict([(i, {'taxa de transmissao': tx_transmissao_reg[i]}) 
                  for i in G_reg.nodes])

        nx.set_node_attributes(
            G_reg,
            dict([(i, {'taxa de transmissao': tx_transmissao_reg[i]}) 
                  for i in G_reg.nodes])
            )

        self.redes = [G_reg]
        self.redes_tx_transmissao = [tx_transmissao_reg]

        aux = np.array(
            [
                np.sum(
                    self.f_kernel(
                        np.linalg.norm(
                            self.pop_posicoes - self.pop_posicoes[i], axis=1)
                        )
                    ) 
                    for i in range(self.num_pop)
            ]
        )
        self.pop_fator_tx_transmissao_c = self.beta_c / aux
        
class Pop350(Cenario):

    def __init__(self):
        self.nome = 'Pop 350'
        self.define_parametros()
        self.inicializa_pop_estado()
        self.cria_redes()

    def define_parametros(self):

        # posições
        self.populacao_por_area = np.array(
            [
                [16, 11, 0, 0,  0,  6,  4,  8,  8,  6],
                [10, 12, 12, 6, 8, 9,  8,  6,  7,  5],
                [0, 10, 14, 10, 12,  8,  0,  0,  6,  8],
                [0, 12, 10, 14, 11,  9,  0,  0,  5,  7],
                [9, 11, 0, 12, 10,  7,  8,  7,  8, 0]
            ])

        self.num_pop = self.populacao_por_area.sum()

        self.pop_estado_0 = np.ones(self.num_pop)

        np.random.seed(seed = 127)
        self.attr_pos = dict()
        k = 0
        N, M = self.populacao_por_area.shape
        for m in range(M):
            for n in range(N):
                for i in range(self.populacao_por_area[n,m]):
                    self.attr_pos.update(
                        {k: [m + np.random.rand(), N - n + np.random.rand()]})
                    k += 1
        self.pop_posicoes = np.array(list(self.attr_pos.values()))

        # idades
        self.censo_fracoes = [0.15079769082144653, # 0 a 9 anos
                              0.17906470565542282, # 10 a 19 anos
                              0.18007108135150324, # 20 a 29 anos
                              0.15534569934620965, # 30 a 39 anos
                              0.13023309451263393, # 40 a 49 anos
                              0.09654553673621215, # 50 a 59 anos
                              0.059499784853198616, # 60 a 69 anos
                              0.033053176013799715, # 70 a 79 anos
                              0.015389230709573343] # 80 ou mais

        self.pop_idades = \
            np.random.choice(9, self.num_pop, p=self.censo_fracoes)

        self.raio_residencial = 0.6

        self.alpha_r = 0.8

        zeta_idade = lambda x: power_decay(50.0, 2.0, abs(x-35))

        self.distribuicao_social = [
            10, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
            ]
        self.rho_forma = 0.2 # shape factor of gamma distribution
        self.rho_escala = 5 # scale (mean value = scale * shape)
        self.pop_rho = np.random.gamma(self.rho_forma, self.rho_escala,
                                       self.num_pop) 

        self.a_kernel = 1.0
        self.b_kernel = 1.5
        self.f_kernel = partial(power_decay, self.a_kernel, self.b_kernel)

        self.num_infectados_0 = 8

        self.beta_r = 0.16
        self.beta_s = 0.24
        self.beta_c = 0.04
        self.gamma = 0.1        

    def inicializa_pop_estado(self):
        np.random.seed(seed = 342)
        #self.pop_estado_0 = np.ones(num_pop, dtype=np.uint8)
        self.pop_estado_0 = np.ones(self.num_pop)
        infectados_0 = np.random.choice(self.num_pop,
                                        self.num_infectados_0, 
                                        replace=False)
        #self.pop_estado_0[infectados_0] = \
        #    2*np.ones(num_infectados_0, dtype=np.uint8)
        self.pop_estado_0[infectados_0] = 2*np.ones(self.num_infectados_0)
        self.attr_estado_0 = dict([(i, {'estado': int(self.pop_estado_0[i])}) 
                                   for i in range(self.num_pop)])

    def cria_rede_residencial(self):

        self.G_r = nx.random_geometric_graph(
            self.num_pop, self.raio_residencial,
            pos=self.attr_pos, seed=1327)

        nx.set_node_attributes(
            self.G_r, 
            dict([(i, {'estado': int(self.pop_estado_0[i])})
                  for i in range(self.num_pop)])
            )

        nx.set_node_attributes(
            self.G_r,
            dict([(i, {'faixa etária': self.pop_idades[i]})
                  for i in range(self.num_pop)])
            )

        nx.set_node_attributes(
            self.G_r,
            dict([(i, {'rho': self.pop_rho[i]}) 
                  for i in range(self.num_pop)])
            )

        self.pop_tx_transmissao_r = np.array(
            [self.beta_r / (1+self.G_r.degree(i))**self.alpha_r 
            for i in self.G_r.nodes]
            )
        attr_transmissao_r = \
            dict([(i, {'taxa de transmissao': self.pop_tx_transmissao_r[i]}) 
                  for i in self.G_r.nodes])

        nx.set_node_attributes(
            self.G_r,
            dict([(i, {'taxa de transmissao': self.pop_tx_transmissao_r[i]}) 
                  for i in self.G_r.nodes])
            )

        nx.set_edge_attributes(self.G_r, 1, 'weight')

    def cria_rede_social(self):
        self.G_s = nx.random_geometric_graph(self.num_pop, 0,
                                             pos=self.attr_pos)
        nx.set_node_attributes(self.G_s, self.attr_estado_0)

        random.seed(721)
        pop_index = list(range(self.num_pop))
        membros = dict()

        for j in range(len(self.distribuicao_social)):
            individuos_aleatorios = \
                random.sample(pop_index, self.distribuicao_social[j])
            for i in individuos_aleatorios:
                pop_index.remove(i)
            membros.update({j: individuos_aleatorios})
            conexoes = [(m,n) for m in individuos_aleatorios 
                        for n in individuos_aleatorios if m != n ]
            self.G_s.add_edges_from(conexoes)

        nx.set_edge_attributes(self.G_s, 1, 'weight')

        self.pop_tx_transmissao_s = \
            np.array([self.beta_s / (1+self.G_s.degree(i)) 
                      for i in self.G_s.nodes])
        attr_transmissao_s = dict([(i, {'taxa de transmissao': self.
                                        pop_tx_transmissao_s[i]}) 
                                   for i in self.G_s.nodes])

        nx.set_node_attributes(self.G_s, attr_transmissao_s)

    def cria_redes(self):
        self.cria_rede_residencial()
        self.cria_rede_social()
        self.redes = [self.G_r, self.G_s]
        self.redes_tx_transmissao \
            = [self.pop_tx_transmissao_r, self.pop_tx_transmissao_s]

        aux = np.array(
            [
                np.sum(
                    self.f_kernel(
                        np.linalg.norm(
                            self.pop_posicoes - self.pop_posicoes[i], axis=1)
                        )
                    ) 
                    for i in range(self.num_pop)
            ]
        )
        self.pop_fator_tx_transmissao_c = self.beta_c / aux
        
class Multi350(Cenario):

    def __init__(self, x_vezes, y_vezes, pop_vezes):
        self.nome = 'Pop Multi 350'
        self.define_parametros(x_vezes, y_vezes, pop_vezes)
        self.inicializa_pop_estado()
        self.cria_redes()

    @staticmethod
    def multiplicador(h, v, lista = [], n = 0, f = None):
        if not f:
            f = lambda x: x/2
    #        f = lambda x: x**(v/h)
    #        f = lambda x: x**(0.85)
    #        f = lambda x: max(x - 1, 1)

        if n == 0:
            n = h*v
        if lista == []:
            lista = [1 for j in range(h)]
            n -= h
        if n > 0:
            v -= 1
            hp = f(min(n,h))
            hn = int(hp)
            if hp > hn:
                hn += 1
            assert(0 < hp <= hn), 'A função de decaimento deve ser não-crescente e não-nula'
            for j in range(hn):
                if n > 0:
                    lista[j] += 1
                    n -= 1
            if n > 0:
                lista = Multi350.multiplicador(h, v, lista, n, f)            
        return lista

    def define_parametros(self, x_vezes, y_vezes, pop_vezes):

        # posições
        populacao_350_lista = \
            [
                [16, 11, 0, 0,  0,  6,  4,  8,  8,  6],
                [10, 12, 12, 6, 8, 9,  8,  6,  7,  5],
                [0, 10, 14, 10, 12,  8,  0,  0,  6,  8],
                [0, 12, 10, 14, 11,  9,  0,  0,  5,  7],
                [9, 11, 0, 12, 10,  7,  8,  7,  8, 0]
            ]

        populacao_350 = np.array(populacao_350_lista)
        
        # outra opção é usar np.tile pra replicar a população
        aux = []
        for l in populacao_350_lista:
            aux.append( x_vezes * l )
            
        self.populacao_por_bloco = pop_vezes * np.array( y_vezes * aux)

        self.num_pop = self.populacao_por_bloco.sum()

        self.pop_estado_0 = np.ones(self.num_pop)

        censo_residencial \
            = np.array([.21, .26, .20, .17, .08, .04, .02, 0.02])

        self.pos_residencias, self.pos_individuos, \
            self.res_individuos, self.pop_blocos_indices \
            = aloca_residencias_e_individuos(
                self.populacao_por_bloco, censo_residencial)

        self.attr_pos \
            = {j: self.pos_individuos[j] 
               for j in range(self.num_pop)}

        self.pop_posicoes = np.array(self.pos_individuos)

        # idades
        self.censo_fracoes = [0.15079769082144653, # 0 a 9 anos
                              0.17906470565542282, # 10 a 19 anos
                              0.18007108135150324, # 20 a 29 anos
                              0.15534569934620965, # 30 a 39 anos
                              0.13023309451263393, # 40 a 49 anos
                              0.09654553673621215, # 50 a 59 anos
                              0.059499784853198616, # 60 a 69 anos
                              0.033053176013799715, # 70 a 79 anos
                              0.015389230709573343] # 80 ou mais

        self.pop_idades = \
            np.random.choice(9, self.num_pop, p=self.censo_fracoes)

        self.alpha_r = 0.8

        zeta_idade = lambda x: power_decay(50.0, 2.0, abs(x-35))

        distribuicao_social_350 = [
            10, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3,
            4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1
        ]

        multiplic_distrib = self.multiplicador(x_vezes*y_vezes,pop_vezes)

        self.distribuicao_social = []
        for i in range(x_vezes*y_vezes):
            for j in range(len(distribuicao_social_350)):
                self.distribuicao_social \
                    += [multiplic_distrib[i]*distribuicao_social_350[j]]
        self.distribuicao_social.sort(reverse = True)
        
        self.rho_forma = 0.2 # shape factor of gamma distribution
        self.rho_escala = 5 # scale (mean value = scale * shape)
        self.pop_rho = np.random.gamma(self.rho_forma, self.rho_escala,
                                       self.num_pop) 

        self.a_kernel = 1.0
        self.b_kernel = 1.5
        self.f_kernel = partial(power_decay, self.a_kernel, self.b_kernel)

        self.num_infectados_0 = int(0.01*self.num_pop)

        self.beta_r = 0.16
        self.beta_e = 0.24
        self.beta_c = 0.04
        self.gamma = 0.1

    def inicializa_pop_estado(self):
        np.random.seed(seed = 342)
        #self.pop_estado_0 = np.ones(num_pop, dtype=np.uint8)
        self.pop_estado_0 = np.ones(self.num_pop)
        infectados_0 = np.random.choice(self.num_pop,
                                        self.num_infectados_0, 
                                        replace=False)
        #self.pop_estado_0[infectados_0] = \
        #    2*np.ones(num_infectados_0, dtype=np.uint8)
        self.pop_estado_0[infectados_0] = 2*np.ones(self.num_infectados_0)
        self.attr_estado_0 = dict([(i, {'estado': int(self.pop_estado_0[i])}) 
                                   for i in range(self.num_pop)])

    def inicializa_infeccao(self, num_infectados_0,
                            beta_r, beta_e, beta_c, gamma):

        self.num_infectados_0 = num_infectados_0

        self.inicializa_pop_estado()

        self.beta_r = beta_r
        self.beta_e = beta_e
        self.beta_c = beta_c
        self.gamma = gamma

        self.atualiza_redes()

    def cria_rede_residencial(self):

        self.G_r = nx.random_geometric_graph(
            self.num_pop, 0,
            pos=self.attr_pos, seed=1327)

        for individuos in self.res_individuos:
            if len(individuos) > 1:
                self.G_r.add_edges_from(
                    [(i,j) for i in individuos for j in individuos])

        nx.set_node_attributes(self.G_r, self.attr_estado_0)

        nx.set_node_attributes(
            self.G_r,
            dict([(i, {'faixa etária': self.pop_idades[i]})
                  for i in range(self.num_pop)])
            )

        nx.set_node_attributes(
            self.G_r,
            dict([(i, {'rho': self.pop_rho[i]}) 
                  for i in range(self.num_pop)])
            )

        self.pop_tx_transmissao_r = np.array(
            [self.beta_r / (1+self.G_r.degree(i))**self.alpha_r 
            for i in self.G_r.nodes]
            )
        attr_transmissao_r = \
            dict([(i, {'taxa de transmissao': self.pop_tx_transmissao_r[i]}) 
                  for i in self.G_r.nodes])

        nx.set_node_attributes(self.G_r, attr_transmissao_r)

#        nx.set_node_attributes(
#            self.G_r,
#            dict([(i, {'taxa de transmissao': self.pop_tx_transmissao_r[i]}) 
#                  for i in self.G_r.nodes])
#            )

        nx.set_edge_attributes(self.G_r, 1, 'weight')

    def cria_rede_empresarial(self):

        random.seed(721)
        pop_index = list(range(self.num_pop))
        membros = dict()

        for j in range(len(self.distribuicao_social)):
            individuos_aleatorios = \
                random.sample(pop_index, self.distribuicao_social[j])
            for i in individuos_aleatorios:
                pop_index.remove(i)
            membros.update({j: individuos_aleatorios})
            conexoes = [(m,n) for m in individuos_aleatorios 
                        for n in individuos_aleatorios if m != n ]

        self.G_e = nx.random_geometric_graph(
            self.num_pop, 0, pos = self.attr_pos)

        nx.set_node_attributes(self.G_e, self.attr_estado_0)

        for individuos_aleatorios in membros.values():
            conexoes = [(m,n) for m in individuos_aleatorios 
                        for n in individuos_aleatorios if m != n ]
            self.G_e.add_edges_from(conexoes)

        nx.set_edge_attributes(self.G_e, 1, 'weight')

        self.pop_tx_transmissao_e = \
            np.array([self.beta_e / (1+self.G_e.degree(i)) 
                      for i in self.G_e.nodes])
        attr_transmissao_e = dict([(i, {'taxa de transmissao': self.
                                        pop_tx_transmissao_e[i]}) 
                                   for i in self.G_e.nodes])

        nx.set_node_attributes(self.G_e, attr_transmissao_e)

    def cria_redes(self):
        self.cria_rede_residencial()
        self.cria_rede_empresarial()
        self.redes = [self.G_r, self.G_e]
        self.redes_tx_transmissao= [
            self.pop_tx_transmissao_r, 
            self.pop_tx_transmissao_e
        ]

        aux = np.array(
            [
                np.sum(
                    self.f_kernel(
                        np.linalg.norm(
                            self.pop_posicoes - self.pop_posicoes[i],
                            axis=1)
                        )
                    ) 
                    for i in range(self.num_pop)
            ]
        )
        self.pop_fator_tx_transmissao_c = self.beta_c / aux
 
    def atualiza_redes(self):

        nx.set_node_attributes(self.G_r, self.attr_estado_0)

        nx.set_node_attributes(self.G_e, self.attr_estado_0)

        self.pop_tx_transmissao_r = np.array(
            [self.beta_r / (1+self.G_r.degree(i))**self.alpha_r 
            for i in self.G_r.nodes]
            )
        attr_transmissao_r = \
            dict([(i, {'taxa de transmissao': self.pop_tx_transmissao_r[i]}) 
                  for i in self.G_r.nodes])

        nx.set_node_attributes(self.G_r, attr_transmissao_r)

        self.pop_tx_transmissao_e = \
            np.array([self.beta_e / (1+self.G_e.degree(i)) 
                      for i in self.G_e.nodes])
        attr_transmissao_e = dict([(i, {'taxa de transmissao': self.
                                        pop_tx_transmissao_e[i]}) 
                                   for i in self.G_e.nodes])

        nx.set_node_attributes(self.G_e, attr_transmissao_e)

        self.cria_redes()

class RiodeJaneiro(Cenario):

    def __init__(self, tx_reducao):
        self.nome = 'Rio de Janeiro'
        self.define_parametros(tx_reducao)
        self.inicializa_pop_estado()
        self.cria_redes()

    def define_parametros(self, tx_reducao):

        # posições dos indivíduos e de suas residências
        landscan_rio \
            = np.load('../input/dados_rio/landscan_rio.npy').astype(int)
        landscan_rio = np.maximum(landscan_rio, 0)
        
        self.pop_por_bloco = (landscan_rio/tx_reducao).astype(int)

        self.num_pop = self.pop_por_bloco.sum()

        self.pop_estado_0 = np.ones(self.num_pop)

        censo_residencial \
            = np.array([.21, .26, .20, .17, .08, .04, .02, 0.02])

        num_tam_res = len(censo_residencial) # tamanho máximo

        self.pos_residencias, self.pos_individuos, \
            self.res_individuos, self.pop_blocos_indices \
            = aloca_residencias_e_individuos(
                self.pop_por_bloco, censo_residencial)

        self.attr_pos = {j: self.pos_individuos[j] 
                         for j in range(self.num_pop)}

        self.pop_posicoes = np.array(self.pos_individuos)

        # idades
        piramide_etaria \
            = pd.read_csv('../input/dados_rio/piramide_etaria_MRJ.csv')

        idades_grupos = np.array([int(p[0:3]) 
                                  for p in piramide_etaria.columns[1:]])

        idades_fracoes_grupos \
            = piramide_etaria.iloc[0][1:].values/piramide_etaria.iloc[0][0]

        self.idade_max = 101

        self.idades_fracoes = obtem_idades_fracoes(
            idades_grupos, idades_fracoes_grupos, self.idade_max)

        self.pop_idades = gera_idades(
            self.num_pop, num_tam_res,
            self.res_individuos, self.idades_fracoes)

        # Dados para a rede escolar
        self.escolas_municipais \
            = np.load('../input/dados_rio/municipais.npy').astype(int)

        esc_escolha, self.dist_escolas \
            = rede_escolar.distribui_escolas(
                tx_reducao, self.escolas_municipais)        

        dist_idades_na_escola = rede_escolar.distribuicao_idade(
            self.num_pop, censo_residencial, self.res_individuos,
            self.pop_idades, piramide_etaria, idades_grupos,
            idades_fracoes_grupos, self.idade_max)

        self.alunos = rede_escolar.escolhe_alunos_idade(self.pop_idades,
                                                   dist_idades_na_escola)
        
        self.escolas = rede_escolar.aloca_alunos(self.alunos,
                                                 esc_escolha,
                                                 self.pos_individuos)

        # Dados para as empresas

        num_pea_2010 = np.array([0, 902115, 1448515, 495289, 241619])
        pea_fracao = 0.4885
        num_pea = int(pea_fracao * self.num_pop)
        pea_idades_faixas = np.array([0, 16, 30, 50, 60])
        pea_fracoes_faixas = num_pea_2010/num_pea_2010.sum()

        pea_fracoes = obtem_idades_fracoes(pea_idades_faixas,
                                           pea_fracoes_faixas,
                                           self.idade_max)
        
        tam_min = 3
        tam_max = 500
        z3_a = 4.5
        z3_c = 1.1
        a_dist = 3
        c_dist = 4

        self.emp_bloco_pos, self.emp_por_bloco, self.emp_tam, self.emp_membros \
            = gera_empresas(self.pop_por_bloco, self.pop_idades,
                            self.pop_blocos_indices, num_pea, pea_fracoes,
                            tam_min, tam_max, z3_a, z3_c, a_dist, c_dist)

        # Parâmetros para a infeção

        self.alpha_r = 0.8

        zeta_idade = lambda x: power_decay(50.0, 2.0, abs(x-35))

        self.rho_forma = 0.2 # shape factor of gamma distribution
        self.rho_escala = 5 # scale (mean value = scale * shape)
        self.pop_rho = np.random.gamma(self.rho_forma, self.rho_escala,
                                       self.num_pop) 

        self.a_kernel = 1.0
        self.b_kernel = 1.5
        self.f_kernel = partial(power_decay, self.a_kernel, self.b_kernel)

        self.num_infectados_0 = int(0.01*self.num_pop)

        self.beta_r = 0.16
        self.beta_esc = 0.48
        self.beta_emp = 0.24
        self.beta_c = 0.04
        self.gamma = 0.1

    def inicializa_pop_estado(self):
        #self.pop_estado_0 = np.ones(num_pop, dtype=np.uint8)
        self.pop_estado_0 = np.ones(self.num_pop)
        infectados_0 = np.random.choice(self.num_pop,
                                        self.num_infectados_0, 
                                        replace=False)
        #self.pop_estado_0[infectados_0] = \
        #    2*np.ones(num_infectados_0, dtype=np.uint8)
        self.pop_estado_0[infectados_0] = 2*np.ones(self.num_infectados_0)
        self.attr_estado_0 = dict([(i, {'estado': int(self.pop_estado_0[i])}) 
                                   for i in range(self.num_pop)])

    def inicializa_infeccao(self, num_infectados_0,
                            beta_r, beta_esc, beta_emp, beta_c, gamma):

        self.num_infectados_0 = num_infectados_0

        self.inicializa_pop_estado()

        self.beta_r = beta_r
        self.beta_esc = beta_esc
        self.beta_emp = beta_emp
        self.beta_c = beta_c
        self.gamma = gamma

        self.atualiza_redes()

    def cria_rede_residencial(self):

        self.G_r = nx.random_geometric_graph(
            self.num_pop, 0,
            pos=self.attr_pos)

        for individuos in self.res_individuos:
            if len(individuos) > 1:
                self.G_r.add_edges_from(
                    [(i,j) for i in individuos for j in individuos])

        nx.set_node_attributes(self.G_r, self.attr_estado_0)

        nx.set_node_attributes(
            self.G_r,
            dict([(i, {'faixa etária': self.pop_idades[i]})
                  for i in range(self.num_pop)])
            )

        nx.set_node_attributes(
            self.G_r,
            dict([(i, {'rho': self.pop_rho[i]}) 
                  for i in range(self.num_pop)])
            )

        self.pop_tx_transmissao_r = np.array(
            [self.beta_r / (1+self.G_r.degree(i))**self.alpha_r 
            for i in self.G_r.nodes]
            )
        attr_transmissao_r = \
            dict([(i, {'taxa de transmissao': self.pop_tx_transmissao_r[i]}) 
                  for i in self.G_r.nodes])

        nx.set_node_attributes(self.G_r, attr_transmissao_r)

        nx.set_edge_attributes(self.G_r, 1, 'weight')

    def cria_rede_escolar(self):

        self.G_esc = nx.random_geometric_graph(self.num_pop, 0,
                                               pos=self.attr_pos)

        for escola in self.escolas:
            if len(escola) > 1:
                self.G_esc.add_edges_from(
                    [(i,j) for i in escola for j in escola if i < j]
                )


        nx.set_node_attributes(self.G_esc, self.attr_estado_0)

        nx.set_edge_attributes(self.G_esc, 1, 'weight')

        self.pop_tx_transmissao_esc = \
            np.array([self.beta_esc / (1+self.G_esc.degree(i)) 
                      for i in self.G_esc.nodes])

        attr_transmissao_esc = dict([(i, {'taxa de transmissao': self.
                                        pop_tx_transmissao_esc[i]}) 
                                   for i in self.G_esc.nodes])

        nx.set_node_attributes(self.G_esc, attr_transmissao_esc)

    def cria_rede_empresarial(self):

        self.G_emp = nx.random_geometric_graph(
            self.num_pop, 0, pos = self.attr_pos)

        nx.set_node_attributes(self.G_emp, self.attr_estado_0)

        for membros in self.emp_membros:
            if len(membros) > 1:
                self.G_emp.add_edges_from(
                    [(i,j) for i in membros for j in membros if i < j])


        nx.set_edge_attributes(self.G_emp, 1, 'weight')

        self.pop_tx_transmissao_emp = \
            np.array([self.beta_emp / (1+self.G_emp.degree(i)) 
                      for i in self.G_emp.nodes])
        attr_transmissao_emp \
            = dict([(i, 
                     {'taxa de transmissao': self.pop_tx_transmissao_emp[i]}
                    ) 
                    for i in self.G_emp.nodes])

        nx.set_node_attributes(self.G_emp, attr_transmissao_emp)

    def cria_redes(self):
        self.cria_rede_residencial()
        self.cria_rede_escolar()
        self.cria_rede_empresarial()
        self.redes = [self.G_r, self.G_esc, self.G_emp]
        self.redes_tx_transmissao= [
            self.pop_tx_transmissao_r, 
            self.pop_tx_transmissao_esc,
            self.pop_tx_transmissao_emp
        ]

        aux = np.array(
            [
                np.sum(
                    self.f_kernel(
                        np.linalg.norm(
                            self.pop_posicoes - self.pop_posicoes[i],
                            axis=1)
                        )
                    ) 
                    for i in range(self.num_pop)
            ]
        )
        self.pop_fator_tx_transmissao_c = self.beta_c / aux
 
    def atualiza_redes(self):

        nx.set_node_attributes(self.G_r, self.attr_estado_0)

        nx.set_node_attributes(self.G_esc, self.attr_estado_0)

        nx.set_node_attributes(self.G_emp, self.attr_estado_0)

        self.pop_tx_transmissao_r = np.array(
            [self.beta_r / (1+self.G_r.degree(i))**self.alpha_r 
            for i in self.G_r.nodes]
            )
        attr_transmissao_r = \
            dict([(i, {'taxa de transmissao': self.pop_tx_transmissao_r[i]}) 
                  for i in self.G_r.nodes])

        nx.set_node_attributes(self.G_r, attr_transmissao_r)

        self.pop_tx_transmissao_esc = \
            np.array([self.beta_esc / (1+self.G_esc.degree(i)) 
                      for i in self.G_esc.nodes])
        attr_transmissao_esc = dict([(i, {'taxa de transmissao': self.
                                        pop_tx_transmissao_esc[i]}) 
                                   for i in self.G_esc.nodes])

        nx.set_node_attributes(self.G_esc, attr_transmissao_esc)

        self.pop_tx_transmissao_emp = \
            np.array([self.beta_emp / (1+self.G_emp.degree(i)) 
                      for i in self.G_emp.nodes])
        attr_transmissao_emp = dict([(i, {'taxa de transmissao': self.
                                        pop_tx_transmissao_emp[i]}) 
                                   for i in self.G_emp.nodes])

        nx.set_node_attributes(self.G_emp, attr_transmissao_emp)

        self.cria_redes()
