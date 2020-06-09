#!/anaconda3/envs/py38/bin/python
# -*- coding: utf-8 -*-
"""
Módulo para análise das redes utilizadas nos modelos epidemiológicos.
"""

import networkx as nx

from sys import getsizeof

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

def exibe_memoria(num):
    '''
    Retorna um número em unidades adequadas de memória.
    '''
    
    size_units = {'byte(s)': 1, 'Kb': 1024, 'Mb': 1048576, 'Gb': 1073741824, 'Tb': 1099511627776}
    for u,v in size_units.items():
        if num / v >= 1:
            footprint = f'{num/v:.1f} {u}'
    return footprint

def num_medio_conexoes_peso(G):
    '''
    Retorna o número médio de conexões ponderado pelo atributo 'weight' de
    cada aresta.
    '''

    return sum([d[1] for d in G.degree(weight='weight')])/G.number_of_nodes()

def analise_rede(G, info=True, node_size=0, pos=None, hist=False):
    '''
    Exibe diversas propriedades da rede.
    '''

    num_vertices = G.number_of_nodes()
    num_arestas = G.number_of_edges() # usado na versão antiga
    num_arestas_weighted = G.size(weight='weight') # usado na versão atual
    num_arestas_por_vertice_weighted = [j for i,j in G.degree(weight='weight')]
    num_medio_conexoes = 2*num_arestas_weighted/num_vertices

    if info:
        mem_G = getsizeof(dict(G.nodes(data=True))) + getsizeof(list(G.edges(data=True)))
        print(f'Memória utilizada pelo grafo: {exibe_memoria(mem_G)}')
        print(f'Memória utilizada pela matriz de adjacências: {exibe_memoria(getsizeof(nx.to_numpy_array(G)))}')
        print(f'Número de vértices: {num_vertices}')
        print(f'Número de arestas: {num_arestas}')
        print(f'Número de arestas com peso: {num_arestas_weighted}')
        print(f'Número médio de conexões por indivíduo: {num_medio_conexoes:.1f}')
        print(f'Número médio de conexões por indivíduo com peso: {num_medio_conexoes_peso(G):.1f}')

    if node_size:
        color = ['tab:blue', 'tab:red', 'tab:green']
        pop_estado = nx.get_node_attributes(G,'estado')
        color_map = [color[pop_estado[j]-1] for j in range(num_vertices)]
        plt.figure(figsize=(10,6))
        if pos:
            nx.draw(G, pos, node_size=node_size, node_color=color_map, alpha=0.5)
        else:
            nx.draw(G, node_size=node_size, node_color=color_map, alpha=0.5)
        plt.title('Rede de indivíduos e de suas conexões', fontsize=16)
        plt.show()

    if hist:
        plt.figure(figsize=(10,6))
        plt.hist(num_arestas_por_vertice_weighted, 50, facecolor='g', alpha=0.75)
        plt.xlabel('num. arestas', fontsize=14)
        plt.ylabel('num. vertices', fontsize=14)
        plt.title('Histograma com a quantidade de indivíduos por número de conexões', fontsize=16)
        plt.show()
