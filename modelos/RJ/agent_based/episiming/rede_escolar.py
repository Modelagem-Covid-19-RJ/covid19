import numpy as np
import random

import networkx as nx

def distribuicao_idade(num_pop, censo_residencial, res_individuos, pop_idades, piramide_etaria, idades_grupos, idades_fracoes_grupos, idade_max):
    num_pop_0a19 = int(num_pop*np.sum(idades_fracoes_grupos[0:20]))

    p_esc_a_idades_grupos = np.array([0, 2, 3, 6, 11, 15, 17, 19, 20, 100])
    p_esc_a_idades_fracoes_grupos = np.array([0, .4, .8, .95, .95, .95, .95, .75, 0])
    num_tam_res = len(censo_residencial)
    distrib_tam_por_res = np.array([len(res_individuos[k]) for k in range(len(res_individuos))])
    distrib_res = np.array([len(distrib_tam_por_res[distrib_tam_por_res == j]) for j in range(1,num_tam_res+1)])

    p_esc_a_idades_fracoes = list()
    for j in range(len(p_esc_a_idades_grupos)-1):
        p_esc_a_idades_fracoes += (p_esc_a_idades_grupos[j+1] - p_esc_a_idades_grupos[j])*[p_esc_a_idades_fracoes_grupos[j]]

    p_esc_a_idades_fracoes = np.array(p_esc_a_idades_fracoes)

    num_ativos_escola = int(p_esc_a_idades_fracoes.sum()/100 * num_pop)

    pessoas_por_idade = list()

    for idade_escolar in range(20):
        pessoas_por_idade.append(np.count_nonzero(pop_idades == idade_escolar))
    
    pessoas_por_idade = np.array(pessoas_por_idade)
    idades_na_escola_fracoes = p_esc_a_idades_fracoes[:20]
    idades_na_escola = np.round(idades_na_escola_fracoes*pessoas_por_idade).astype(int)
    return idades_na_escola


def escolhe_alunos_idade(pop_idades, idades_na_escola):
    alunos = [np.random.choice(np.arange(len(pop_idades))[pop_idades == i], idades_na_escola[i], replace = False) for i in range(20)]
    return alunos

def distribui_escolas(tx_reducao, escolas):
    total_escolas = 3065
    rng_escolas = np.arange(np.prod(np.shape(escolas)))
    weight_escolas = (escolas/np.sum(escolas)).flatten()

    index_escolas = rng_escolas[weight_escolas != 0]  ## Guardando o indice das escolas
    weight_escolas = weight_escolas[weight_escolas != 0]
    esc_escolha = np.array(random.choices(index_escolas, weight_escolas, k = np.floor(np.sum(escolas*2/tx_reducao)).astype(int)))

    row_escolas = np.floor(esc_escolha/83)
    col_escolas = np.mod(esc_escolha,83)
    escolas_escolha_por_blocos = np.zeros((39,83))
    for i,j in zip(row_escolas,col_escolas):
        escolas_escolha_por_blocos[int(i)][j] += 1

    return esc_escolha, escolas_escolha_por_blocos

def aloca_alunos(alunos, escola_escolha, pos_individuos):
    row_escolas = np.floor(escola_escolha/83)
    col_escolas = np.mod(escola_escolha,83)

    pos_x_escolas = col_escolas + np.random.rand(len(escola_escolha))
    pos_y_escolas = row_escolas + np.random.rand(len(escola_escolha))
    pos_escolas = np.array([pos_x_escolas,pos_y_escolas]).T
    alunos_array = np.hstack(alunos)

    pos_individuos_escolas = np.array(pos_individuos)[alunos_array]
    dist_indiv_esc = [np.linalg.norm(p_i - pos_escolas, axis = 1).argsort()[:3] for p_i in pos_individuos_escolas]

    indiv_esc = [random.choices(i)[0] for i in dist_indiv_esc]

    escolas = list()
    for i in range(len(escola_escolha)):
        aux = []
        for j in range(len(indiv_esc)):
            if indiv_esc[j] == i:
                aux.append(alunos_array[j])
        escolas.append(aux)

    return escolas

def gera_rede_escolar(num_pop, attrib_pos_individuos, escolas):
    G_esc = nx.random_geometric_graph(num_pop, 0, pos=attrib_pos_individuos)

    for aluno in escolas:
        if len(aluno) > 1:
            G_esc.add_edges_from([(i,j) for i in aluno for j in aluno if i < j])
    return G_esc