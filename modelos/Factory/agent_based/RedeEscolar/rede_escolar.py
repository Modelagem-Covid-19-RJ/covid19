import numpy as np
import random

import networkx as nx

# fracoes_grupos_2 = array([0.14666667, 0.40888889, 0.        , 0.        , 0.        ])
# fracoes_grupos_3 = array([0.        , 0.        , 0.25569862, 0.        , 0.        ])
# fracoes_grupos_5 = array([0.        , 0.        , 0.        , 0.23945578, 0.        ])
# fracoes_grupos_6 = array([0.14346756, 0.39997017, 0.45939074, 0.        , 0.        ])
# fracoes_grupos_7 = array([0.        , 0.        , 0.        , 0.        , 0.00673317])
# fracoes_grupos_15 = array([0.        , 0.        , 0.07249391, 0.16689342, 0.        ])
# fracoes_grupos_21 = array([0.        , 0.        , 0.04097482, 0.        , 0.01296758])
# fracoes_grupos_30 = array([0.0303915 , 0.08472782, 0.09731519, 0.22403628, 0.        ])
# fracoes_grupos_35 = array([0.        , 0.        , 0.        , 0.11882086, 0.01633416])
# fracoes_grupos_42 = array([0.00689038, 0.01920955, 0.02206336, 0.        , 0.00698254])
# fracoes_grupos_105 = array([0.        , 0.        , 0.0137896 , 0.03174603, 0.00436409])
# fracoes_grupos_210 = array([0.00258389, 0.00720358, 0.00827376, 0.01904762, 0.00261845])

def distribuicao_idade(num_pop, censo_residencial, res_individuos, pop_idades, piramide_etaria, idades_grupos, idades_fracoes_grupos, idade_max):
    num_pop_0a19 = int(num_pop*np.sum(idades_fracoes_grupos[0:20]))

    p_esc_a_idades_grupos = np.array([0, 4, 6, 15, 20, 100])
    p_esc_a_idades_fracoes_grupos = np.array([.33, .92, .97, .80, 0.05])
    num_tam_res = len(censo_residencial)
    distrib_tam_por_res = np.array([len(res_individuos[k]) for k in range(len(res_individuos))])
    distrib_res = np.array([len(distrib_tam_por_res[distrib_tam_por_res == j]) for j in range(1,num_tam_res+1)])

    p_esc_a_idades_fracoes = list()
    for j in range(len(p_esc_a_idades_grupos)-1):
        p_esc_a_idades_fracoes += (p_esc_a_idades_grupos[j+1] - p_esc_a_idades_grupos[j])*[p_esc_a_idades_fracoes_grupos[j]]

    p_esc_a_idades_fracoes = np.array(p_esc_a_idades_fracoes)

    num_ativos_escola = int(p_esc_a_idades_fracoes.sum()/100 * num_pop) # adicionar a taxa de redução como propriedade do obj

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