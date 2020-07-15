def carrega_dic_casos(file):
    lst = np.load(file)
    dic_casos = {i+1: j for (i,j) in enumerate(lst)}
    return dic_casos

def corrige_subnotificacao(dic_casos):
    rng = np.arange(1,164)
    itens = map(lambda x: x*8, dic_infectados.values())
    dic_casos_corrigida = {i: j for i,j in zip(rng,itens)}
    return dic_casos_corrigida

def corrige_mtrx(mtrx, tipo = 0):
    if tipo == 0:
        new_mtrx = np.zeros(np.shape(mtrx))
        for i in range(39):
            new_mtrx[np.abs(i-38)] = mtrx[i]
    else:
        new_mtrx = np.zeros(np.shape(mtrx))
        for i in range(390):
            new_mtrx[np.abs(i-389)] = mtrx[i]
    return new_mtrx

def reducao_casos(tx_reducao, dic_casos):
    if tx_reducao == 1:
        dic_casos_reduzida = dic_casos
    else:
        lst_casos = np.array(list(dic_casos.values()))
        qt_total = np.rint(np.sum(lst_casos)/tx_reducao)
        rng_blocos = np.arange(len(lst_casos))
        pesos = (lst_casos/np.sum(lst_casos))
        escolha = np.array(random.choices(rng_blocos, pesos, k = int(qt_total)))
        lst_reduzida_casos = np.zeros(len(lst_casos))
        for i in escolha:
            lst_reduzida_casos[i] += 1
        dic_casos_reduzida = {i+1: j for (i,j) in enumerate(lst_reduzida_casos)}
    return dic_casos_reduzida

def distribuicao_inicial_casos(tx_reducao, dic_casos, mtrx_bairros, res_posicoes, res_individuos):    
    mtrx_bairros = corrige_mtrx(mtrx_bairros, 1)
    pos_res_blocos = np.round(np.array(res_posicoes)*[9.9,9.8])
    posicao_bairro = np.array([mtrx_bairros[int(x[1]), int(x[0])] for x in pos_res_blocos])
    dic_casos_reduzida = reducao_casos(tx_reducao, dic_casos)
    
    beta = .6
    casos = []
    ids_bairros = np.arange(1,164)
    rng = np.arange(len(pos_res_blocos))
    for i in ids_bairros:
        res_no_bairro = posicao_bairro == i
        indices_no_bairro = rng[res_no_bairro]
        qt_casos = dic_casos_reduzida[i]
        if (qt_casos > 0) & (len(indices_no_bairro) > 0):
            j = 0
            dist = 0
            permt_residencias = np.random.permutation(indices_no_bairro)
            while(qt_casos > 0 and j < len(permt_residencias)):
                res_caso = permt_residencias[j]
                caso_primario = np.random.choice(res_individuos[res_caso])
                indice_primario = res_individuos[res_caso].index(caso_primario)
                qt_casa = len(res_individuos[res_caso])
                infecta = np.random.rand(qt_casa)
                infecta[indice_primario] = 1
                rng_res = np.arange(qt_casa)
                indice_infectados = rng_res[infecta > beta]
                infectados = np.array(res_individuos[res_caso])[indice_infectados]
                if qt_casos - len(infectados) < 0:
                    infectados = infectados[:int(qt_casos)]
                qt_casos -= len(infectados)
                dist += len(infectados)
                casos.append(infectados)
                j += 1
    return np.hstack(casos)

def inicia_casos(tx_reducao, dic_casos_file, mtrx_bairros_fino, pos_residencias, res_individuos):
    mtrx_bairros_fino_corrigida = corrige_mtrx(mtrx_bairros_fino, 1)
    dic_casos = carrega_dic_casos(dic_casos_file)
    dic_casos_reduzida = reducao_casos(tx_reducao,dic_casos)
    return distribuicao_inicial_casos(tx_reducao, dic_casos_reduzida, mtrx_bairros_fino, pos_residencias, res_individuos)