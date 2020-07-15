def refina_matriz(matriz, filtro):
    '''
    Refina uma matriz via interpolação linear e uma matriz fina como filtro.  
    '''
    matriz_fina = np.zeros_like(filtro)
    loc_sobra = list() 
    
    tx_refinamento_x = int(filtro.shape[1]/matriz.shape[1])
    tx_refinamento_y = int(filtro.shape[0]/matriz.shape[0])
    xs = list(range(matriz.shape[1]))
    ys = list(range(matriz.shape[0]))

    xs_fino = np.arange(0, matriz.shape[1], 1/tx_refinamento_x)
    ys_fino = np.arange(0, matriz.shape[0], 1/tx_refinamento_y)

    f = interp2d(xs, ys, matriz, kind='linear')
    matriz_interp = f(xs_fino, ys_fino)*np.minimum(filtro,1)

    for j in xs:
        for i in ys:
            if matriz[i,j]:
                matriz_interp_local \
                    = matriz_interp[i*tx_refinamento_y:(i+1)*tx_refinamento_y,
                                    j*tx_refinamento_x:(j+1)*tx_refinamento_x]
                if matriz_interp_local.sum() == 0:
                    loc_sobra.append([i,j])
                else:
                    distrib = np.floor(matriz[i,j]*matriz_interp_local
                                           / matriz_interp_local.sum()
                                      ).astype('int')
                    sobra = matriz[i,j] - distrib.sum()
                    sobra_posicionamento \
                        = np.random.choice(tx_refinamento_x*tx_refinamento_y,
                                           int(sobra),
                                           replace=False,
                                           p=(matriz_interp_local
                                              /matriz_interp_local.sum()
                                             ).flatten()
                                          )
                    for loc in sobra_posicionamento:
                        distrib[loc // tx_refinamento_x,
                                loc % tx_refinamento_x] += 1
                    matriz_fina[i*tx_refinamento_y:(i+1)*tx_refinamento_y,
                                j*tx_refinamento_x:(j+1)*tx_refinamento_x] \
                        = distrib

    matriz_sobra = np.zeros_like(matriz)
    for ij in loc_sobra:
        matriz_sobra[ij[0], ij[1]] = matriz[ij[0], ij[1]]
    
    return matriz_fina, matriz_sobra


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


def reducao_escolas(tx_reducao, mtrx_escolas):
    """Função que faz a redução da quantidade de escolas em dado Heatmap

    Args:
        tx_reducao (float): A taxa de redução a qual esta submetido o cenário.
        mtrx_escolas (array): Array que contém as matrizes/heatmaps da distribuição das escolas por 
        modadlidade, também pode ser passado somente uma matriz.

    Returns:
        array: Array de matrizes/heatmaps reduzidos
    """
    ## Caso a taxa de redução seja 1, não fazer nada
    if tx_reducao == 1:
        return mtrx_escolas
    else:
        ## Caso seja passado somente um heatmap, ao inves de uma lista de heatmaps
        ## Transforma em uma lista unitária de heatmap
        if len(mtrx_escolas.shape) < 3:
            mtrx_escolas = np.array([mtrx_escolas])
        total_escolas = np.sum(mtrx_escolas)
        rng = np.arange(np.prod(mtrx_escolas[0].shape))
        ## Para cada modalidade, a quantidade de escolas reduzidas
        qt_modalidade = map(lambda x: np.rint(np.sum(x)/tx_reducao).astype(int), mtrx_escolas)
        ## Faz uma escolha aleatória com pesos nos blocos do heatmap de cada modalidade
        pesos_modalidade = map(lambda x: (x/np.sum(x)).flatten(), mtrx_escolas)
        modalidade_reduzidas = map(lambda x,y: random.choices(rng, x, k = y), pesos_modalidade, qt_modalidade)

        ## Preenchemos, para cada modalidade, os blocos sorteados
        mtrx_reduzida = np.zeros(mtrx_escolas.shape)
        for i,m in enumerate(modalidade_reduzidas):
            m = np.array(m)
            row_escolas = np.floor(m/mtrx_escolas.shape[-1])
            col_escolas = np.mod(m,mtrx_escolas.shape[-1])
            for k,j in zip(row_escolas,col_escolas):
                mtrx_reduzida[i][int(k)][j] += 1
        return mtrx_reduzida

def dist_idade_modalidade(idade_max, grupos_idade_escolar, dist_grupos_idade):
    """Função que calcula a quantidade de matriculas por idade e modalidade escolar, segundo censo

    Args:
        idade_max (int): Idade max da população do cenário
        grupos_idade_escolar (array): Array com a separação de idades por grupo, onde cada par de elementos
        representa um intervalo do tipo [a,b)
        dist_grupos_idade (array): Array com a distribuição de cada grupo de idades, segundo censo.
        Pode ser passado apenas um array com a distribuição, ou um array de arrays com distribuição
        que representa a distribuição por modalidades

    Returns:
        array: Array com matrizes de tamanho 100, representando a quantidade total de cada idade de aluno em 
        uma modalidade
    """
    
    ## Caso seja passado somente um array em dist_grupos_idade
    if len(dist_grupos_idade.shape) < 2:
        dist_grupos_idade = np.array([dist_grupos_idade])
    
    ## Matriz com a distribuição de idade, separado idade a idade
    dist_idade = np.zeros((len(dist_grupos_idade),idade_max))
    
    ## Transforma a informação de distribuição por grupo de idade para distribuição por idade
    for j in range(len(dist_grupos_idade)):
        for i in range(len(grupos_idade_escolar)):
            if i < len(dist_grupos_idade[j]):
                dist_idade[j][grupos_idade_escolar[i]:grupos_idade_escolar[i+1]] = dist_grupos_idade[j][i]
                
    ## Calcula a quantidade de alunos na idade, por modalidade
    dist_idade_pop = np.array([np.count_nonzero(pop_idades == i) for i in range(idade_max)])
    dist_idade_escolar = np.round(dist_idade[:grupos_idade_escolar[-2]]*dist_idade_pop)
    return dist_idade_escolar

def escolhe_alunos_idade(pop_idades, idades_na_escola):
    """Função que escolhe uma quantidade de individuos na população, pela idade, para serem matriculados
    em alguma escola

    Args:
        pop_idades (list(int)): Uma lista da idade de cada invididuo na população
        idades_escola (list(int)): Uma lista com a distribuição de alunos por idade

    Returns:
        array: Array com os indices dos alunos na população, separados pela distribuição de idade
    """
    alunos_modalidade = []
    for j in range(len(idades_na_escola)):
        alunos = np.array([np.random.permutation(np.arange(len(pop_idades))[pop_idades == i])[:int(idades_na_escola[j][i])] for i in range(len(idades_na_escola[j]))])
        alunos_modalidade.append(alunos)
    return np.array(alunos_modalidade)

def gera_pos_escolas(mtrx_escolas):
    """Função que gera as escolas, indexando pela sua posição

    Args:
        mtrx_escolas (array): Lista com heatmaps de escolas por modalidade

    Returns:
        array: Lista com a posição de cada escola, por modalidade
    """
    
    pos_escolas_modalidade = []
    rng = np.arange(np.prod(mtrx_escolas[0].shape))
    for m in mtrx_escolas:
        qt_por_pos = m[m > 0].astype(int)
        pos = rng[m.flatten() > 0]
        aux = []
        for p,q in zip(pos,qt_por_pos):
            for i in range(q):
                aux.append((np.mod(p,m.shape[1]), p//m.shape[1]))
        pos_escolas_modalidade.append(aux)
    return pos_escolas_modalidade

def aloca_alunos(alunos, pos_escolas, pos_individuos):
    """Função que aloca um aluno em uma das 3 escolas mais próximas de sua posição

    Args:
        alunos (array): Array com os indices dos alunos na população, separados pela distribuição de idade,
        este input é o output da função escolhe_aluno_idades
        pos_escolas (array): Lista com a posição de cada escola, por modalidade, este input é o
        output da função gera_pos_escolas

    Returns:
        array: Indice de alunos na população, separados por modalidade de escola
    """
    
    matriculas_modalidade = [[ [] for _ in range(len(p))] for p in pos_escolas]
    for i in range(len(alunos)):

        alunos_modalidade = np.hstack(alunos[i])
        pos_alunos_modalidade = np.array(pos_individuos)[alunos_modalidade]*10
        dist_aluno_esc = [np.linalg.norm(p_i - pos_escolas[i], axis = 1).argsort()[:6] for p_i in pos_alunos_modalidade if len(pos_escolas[i]) > 0]
        matricula = [random.choices(k)[0] for k in dist_aluno_esc]
        for k in range(len(matricula)):
            matriculas_modalidade[i][matricula[k]].append(alunos_modalidade[k])
            
    return np.array(matriculas_modalidade)

def gera_rede_escolar(tx_reducao, pos_individuos, pop_idades, mtrx_escolas, grupos, fracoes_grupos):
    escolas_reduzida = reducao_escolas(tx_reducao, escolas)
    escolas_reduzida_corrigidas = [corrige_mtrx(i) for i in escolas_reduzida]
    mtrx_escolas_reduzida = np.array([refina_matriz(i, bairros_fino)[0] for i in escolas_reduzida_corrigidas])
    qt_idade_modalidade = dist_idade_modalidade(100, grupos, fracoes_grupos)
    alunos = escolhe_alunos_idade(pop_idades, qt_idade_modalidade)
    pos_escolas_modalidade = gera_pos_escolas(mtrx_escolas_reduzida)
    matriculas = aloca_alunos(alunos, pos_escolas_modalidade, pos_individuos)
    return matriculas

def gera_grafo_rede_escolar(pos_individuos, matriculas):
    attrib_pos_individuos = {j: pos_individuos[j] for j in range(len(pos_individuos))}
    G_esc = nx.random_geometric_graph(num_pop, 0, pos = attrib_pos_individuos)
    for modalidade in matriculas:
    for escola in modalidade:
            G_esc.add_edges_from([(i,j) for i in escola for j in escola if i < j])
    return G_esc