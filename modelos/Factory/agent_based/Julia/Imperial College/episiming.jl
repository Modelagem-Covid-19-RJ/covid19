using StatsBase

struct Bairro
    nResidencias::Integer # numero de residencias naquele bairro
    nPessoas::Integer # numero de pessoas naquele bairro
    residencias::Array{T} where T <: Integer # indice das residencias do bairro
    pessoas::Array{T} where T <: Integer # indice das pessoas do bairro
    distancias::Array{T} where T <: Number # distancia media entre uma pessoa do bairro e os demais bairros
end

struct Residencia
    n::Integer # numero de pessoas naquela residencia
    pessoas::Array{T} where T <: Integer # indice das pessoas da residencia
    bairro::Integer # indice do bairro que a residencia pertence
    posicao::Array{T} where T <: Number # posicao da residencia no mapa
end

struct Social
    n::Integer # numero de pessoas naquela rede social
    pessoas::Array{T} where T <: Integer # pessoas que coimpoe a rede social
end

struct Pessoas
    residencias::Array{T} where T <: Integer # lista com indices da residencia onde cada uma das pessoas residem
    bairros::Array{T} where T <: Integer # lista com indices do bairro onde cada uma das pessoas residem
    social::Array{T} where T <: Integer # lista com indices da rede social na qual cada uma das pessoas pertencem
    posicao::Array{T, 2} where T <: Number # posicao de cada pessoa
end

struct Populacao
    rodada::Integer # n da rodada (nome da pasta com arquivos)
    n::Integer # tamanho da população
    I0::Integer # quantidade inicial de infectados
    estadoInicial::Array{T} where T <: Integer # estado inicial para cada individuo
    estadoAtual::Array{T} where T <: Integer # estado inicial de cada individuo
    pessoas::Pessoas # pessoas que compoem a populacao
    residencias::Array{Residencia} # lista com as residencias
    bairros::Array{Bairro} # lista de bairros
    sociais::Array{Social} # lista de redes sociais
end

function Populacao(n::Integer, I0::Integer, estadoInicial::Array{T} where T <: Integer, estadoAtual::Array{T} where T <: Integer, pessoas::Pessoas, residencias::Array{Residencia}, bairros::Array{Bairro}, sociais::Array{Social})
    return Populacao(-1, n, I0, estadoInicial, estadoAtual, pessoas, residencias, bairros, sociais)
end

function selectiveRand(v)
    """
        Função para geração de números aleatórios de acordo com um vetor booleano
    
        Parametros:
            v: vetor booleano
        
        Saida:
            aux: vetor com um número aleatório nas entradas verdadeiras de v o 0 nas demais
    """
    aux = zeros(length(v))
    aux[v] .= rand(sum(v))
    return aux
end

function rowWiseNorm(A)
    """
        Função para calcular a norma dos vetores linha de uma matriz
    """
    return sqrt.(sum(abs2, A, dims=2)[:, 1])
end

function calculaDistanciaFina(populacao::Populacao, bairro::Bairro, suscetiveisBairro, infectadosBairro, fKernel; T=Float16)
    """
        Cálculo da distancia entre as pessoas de um mesmo bairro
    """
    aux = zeros(T, length(suscetiveisBairro), length(infectadosBairro))
    Threads.@threads for i in 1:length(suscetiveisBairro)
        aux[i, :] += fKernel(populacao.pessoas.posicao[infectadosBairro, :] .- populacao.pessoas.posicao[suscetiveisBairro[i], :]')
    end
    return aux
end

function escreveDistancias(populacao, rodada, fKernel; T=Float16)
    """
        Calcula e escreve a matriz de distancias entre pessoas do mesmo bairro, para todos os bairros
    """
    rm(joinpath("saidas", string(rodada)), recursive=true, force=true)
    mkdir(joinpath("saidas", string(rodada)))
    dir = joinpath("saidas", string(rodada), "dist")
    mkdir(dir)
    for i in 1:length(populacao.bairros)
        bairro = populacao.bairros[i]
        file = open(joinpath(dir, string(i) * ".eps"), "w")
        aux = Array{T, 2}(calculaDistanciaFina(populacao, bairro, bairro.pessoas, bairro.pessoas, fKernel))
        write(file, aux)
        close(file)
        aux = nothing
        file = nothing
        GC.gc()
    end
end

function abreEps(rodada, i, T=Float16)
    """
        Abre a matriz de distancias de um bairro
    """
    file = open(joinpath("saidas", string(rodada), "dist", string(i) * ".eps"))
    aux = reinterpret(T, read(file))
    close(file)
    n = Int(sqrt(length(aux)))
    return reshape(aux, (n, n))
end

function leDistancia(populacao::Populacao, suscetiveis, infectados, fKernel)
    """
        Calcula soma das distancias entre os infectados e os suscetiveis, 
        tomando a distancia media entre pessoas de bairros diferentes e lendo a matriz de distancias
        para pessoas do mesmo bairro.
    """
    aux = zeros(populacao.n)
    infectadosBairros = [sum(infectados[i.pessoas]) for i in populacao.bairros]
    Threads.@threads for i in 1:length(populacao.bairros)
        bairro = populacao.bairros[i]
        suscetiveisBairro = suscetiveis[bairro.pessoas]
        infectadosBairro = infectados[bairro.pessoas]
        aux[bairro.pessoas[suscetiveisBairro]] .+= sum(infectadosBairros .* bairro.distancias)
        aux[bairro.pessoas[suscetiveisBairro]] .+= sum(abreEps(populacao.rodada, i)[suscetiveisBairro, infectadosBairro], dims=2)[:, 1]
    end
    return aux[suscetiveis]
end

function calculaDistancia(populacao::Populacao, suscetiveis, infectados, fKernel)
    """
        Calculo da distância entre todas as pessoas, tomando a distnacia media entre pessoas de bairros diferentes
    """
    aux = zeros(populacao.n)
    infectadosBairros = [sum(infectados[i.pessoas]) for i in populacao.bairros]
    for bairro in populacao.bairros
        suscetiveisBairro = bairro.pessoas[suscetiveis[bairro.pessoas]]
        boolInfectadosBairro = infectados[bairro.pessoas]
        infectadosBairro = bairro.pessoas[boolInfectadosBairro]
        aux[suscetiveisBairro] .+= sum(infectadosBairros .* bairro.distancias)
        aux[suscetiveisBairro] .+= sum(calculaDistanciaFina(populacao, bairro, suscetiveisBairro, infectadosBairro, fKernel), dims=2)[:, 1]
    end
    return aux[suscetiveis]
end

function miniPassoMatricial(estados::Array{T} where T <: Integer)
    """
        Passo matricial para calcular exposicao de uma rede completa
    """
    popSuscetiveis = estados.==1
    popInfectados = estados.==2
    
    n = length(estados)
    nInfectados = sum(popInfectados)
    nSuscetiveis = sum(popSuscetiveis)

    expostos = zeros(Bool, n, n)
    expostos[popSuscetiveis, popInfectados] .= true
    return sum(expostos, dims=2)[:, 1]
end

function passoMisto(populacao::Populacao, 
        txTransmissaoR::Array{T} where T <: Number, txTransmissaoS::Array{T} where T <: Number, txTransmissaoG::Array{T} where T <: Number, 
        γ::Number, δ::Number, fKernel)
    """
        Entrada:
            populacao: 
            txTransmissaoR: taxa de transmissão residencial
            txTransmissaoS: taxa de transmissão social
            txTransmissaoG: taxa de transmissão global
            γ: probabilidade de não recuperação
            δ: tamanho do passo temporal
            fKernel: ?
    """
    popSuscetiveis = populacao.estadoAtual.==1
    popInfectados = populacao.estadoAtual.==2
    
    contatos = zeros(populacao.n)
    Threads.@threads for i in populacao.residencias
        contatos[i.pessoas] .+= sum(popInfectados[i.pessoas]) .* txTransmissaoR[i.pessoas]
        #contatos[i.pessoas] .+= miniPassoMatricial(populacao.estadoAtual[i.pessoas]) .* txTransmissaoR[i.pessoas]
    end
    
    Threads.@threads for i in populacao.sociais
        contatos[i.pessoas] .+= sum(popInfectados[i.pessoas]) .* txTransmissaoS[i.pessoas]
        #contatos[i.pessoas] .+= miniPassoMatricial(populacao.estadoAtual[i.pessoas]) .* txTransmissaoS[i.pessoas]
    end
    
    if populacao.rodada == -1
        contatos[popSuscetiveis] .+= calculaDistancia(populacao, popSuscetiveis, popInfectados, fKernel) .* txTransmissaoG[popSuscetiveis]
    else
        contatos[popSuscetiveis] .+= leDistancia(populacao, popSuscetiveis, popInfectados, fKernel) .* txTransmissaoG[popSuscetiveis]
    end

    prob = exp.(- δ .* contatos)
    
    novosInfectados = selectiveRand(popSuscetiveis) .> prob
    novosRecuperados = selectiveRand(popInfectados) .> γ
    
    infectados = ((popInfectados .& (.~novosRecuperados)) .| novosInfectados)
    suscetiveis = (popSuscetiveis .& (.~novosInfectados))

    return 3 .* ones(Int, populacao.n) .- 2 .* suscetiveis .- infectados
end

function evolucaoMista(
        populacao::Populacao, tempos::AbstractArray{T} where T <: Number, nSim::Integer, 
        txTransmissaoR::Array{T} where T <: Number, txTransmissaoS::Array{T} where T <: Number, txTransmissaoG::Array{T} where T <: Number, γ::Number, fKernel; timing=false)
    """
        Entrada:
            resumoPop: 
            tempos:
            txTransmissaoR: taxa de transmissão residencial
            txTransmissaoS: taxa de transmissão social
            txTransmissaoG: taxa de transmissão global
            γ: parâmetro da exponencial de não recuperação
            δ: tamanho do passo temporal
            fKernel: ?
    """    
    nT = length(tempos)
    passos = tempos[2:end] - tempos[1:(end-1)]
    Γ = exp.(- γ .* passos)
    
    S = zeros(nSim, nT)
    I = zeros(nSim, nT)
    R = zeros(nSim, nT)
    
    S[:, 1] .= populacao.n - populacao.I0
    I[:, 1] .= populacao.I0
    
    for j in 1:nSim
        populacao.estadoAtual .= populacao.estadoInicial
        for (k, δ) in enumerate(passos)
            if timing
                @time populacao.estadoAtual .= passoMisto(populacao, txTransmissaoR, txTransmissaoS, txTransmissaoG, Γ[k], δ, fKernel)
            else
                populacao.estadoAtual .= passoMisto(populacao, txTransmissaoR, txTransmissaoS, txTransmissaoG, Γ[k], δ, fKernel)
            end
            
            S[j, k+1] = sum(populacao.estadoAtual .== 1)
            I[j, k+1] = sum(populacao.estadoAtual .== 2)
            R[j, k+1] = sum(populacao.estadoAtual .== 3)
        end
    end
    return S,I,R
end

function geraQuadrado(nPessoas, geradorResidencias, shift, nBai, limitPop=5000)
    residencia = geradorResidencias(nPessoas)
    nResidencias = maximum(residencia)
    residencias = [Int[] for i in 1:nResidencias]   
    for i in 1:nResidencias
        append!(residencias[i], (1:nPessoas)[residencia .== i])
    end
    residenciasPos = rand(nResidencias, 2)
    
    sqnDivisoes = ceil(Int, sqrt(nPessoas / limitPop))

    residenciasBai = [
        ceil(Int, residenciasPos[i, 1] * sqnDivisoes) + 
        floor(Int, residenciasPos[i, 2] * sqnDivisoes) * sqnDivisoes
        for i in 1:nResidencias
    ]
    
    listaResidencias = [Residencia(length(j), j, residenciasBai[i] + nBai, residenciasPos[i, :] .+ shift) for (i, j) in enumerate(residencias)]

    # determina a posicao e bairro de cada pessoa
    posicao = zeros(nPessoas, 2)
    bairro = zeros(Int, nPessoas)
    for i in 1:nPessoas
        posicao[i, :] .= residenciasPos[residencia[i], :] .+ shift
        bairro[i] = residenciasBai[residencia[i]]
    end
    
    return sqnDivisoes, residencia, bairro, posicao, listaResidencias, residenciasBai
end

function distribuiPessoas(dadosPessoas, geradorResidencias, I0, nSociais, fKernel, limitPop=5000; writeDisk=false)
    nPessoas = sum(dadosPessoas)
    (n, m) = size(dadosPessoas)
    
    indices = hcat([[i + (j - 1) * n for i in 1:n] for j in 1:m]...);
    indices = indices[dadosPessoas .> 0]
    
    residencias = Int[] # tamanho == nPessoas
    bairros = Int[] # tamanho == nPessoas
    residenciasBai = Int[] # tamanho == nResidencias
    listaResidencias = Residencia[] # tamanho == nResidencias
    posicoes = zeros(0, 2)
    nRes = 0
    nBai = 0
    for k in indices
        j = ceil(Int, k / n)
        i = k - (j - 1) * n
        (sqnDivisoes, residencia, bairro, posicao, listaResidencia, residenciaBai) = geraQuadrado(dadosPessoas[k], geradorResidencias, [j-1, i-1], nBai, limitPop)
        
        append!(listaResidencias, listaResidencia)
        append!(residenciasBai, residenciaBai .+ nBai)
        append!(residencias, residencia .+ nRes)
        append!(bairros, bairro .+ nBai)
        posicoes = vcat(posicoes, posicao)
        nRes += length(listaResidencia)
        nBai += sqnDivisoes * sqnDivisoes
    end
    
    social = rand(1:nSociais, nPessoas)
    sociais = [Social(sum(social .== i), (1:nPessoas)[social .== i]) for i in 1:nSociais]
    
    pessoas = Pessoas(residencias, bairros, social, posicoes)
    
    bairrosResidencias = [(1:nRes)[residenciasBai .== i] for i in 1:nBai]
    bairrosPessoas = [(1:nPessoas)[bairros .== i] for i in 1:nBai]
    
    # calcula o centro do bairro e a distancia media entre os bairros
    centros = vcat([mean(posicoes[i, :], dims=1) for i in bairrosPessoas]...)
    dist = hcat([fKernel(centros .- centros[i, :]') for i in 1:nBai]...)
    replace!(dist, NaN=>0.)
    
    #cria lista de bairros e populacao
    listaBairros = [
        Bairro(
            length(bairrosResidencias[i]),
            length(bairrosPessoas[i]),
            bairrosResidencias[i],
            bairrosPessoas[i],
            dist[i, :]
            ) for i in 1:nBai
    ]
    
    pop0 = ones(Int, nPessoas)
    pop0[StatsBase.sample(1:nPessoas, I0, replace=false)] .= 2;
    
    if writeDisk
        rodada = length(readdir("saidas")) + 1
        populacao = Populacao(rodada, nPessoas, I0, pop0, copy(pop0), pessoas, listaResidencias, listaBairros, sociais)
        
        # escreve matriz de distancias das pessoas de cada bairro
        escreveDistancias(populacao, rodada, fKernel)
        return populacao
    else
        return Populacao(-1, nPessoas, I0, pop0, copy(pop0), pessoas, listaResidencias, listaBairros, sociais)
    end
end

function geraResidencias(nPessoas::Integer, pesos::AbstractArray{T} where T <: Number)
    n = length(pesos)
    pessoas = Int[]
    while sum(pessoas) < nPessoas
        push!(pessoas, sample(1:n, weights(pesos)))
    end
    pessoas[end] = nPessoas - sum(pessoas[1:(end-1)])
    return vcat([ones(Int, j) * i for (i, j) in enumerate(pessoas)]...);
end

function geraResidencias(nPessoas::Integer, densidade::Number)
    nResidencias = ceil(Int, nPessoas / densidade)
    return rand(1:nResidencias, nPessoas)
end