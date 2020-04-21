import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates


def read_data(df, tipo):
    if tipo == 'confirmados':
        df  = df.drop(columns = ['SEXO', 'IDADE '])
        df.rename(mapper = {'FX_ETARIA' : 'Faixa Etaria', 'MUNIC_RESIDENCIA': 'Municipio', 'BAIRRO_RESIDENCIA': 'Bairro',
                               'DT_NOT': 'Data'}, axis = 1, inplace = True)
        c = df.columns
        df[[c[2], c[3]]] = df[[c[3], c[2]]]
        df.rename(mapper = {'Bairro':'Data', 'Data':'Bairro'}, axis = 1, inplace = True)
        df = df.drop(labels = len(df)-1, axis = 0)
        return df
    elif tipo == 'obitos':
        df = df.drop(columns = ['SEXO', 'IDADE'])
        df.rename(mapper = {'FXETARIA':'Faixa Etaria', 'MUN_RES': 'Municipio', 'DT_OBITO':'Data'}, axis = 1, inplace = True)
        return df
    else:
        print('tipos são confirmados ou obitos')


def get_data(cidade, df, T_fim, T_start = '29-03-2020'):
    df_cidade = df.loc[df['Municipio'] == cidade]
    n = len(df_cidade)
    df = df_cidade
    k = n - len(df_cidade.dropna(subset=['Data']))
    df_cidade = df_cidade.dropna(subset=['Data'])
    print('No total ' + str(k) + ' dados foram inutilizados')
    l = list(df_cidade['Data'])
    start = dt.datetime.strptime(T_start, "%d-%m-%Y")
    then = start + dt.timedelta(days=T_fim)
    days = mdates.drange(start,then,dt.timedelta(days=1))
    dias = [str(mdates.num2date(v)).split(' ')[0] for v in days]
    dados_por_dia = len(dias)*[0]
    df_cidade_new = pd.to_datetime(df_cidade['Data'], dayfirst=True, format='%d/%m/%Y')
    times =list(df_cidade_new)
    times = [str(time).split(' ')[0] for time in times]
    for i, dia in enumerate(dias):
        dados_por_dia[i] += times.count(dia)
    dados = [sum(dados_por_dia[:i]) for i in range(1, len(dados_por_dia)+1)]
    acumulado_inicial = n - dados[-1]
    dados = [acumulado_inicial + dado for dado in dados]
    return dados, dados_por_dia

def set_df(df, dt_start, dt_fim, municipios = 'all', skip = False, header = ['Data', 'Municipio', 'Casos']):
    if municipios == 'all':
        municipios = set(df['Municipio'])
    else:
        if skip == True:
            municipios = set(df[~df['Municipio'].str.contains('|'.join(municipios))]['Municipio'])
        else:
            municipios = set(df[df['Municipio'].str.contains('|'.join(municipios))]['Municipio'])

    ## Configurando os dias
    dt_start = dt.datetime.strptime(dt_start, '%d-%m-%Y')
    dt_fim = dt.datetime.strptime(dt_fim, '%d-%m-%Y')
    days = mdates.drange(dt_start, dt_fim, dt.timedelta(days = 1))

    dt_start = dt.datetime.strftime(dt_start, '%d-%m-%Y')
    dt_fim = dt.datetime.strftime(dt_fim, '%d-%m-%Y')
    dates = [dt.datetime.strftime(mdates.num2date(i), '%d-%m-%Y').replace('-','/') for i in days]

    ## Extraindo os dados e organizando em listas para o Dicionario
    casos = []
    nome_m = []
    data_m = []
    for m in municipios:
        lst = []
        lst_m = [m]*len(dates)
        lst_d = []
        for d in dates:
            df = df[df['Municipio'] == m]
            lst.append(len(df[df['Data'] == d]))
            lst_d.append(d)
        nome_m.append(lst_m)
        casos.append(lst)
        data_m.append(lst_d)

    ## Criando o Dicionario
    lst = [[],[],[]]
    for c, m, d in zip(casos, nome_m, data_m):
        lst[0] += d
        lst[1] += m
        lst[2] += c

    dic = {}
    for i,h in enumerate(header):
        dic.update({h: lst[i]})

    ## Transformando o dicionario em DataFrame
    df = pd.DataFrame(dic)

    return df
