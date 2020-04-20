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