import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import unidecode


def read_data(df, fonte,tipo = 'confirmados'):
    if fonte == 'prefeitura':
        df  = df.drop(columns = ['sexo', 'classificação_final', 'dt_inicio_sintomas'])
        df.rename(mapper = {'faixa_etária' : 'Faixa Etaria', 'bairro_resid__estadia': 'Bairro',
                            'ap_residencia_estadia': 'AP Residência',
                               'dt_notific': 'Data', 'evolução': 'Situação atual'}, axis = 1, inplace = True)
        c = df.columns
        df[[c[1], c[-1]]] = df[[c[-1], c[1]]]
        df[[c[-2], c[-1]]] = df[[c[-1], c[-2]]]
        df[[c[2], c[3]]] = df[[c[3], c[2]]]
        df.rename(mapper = {'Bairro':'Situação atual', 'Situação atual':'Faixa etária',
                            'Faixa Etaria': 'AP Residência', 'AP Residência': 'Bairro'}, axis = 1, inplace = True)
        df = df.drop(labels = len(df)-1, axis = 0)
    if fonte == 'estado':
        if tipo == 'confirmados':
            df = df.drop(columns = ['sexo', 'uf', 'dt_sintoma', 'comorbidades', 'dt_obito'])
            df.rename(mapper = {'idade': 'Idade', 'municipio_res':'Município', 'dt_coleta / dt_notif':'Data',
                                            'evolucao':'Situação atual'}, axis = 1, inplace = True)
            c = df.columns
            df[[c[0],c[2]]] = df[[c[2],c[0]]]
            df[[c[1],c[-1]]] = df[[c[-1],c[1]]]
            df[[c[2],c[-1]]] = df[[c[-1],c[2]]]
            df.rename(mapper = {'Idade':'Data', 'Data':'Município', 'Município':'Situação atual',
                                                                  'Situação atual': 'Idade'}, axis = 1, inplace = True)

            df['Município'] = [unidecode.unidecode(str(v).upper()) for v in df['Município']]
        elif tipo == 'obitos':
            c = df.columns
            df = df.drop(columns = [c[-1], c[-2], 'COMORBIDADE', 'SEXO', 'CONFIRMAÇÃO'])
            df.rename(mapper={'DIVULGAÇÃO':'Data', 'MUNICÍPIO':'Município', 'IDADE':'Idade'}, axis = 1, inplace = True)
            df['Município'] = [unidecode.unidecode(str(v).upper()) for v in df['Município']]
    return df

def get_data(local, df, fonte, T_fim, T_start = '29-03-2020', to_print = True):
    if fonte == 'estado':
        if 'Situação atual' in df.columns:
            df_cidade = df.loc[df['Município'] == local]
            df_cidade = df_cidade.loc[df_cidade['Situação atual'] != 'NAO']
            n = len(df_cidade)
            k = n - len(df_cidade.dropna(subset=['Data']))
            df_cidade = df_cidade.dropna(subset=['Data'])
            if to_print:
                print('No total ' + str(k) + ' dados foram inutilizados')
            l = list(df_cidade['Data'])
            start = dt.datetime.strptime(T_start, "%d-%m-%Y")
            then = dt.datetime.strptime(T_fim, "%d-%m-%Y")
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
        else:
            df_cidade = df.loc[df['Município'] == local]
            n = len(df_cidade)
            df_cidade = df_cidade.dropna(subset=['Data'])
            l = list(df_cidade['Data'])
            start = dt.datetime.strptime(T_start, "%d-%m-%Y")
            then = dt.datetime.strptime(T_fim, "%d-%m-%Y")
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
    elif fonte == 'prefeitura':
        df_bairro = df.loc[df['Bairro'] == local]
        n = len(df_bairro)
        k = n - len(df_bairro.dropna(subset=['Data']))
        df_bairro = df_bairro.dropna(subset=['Data'])
        if to_print:
            print('No total ' + str(k) + ' dados foram inutilizados')
        l = list(df_bairro['Data'])
        start = dt.datetime.strptime(T_start, "%d-%m-%Y")
        then = dt.datetime.strptime(T_fim, "%d-%m-%Y")
        days = mdates.drange(start,then,dt.timedelta(days=1))
        dias = [str(mdates.num2date(v)).split(' ')[0] for v in days]
        dados_por_dia = len(dias)*[0]
        df_bairro_new = pd.to_datetime(df_bairro['Data'], dayfirst=True, format='%d/%m/%Y')
        times =list(df_bairro_new)
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

def download_csv(url = 'http://monitoramento.subpav.rio/COVID19/dados_abertos/Dados_indiv_MRJ_covid19.csv', file_dir = 'data_municipios/dados_prefeitura_rio', file_name = 'Dados_indiv_MRJ_covid19', ext = 'csv', add_date = True ):
    import requests
    req = requests.get(url)
    content = req.content
    if add_date:
        now = dt.datetime.now()
        date = '-'.join([str(now.day), str(now.month), str(now.year)])
        file_name = f'%s/%s_%s.%s' % (file_dir, file_name, date, ext)
    else:
        file_name = f'%s/%s.%s' % (file_dir, file_name, ext)
    file = open(file_name, 'wb')
    file.write(content)
    file.close()
