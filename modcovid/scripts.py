import pandas as pd
import yaml
import git
import sys
from modcovid import settings

root_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.insert(1, root_dir)

settings.init()

with open(settings.CONFIG_FILE, encoding = 'utf-8') as f:
    configs = yaml.load(f, Loader = yaml.FullLoader)

################ set_df ################

def set_df(fonte, *args):
    """Carrega o CSV com dados do covid para determinada cidade ou estado e retorna um pandas.DataFrame com os dados

    Parameters
    ----------
    fonte : str ('prefeitura_rj', 'estado_rj')
        Indica qual a fonte dos dados, com isso a função determina qual CSV carregar e como tratar o DataFrame.
        O arquivo a ser carregado é dado em configs.yml
    *args : dict, optional
        Um dicionário com argumentos extras:
            df_break: Se setado para True, muda o retorno para vários DataFrames
    Returns
    -------
    ret_v: Se nenhum argumento opcional for passado, ret_v é uma lista contendo o DataFrame tratado com todos os casos, e a data de atualização dos dados
            df_break == True: ret_v é uma lista com [DataFrame Tratado, [DataFrame Ativos, DataFrame Recuperados, DataFrame Obitos], Data de Atualização]        
    """

    if fonte == 'prefeitura_rj':
        df = pd.read_csv(root_dir + '/' + configs['csv']['rj']['file_loc']['prefeitura'], encoding = 'utf-8', delimiter = ';')
        df.rename(columns = configs['df']['rename']['rj']['colunas']['prefeitura'], inplace = True)
        dt_att = df['Data_atualização'].values[0]
        for drop in configs['df']['droppable']['rj']['prefeitura']:
            df.drop(drop, axis = 1, inplace = True)
        for r in configs['df']['rename']['rj']['dados']['prefeitura']:
            df[r].replace(configs['df']['rename']['rj']['dados']['prefeitura'][r], inplace = True)
        if args and args[0]['df_break'] == True:
            df_break = []
            for s in configs['df']['status']['rj']['prefeitura']:
                df_break.append(df[df['Evolucao'] == s])
            ret_lst = [df, df_break, dt_att]
        else:
            ret_lst = [df, dt_att]
    elif fonte == 'estado_rj':
        df = pd.read_csv(root_dir + '/' + configs['csv']['rj']['file_loc']['estado'])
        df = df[df['classificacao'] == 'CONFIRMADO']
        df.rename(columns = configs['df']['rename']['rj']['colunas']['estado'], inplace = True)
        for drop in configs['df']['droppable']['rj']['estado']:
            df.drop(drop, axis = 1, inplace = True)
        for r in configs['df']['rename']['rj']['dados']['estado']:
            df[r].replace(configs['df']['rename']['rj']['dados']['estado'][r], inplace = True)
        if args and args[0]['df_break'] == True:
            df_break = []
            for s in configs['df']['status']['rj']['estado']:
                df_break.append(df[df['Evolucao'] == s])
            ret_lst = [df, df_break]
        else:
            ret_lst = df
    else: 
        print('Fonte de dados Inválida, as opções são prefeitura_rj ou estado_rj')
        ret_lst = 'Error'
    return ret_lst

################ gen_timeseries ################

def get_timeseries(df, fonte, localidade, T_fim, T_start = '06/03/2020', *args):
    """Dado um DataFrame organizado por set_df, esta função extrai um timeseries dos dados

    Parameters
    ----------
    df : pandas.DataFrame a ser trabalhado
    fonte : str ('prefeitura_rj', 'estado_rj')
        Indica qual a fonte dos dados

    *args : dict, optional
        Um dicionário com argumentos extras:
            df_break: Se setado para True, muda o retorno para vários DataFrames
    Returns
    -------
    ret_v: Se nenhum argumento opcional for passado, ret_v é uma lista contendo o DataFrame tratado com todos os casos, e a data de atualização dos dados
            df_break == True: ret_v é uma lista com [DataFrame Tratado, [DataFrame Ativos, DataFrame Recuperados, DataFrame Obitos], Data de Atualização]        
    """
    print('a')
    return ret_lst