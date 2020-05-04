import pandas as pd
import yaml
from modcovid import settings

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
        df = pd.read_csv(configs['csv']['rj']['file_loc']['prefeitura'], encoding = 'iso-8859-1', delimiter = ';')
        df.rename(columns = configs['df']['rename']['rj']['prefeitura'], inplace = True)
        dt_att = df['Data_atualização'].values[0]
        df.drop('Data_atualização', axis = 1, inplace = True)
        lst_v = [df, dt_att]
        if args and args[0]['df_break'] == True:
            df_break = []
            for s in configs['df']['status']['rj']['prefeitura']:
                df_break.append(df[df['Evolucao'] == s])
            lst_v = [df, df_break, dt_att]
    return lst_v