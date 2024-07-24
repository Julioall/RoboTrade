import math
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfin

def sigmoid(x):
    """
    Calcula a função sigmoide.

    Parâmetros:
    - x (float): Valor de entrada para a função sigmoide.

    Retorna:
    - float: Resultado da função sigmoide.
    """
    return 1 / (1 + math.exp(-x))

def stock_price_format(n):
    """
    Formata o preço da ação para um formato monetário.

    Parâmetros:
    - n (float): Preço da ação.

    Retorna:
    - str: Preço da ação formatado como string.
    """
    if n < 0:
        return "- ${0:2f}".format(abs(n))
    else:
        return "${0:2f}".format(abs(n))

def dataset_loader(stock_name, initial_date, final_date):
    """
    Carrega os dados de preços de fechamento de uma ação.

    Parâmetros:
    - stock_name (str): O símbolo do ativo.
    - initial_date (str): A data inicial no formato 'YYYY-MM-DD'.
    - final_date (str): A data final no formato 'YYYY-MM-DD'.

    Retorna:
    - pd.Series: Série contendo os preços de fechamento do ativo.
    """
    yfin.pdr_override()
    dataset = pdr.get_data_yahoo(stock_name, start=initial_date, end=final_date)
    close = dataset['Close']
    return close

def state_creator(data, timestep, window_size):
    """
    Cria o estado baseado nos dados de preços usando uma janela deslizante.

    Parâmetros:
    - data (list ou np.array): Dados de preços.
    - timestep (int): O tempo atual.
    - window_size (int): O tamanho da janela.

    Retorna:
    - np.array: Estado calculado.
    """
    starting_id = timestep - window_size + 1
    if starting_id >= 0:
        windowed_data = data[starting_id: timestep + 1]
    else:
        windowed_data = -starting_id * [data[0]] + list(data[0:timestep + 1])

    state = []
    for i in range(window_size - 1):
        state.append(sigmoid(windowed_data[i + 1] - windowed_data[i]))
    return np.array([state])

def download_stock_data(stock_name, initial_date, final_date):
    """
    Baixa dados de ações do Yahoo Finance.

    Parâmetros:
    - stock_name (str): O símbolo do ativo.
    - initial_date (str): A data inicial no formato 'YYYY-MM-DD'.
    - final_date (str): A data final no formato 'YYYY-MM-DD'.

    Retorna:
    - pd.DataFrame: Um DataFrame contendo os dados do ativo.
    """
    yfin.pdr_override()  # Sobrescreve o pandas datareader com yfinance

    # Baixa os dados do ativo
    stock_data = yfin.download(stock_name, start=initial_date, end=final_date)
    return stock_data
