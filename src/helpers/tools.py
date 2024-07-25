from collections import deque
import math
import os
import pickle
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import requests
import yfinance as yfin
import yfinance as yf
import pandas as pd

from constants import API_KEY, INTERVAL

def get_real_time_data_yfinance(ticker, intervalo='1d', periodo='1mo'):
    """
    Obtém dados históricos da ação especificada.

    :param ticker: O símbolo da ação (e.g., 'AAPL' para Apple).
    :param intervalo: Intervalo dos dados ('1d' para diário, '5m' para 5 minutos).
    :param periodo: Período para o qual obter os dados ('1mo' para 1 mês, '1y' para 1 ano).
    :return: DataFrame com os dados da ação.
    """
    acao = yf.Ticker(ticker)
    
    # Para dados diários
    if intervalo == '1d':
        dados = acao.history(period=periodo)
    
    # Para dados intradiários
    elif intervalo == '5m':
        dados = acao.history(period=periodo, interval=intervalo)
    
    else:
        raise ValueError("Intervalo não suportado. Use '1d' para diário ou '5m' para intradiário.")
    
    return dados

def get_real_time_data_alpha(stock_name):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={stock_name}&interval={INTERVAL}&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    
    # Imprimir os dados retornados para verificar a estrutura
    print("Dados recebidos da API:")
    print(data)
    
    if 'Time Series (5min)' not in data:
        print("Erro ao formatar os dados: A chave 'Time Series (5min)' não foi encontrada nos dados retornados.")
        return None
    
    return data['Time Series (5min)']



def format_data_for_state_creator(data):
    """
    Formata os dados recebidos da API para extrair os preços de fechamento.
    
    Parâmetros:
    - data (dict): Dados recebidos da API.

    Retorna:
    - list: Lista de preços de fechamento.
    """
    try:
        # Verifica se a chave 'Time Series (5min)' está presente
        if 'Time Series (5min)' in data:
            time_series = data['Time Series (5min)']
            # Ordena os dados pela chave do tempo e extrai os preços de fechamento
            sorted_data = sorted(time_series.items())
            closing_prices = [float(item[1]['4. close']) for item in sorted_data]
            return closing_prices
        else:
            raise ValueError("A chave 'Time Series (5min)' não foi encontrada nos dados retornados.")
    except Exception as e:
        print(f"Erro ao formatar os dados: {e}")
        return []

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

def dataset_loader(stock_name, start_date, end_date):
    """
    Carrega os dados do conjunto de dados para um determinado período.

    Parâmetros:
    - stock_name (str): O nome do estoque.
    - start_date (str): A data inicial no formato 'AAAA-MM-DD'.
    - end_date (str): A data final no formato 'AAAA-MM-DD'.

    Retorna:
    - pd.DataFrame: Os dados do conjunto de dados.
    """
    data = yf.download(stock_name, start=start_date, end=end_date)
    return data['Close'].values.tolist()


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

# Função para salvar a memória em um arquivo
def save_memory(memory, file_name="trader_memory.pkl"):
    with open(file_name, "wb") as file:
        pickle.dump(memory, file)

# Função para carregar a memória de um arquivo
def load_memory(file_name="trader_memory.pkl"):
    if os.path.exists(file_name):
        with open(file_name, "rb") as file:
            return pickle.load(file)
    return deque(maxlen=2000)  # Retorna uma deque vazia se o arquivo não existir
