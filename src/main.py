from asyncio import constants
from datetime import time
import os
import tensorflow as tf
import tqdm
from constants import FINAL_DATE, INITIAL_DATE, STOCK_NAME
from helpers.tools import dataset_loader, get_real_time_data, get_real_time_data_yfinance, load_memory, save_memory, state_creator, stock_price_format
from models.trade import Trader

# Carregando o conjunto de dados com preços históricos da ação
data = dataset_loader(STOCK_NAME, INITIAL_DATE, FINAL_DATE)

# Parâmetros do agente de treinamento
window_size = 10  # Tamanho da janela de observação
episodes = 1  # Número de episódios de treinamento
batch_size = 32  # Tamanho do lote para treinamento
data_samples = len(data) - 1  # Número de amostras de dados

# Inicializando o agente trader e carregando a memória
trader = Trader(window_size)
trader.memory = load_memory()
trader.model.summary()  # Exibindo um resumo da arquitetura do modelo

# Função para enviar sinais de negociação
def send_signal(action, price):
    print(f'Signal: {action} at price {price}')

# Função de treino do agente
def treiner_agent():
    # Loop de treinamento
    for episode in range(1, episodes + 1):
        print("Episode: {}/{}".format(episode, episodes))
        
        # Criando o estado inicial
        state = state_creator(data, 0, window_size + 1)
        total_profit = 0  # Inicializando o lucro total
        trader.inventory = []  # Reiniciando o inventário do trader

        # Loop através de cada ponto de tempo nos dados
        for t in tqdm.tqdm(range(data_samples)):
            # Decidindo a ação a ser tomada
            action = trader.trade(state)
            
            # Criando o próximo estado
            next_state = state_creator(data, t + 1, window_size + 1)
            reward = 0  # Inicializando a recompensa

            if action == 0:
                # Ação 0: Não fazer nada
                print(" - Sem ação | Total de papeis no portfolio = ", len(trader.inventory))
            elif action == 1:
                # Ação 1: Comprar
                trader.inventory.append(data[t])
                print(" - AI Trader Comprou: ", stock_price_format(data[t]))
            elif action == 2 and len(trader.inventory) > 0:
                # Ação 2: Vender (se houver ações no inventário)
                buy_price = trader.inventory.pop(0)
                reward = max(data[t] - buy_price, 0)  # Calculando a recompensa
                total_profit += data[t] - buy_price  # Atualizando o lucro total
                print(" - AI Trader Vendeu: ", stock_price_format(data[t]), " - Lucro: " + stock_price_format(data[t] - buy_price))

            done = t == data_samples - 1  # Verificando se é o último ponto de tempo
            # Armazenando a experiência na memória do agente
            trader.memory.append((state, action, reward, next_state, done))
            state = next_state  # Atualizando o estado

            if done:
                # Imprimindo o lucro total ao final do episódio
                print("########################")
                print("TOTAL PROFIT: {}".format(total_profit))
                print("########################")

            if len(trader.memory) > batch_size:
                # Treinando o agente se houver memória suficiente
                trader.batch_train(batch_size)

        if episode % 10 == 0:
            # Salvando o modelo a cada 10 episódios
            trader.model.save("ai_trader_{}.h5".format(episode))

        # Salvando a memória ao final de cada episódio
        save_memory(trader.memory)
        print("Treinamento concluído.")

def monitorar_mercado_em_tempo_real(stock_name, window_size, update_interval=60):
    """
    Monitora o mercado em tempo real usando o modelo treinado para tomar decisões de compra/venda.
    Treina um novo modelo se não houver um modelo salvo.

    Parâmetros:
    - stock_name (str): O símbolo do ativo.
    - window_size (int): O tamanho da janela de observação para o estado.
    - update_interval (int): O intervalo de atualização em segundos para obter dados em tempo real.

    Retorna:
    - None
    """
    # Inicializa o trader
    trader = Trader(state_size=window_size)

    # Verifica se há um modelo salvo
    model_file = "ai_trader_latest.keras"
    if os.path.exists(model_file):
        trader.model = tf.keras.models.load_model(model_file)
        print("Modelo carregado com sucesso.")
    else:
        print("Modelo não encontrado. Treinando um novo modelo...")
        # Carregar a memória do trader
        trader.memory = load_memory()
        
        # Treinar o modelo com a memória carregada
        batch_size = 32
        episodes = 1  # Ajuste conforme necessário
        for episode in range(episodes):
            if len(trader.memory) > batch_size:
                trader.batch_train(batch_size)
            if episode % 10 == 0:
                trader.model.save(model_file)
        
        print("Novo modelo treinado e salvo.")

    # Loop de monitoramento em tempo real
    while True:
        try:
            # Obter dados em tempo real
            real_time_data = get_real_time_data_yfinance(stock_name)
            
            # Garantir que há dados suficientes para criar o estado
            if len(real_time_data) >= window_size:
                # Criar o estado atual
                state = state_creator(real_time_data, len(real_time_data) - 1, window_size)
                
                # Decidir a ação com base no estado atual
                action = trader.trade(state)
                
                # Obter o preço atual
                current_price = real_time_data[-1]

                # Executar a ação
                if action == 0:
                    # Ação 0: Não fazer nada
                    print(" - Nenhuma ação | Preço atual: ", stock_price_format(current_price))
                elif action == 1:
                    # Ação 1: Comprar
                    trader.inventory.append(current_price)
                    print(" - Comprado: ", stock_price_format(current_price))
                elif action == 2 and len(trader.inventory) > 0:
                    # Ação 2: Vender (se houver ações no inventário)
                    buy_price = trader.inventory.pop(0)
                    reward = max(current_price - buy_price, 0)
                    print(" - Vendido: ", stock_price_format(current_price), " - Lucro: ", stock_price_format(current_price - buy_price))
                
                # Adicionar a experiência à memória
                if len(trader.memory) > 0:
                    # Considerar o último estado e a última ação
                    last_state, last_action, last_reward, _, _ = trader.memory[-1]
                    next_state = state
                    done = False
                    trader.memory.append((last_state, last_action, last_reward, next_state, done))
                
            # Esperar pelo próximo intervalo de atualização
            time.sleep(update_interval)

        except Exception as e:
            print(f"Erro ao monitorar o mercado: {e}")

# Chamada da função para monitorar o mercado em tempo real
monitorar_mercado_em_tempo_real(STOCK_NAME, window_size)

