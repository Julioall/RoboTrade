# Definindo os parâmetros do treinamento
import datetime


STOCK_NAME = "CRWD"  # Nome da ação a ser negociada
INITIAL_DATE = "2024-03-01"  # Data inicial para o histórico de preços
FINAL_DATE = datetime.date.today().strftime("%Y-%m-%d")  # Data final para o histórico de preços (hoje)

API_KEY = 'DYESFVPA8W8OWG0Y'
INTERVAL = '1min'