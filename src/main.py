import datetime

from helpers.tools import dataset_loader
from models.Trade import Trader



STOCK_NAME = "WEGE3.SA"
INITIAL_DATE = "2021-01-01"
FINAL_DATE = datetime.date.today().strftime("%Y-%m-%d")

data = dataset_loader(STOCK_NAME, INITIAL_DATE, FINAL_DATE)

# Training the Q-Learning Trading Agent
window_size = 10
episodes = 2
batch_size = 32
data_samples = len(data) - 1

trader = Trader(window_size)
trader.model.summary()

# Defining a Training Loop
for episode in range(1, episodes + 1):
    print("Episode: {}/{}".format(episode, episodes))
    state = state_creator(data, 0, window_size + 1)
    total_profit = 0
    trader.inventory = []

    for t in tqdm(range(data_samples)):
        action = trader.trade(state)
        next_state = state_creator(data, t + 1, window_size + 1)
        reward = 0

        if action == 0:
            print(" - Sem ação | Total de papeis no portfolio = ", len(trader.inventory))
        elif action == 1:
            trader.inventory.append(data[t])
            print(" - AI Trader Comprou: ", stock_price_format(data[t]))
        elif action == 2 and len(trader.inventory) > 0:
            buy_price = trader.inventory.pop(0)
            reward = max(data[t] - buy_price, 0)
            total_profit += data[t] - buy_price
            print(" - AI Trader Vendeu: ", stock_price_format(data[t]), " - Lucro: " + stock_price_format(data[t] - buy_price))

        done = t == data_samples - 1
        trader.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("########################")
            print("TOTAL PROFIT: {}".format(total_profit))
            print("########################")

        if len(trader.memory) > batch_size:
            trader.batch_train(batch_size)

    if episode % 10 == 0:
        trader.model.save("ai_trader_{}.h5".format(episode))