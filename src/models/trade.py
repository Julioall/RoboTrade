import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow import keras
from tensorflow.keras import layers

class Trader:
    def __init__(self, state_size, action_space=3, model_name="AITrader"):
        """
        Inicializa o trader de deep Q-learning.

        Parâmetros:
        - state_size (int): O tamanho do estado.
        - action_space (int): O espaço de ações possíveis. Padrão é 3.
        - model_name (str): O nome do modelo. Padrão é "AITrader".
        """
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.inventory = []
        self.model_name = model_name

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995

        self.model = self.model_builder()

    def model_builder(self):
        """
        Constrói o modelo de rede neural para o trader.

        Retorna:
        - tf.keras.Model: O modelo de rede neural compilado.
        """
        model = tf.keras.models.Sequential()
        model.add(layers.Dense(units=32, activation='relu', input_dim=self.state_size))
        model.add(layers.Dense(units=64, activation='relu'))
        model.add(layers.Dense(units=128, activation='relu'))
        model.add(layers.Dense(units=self.action_space, activation='linear'))

        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
        return model

    def trade(self, state):
        """
        Decide a ação a ser tomada pelo trader com base no estado atual.

        Parâmetros:
        - state (np.array): O estado atual.

        Retorna:
        - int: A ação selecionada.
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)

        actions = self.model.predict(state)
        return np.argmax(actions[0])

    def batch_train(self, batch_size):
        """
        Treina o modelo com um lote de experiências armazenadas.

        Parâmetros:
        - batch_size (int): O tamanho do lote para treinamento.
        """
        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            reward = reward
            if not done:
                reward = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target = self.model.predict(state)
            target[0][action] = reward

            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
