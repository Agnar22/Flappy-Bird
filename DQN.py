# DQN
import random
import numpy as np
import cv2
import PIL
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Lambda, Flatten
from keras.optimizers import Adam
from collections import deque

import Gamelogic

action_size = 2
batch_size = 32
n_episodes = 5000


class DQNAgent:

    def __init__(self, inp_action_size):
        self.action_size = inp_action_size

        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.01
        self.learning_rate = 0.01

        self.tau = 100
        self.DQNetwork = self._build_model()
        self.TargetNetwork = self._build_model()
        self.sync_networks(0)

        # self.model = self._build_model()
        self.store_list = []

    def _build_model(self, input_shape=(460, 460, 1)):
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape, output_shape=input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu', name='conv1', input_shape=input_shape, padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (6, 6), activation='relu', name='conv2', padding='same', strides=(2, 2)))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(2, activation='linear'))

        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        for layer in model.layers:
            print(layer.input_shape, layer.output_shape)
        return model

    def sync_networks(self, sync_num):
        name = "Models/model_sync_{}".format(sync_num)
        self.save(name, self.DQNetwork)
        self.load(name, self.TargetNetwork)

    def pre_process(self, image):
        image = np.array(image)
        # print('image', image)
        gray = cv2.cvtColor(image[30:490, 0:460], cv2.COLOR_BGR2GRAY)
        gray = gray.reshape(1, 460, 460, 1)
        # gray=gray.reshape(460, 460)
        # PIL.Image.fromarray(gray).show()
        return gray

    def _remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def _store(self, state, action, reward, done):
        self.store_list.append([state, action, reward, done])

    def store_to_remember(self):
        self.store_list[-1][3] = True
        self.store_list[-1][2] = -10
        for x in range(len(self.store_list) - 1):
            elem = self.store_list[x]
            next_elem = self.store_list[x + 1]
            self._remember(elem[0], elem[1], next_elem[2], next_elem[0], next_elem[3])
        self.store_list = []

    def act_store(self, state, reward, done):
        state = self.pre_process(state)
        action = self.act(state)
        self._store(state, action, reward, done)
        return action

    def act(self, state, pre_process=False):
        if pre_process:
            state = self.pre_process(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.TargetNetwork.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # print(state.shape)
                print('pred', self.TargetNetwork.predict(next_state)[0])
                # print('pred max', np.amax(self.model.predict(next_state)[0]))
                # target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
                target = (reward + self.gamma * np.amax(self.TargetNetwork.predict(next_state)[0]))
            print('target', target, self.epsilon)
            target_f = self.DQNetwork.predict(state)
            target_f[0][action] = target

            self.DQNetwork.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name, model):
        model.load_weights(name)

    def save(self, name, model):
        model.save(name)


agent = DQNAgent(action_size)
# agent.load('model_200.h5')

for x in range(n_episodes):
    # Reset game
    # Run game
    game = Gamelogic.Gamelogic(new_window=True, rendering=True)
    game.add_player(human=False, agent_type="DQN", player=agent, render=True)
    game.run_game(1, 1, return_image=True)
    # game.reset_game(delete_players=True)
    agent.store_to_remember()

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    if x % 50 == 0:
        agent.save("Models/model_" + str(x) + '.h5')
