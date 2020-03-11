# DQN
import random
import numpy as np
import cv2
import PIL
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Lambda, Flatten, Input, average, Add
from keras.backend import repeat_elements
from keras.optimizers import Adam
import keras
import tensorflow as tf
from collections import deque


class DQNAgent:

    def __init__(self, inp_action_size):
        self.action_size = inp_action_size
        self.name="DDDQN"

        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 0.6
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.01
        self.learning_rate = 0.0005

        self.tau = 200
        self.tau_counter = 0
        self.DQNetwork = self._build_model(self.name+"-l")
        self.TargetNetwork = self._build_model(self.name+"-t")
        self.sync_networks(0)

        # self.model = self._build_model()
        self.store_list = []

    def _build_model(self,name,  input_shape=(460, 460, 1)):
        h, w, d=input_shape
        # Constructing a Duelling Deep Q-Network
        # model = Sequential()
        # model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape, output_shape=input_shape))
        # model.add(Conv2D(32, (3, 3), activation='relu', name='conv1', input_shape=input_shape, padding='same'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Conv2D(64, (4, 4), activation='relu', name='conv2', padding='same', strides=(2, 2)))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Flatten())
        # model.add(Dense(256, activation='relu'))
        # model.add(Dense(2, activation='linear'))
        # inputs = Input(shape=input_shape)
        # x = Lambda(lambda x: x / 127.5 - 1, input_shape=input_shape, output_shape=input_shape)(inputs)
        # x = Conv2D(32, (3, 3), activation='relu', name='conv1', padding='same', input_shape=input_shape)(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Conv2D(64, (4, 4), activation='relu', name='conv2', padding='same')(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Flatten()(x)
        #
        # value = Dense(1, activation='relu')(x)
        # action_advantage = Dense(self.action_size, activation='relu')(x)
        # added=keras.layers.add([action_advantage, action_advantage])
        # # averaged=keras.layers.average([action_advantage, action_advantage])
        # # multiplied=repeat_elements(value, self.action_size, 0)
        # average_action_advantage = keras.backend.mean(action_advantage, axis=[-1], keepdims=True)
        # keras.layers.average(action_advantage)
        # # k=repeat_elements(value, self.action_size, -1)
        # # Q_values = Add()([repeat_elements(value, self.action_size, -1), average_action_advantage])
        #
        # model = Model(inputs, average_action_advantage)
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        #
        # model.summary()
        #
        # return model
        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 100, 120, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, h,w,d], name="inputs")

            #
            self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convnet:
            CNN
            ELU
            """
            # Input is 100x120x4
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")

            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            """
            Second convnet:
            CNN
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            """
            Third convnet:
            CNN
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=128,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.layers.flatten(self.conv3_out)

            ## Here we separate into two streams
            # The one that calculate V(s)
            self.value_fc = tf.layers.dense(inputs=self.flatten,
                                            units=512,
                                            activation=tf.nn.elu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name="value_fc")

            self.value = tf.layers.dense(inputs=self.value_fc,
                                         units=1,
                                         activation=None,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name="value")

            # The one that calculate A(s,a)
            self.advantage_fc = tf.layers.dense(inputs=self.flatten,
                                                units=512,
                                                activation=tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name="advantage_fc")

            self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                             units=self.action_size,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="advantages")

            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + tf.subtract(self.advantage,
                                                   tf.reduce_mean(self.advantage, axis=1, keepdims=True))

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # The loss is modified because of PER
            self.absolute_errors = tf.abs(self.target_Q - self.Q)  # for updating Sumtree

            self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            return tf.get_default_graph()

    # The weights from the target network are updated to the DQN-weights
    def sync_networks(self, sync_num):
        name = "Models/model_sync_{}.h5".format(sync_num)
        self.save(name, self.DQNetwork)
        self.load(name, self.TargetNetwork)

    def pre_process(self, image):
        image = np.array(image)
        # print('image', image)
        gray = cv2.cvtColor(image[30:490, 0:460], cv2.COLOR_BGR2GRAY)
        gray = gray.reshape(1, 460, 460, 1)
        # gray1=gray.reshape(460, 460)
        # PIL.Image.fromarray(gray1).show()
        return gray

    def _remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def _store(self, state, action, reward, done):
        self.store_list.append([state, action, reward, done])

    # def store_to_remember(self):
    #     self.store_list[-1][3] = True
    #     self.store_list[-1][2] = -10
    #     for x in range(len(self.store_list) - 1):
    #         elem = self.store_list[x]
    #         next_elem = self.store_list[x + 1]
    #         self._remember(elem[0], elem[1], next_elem[2], next_elem[0], next_elem[3])
    #     self.store_list = []

    # def act_store(self, state, reward, done):
    #     state = self.pre_process(state)
    #     action = self.act(state)
    #     self._store(state, action, reward, done)
    #     return action

    def predict(self, state):
        return self.TargetNetwork.predict(state)

    def act(self, state, pre_process=False):
        if pre_process:
            state = self.pre_process(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.TargetNetwork.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size, minibatch):
        # minibatch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = minibatch

        for state, action, reward, next_state, done in minibatch:

            self.tau_counter += 1
            # Updating target network - part of fixed Q-targets
            if self.tau_counter == self.tau:
                self.tau_counter = 0
                self.sync_networks(random.randint(0, 100000000))

            target = reward
            if not done:
                print('pred', self.TargetNetwork.predict(next_state))

                # Double learning
                target_act = np.argmax(self.DQNetwork.predict(next_state))
                target = reward + self.gamma * self.TargetNetwork.predict(next_state)[0][target_act]
            print('target', target, self.epsilon)
            target_f = self.DQNetwork.predict(state)
            target_f[0][action] = target

            train_history = self.DQNetwork.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name, model):
        model.load_weights(name)

    def save(self, name, model):
        with tf.Session() as sess:
            sess.run(model)
            tf.train.Saver().save(sess, name)
