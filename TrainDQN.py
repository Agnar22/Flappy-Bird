import DQN
import Gamelogic
import numpy as np
import cv2


class Data:
    def __init__(self, memory_size=2048):
        self.states = None
        self.rewards = np.array([])
        self.actions = np.array([])
        self.finished_episodes = np.array([])
        self.probs = np.array([])
        self.next_memory = 0
        self.max_memory = memory_size
        self.alpha = 0.7

    def add_element(self, state, prob, action, finished, reward):
        if self.max_memory > self.actions.shape[0]:
            if self.states is None:
                self.states = state
            else:
                self.states = np.vstack((self.states, state))
            self.rewards = np.append(self.rewards, reward)
            self.probs = np.append(self.probs, prob ** self.alpha)
            self.actions = np.append(self.actions, action)
            self.finished_episodes = np.append(self.finished_episodes, finished)

        else:
            self.states[self.next_memory] = state
            self.rewards[self.next_memory] = reward
            self.probs[self.next_memory] = prob ** self.alpha
            self.actions[self.next_memory] = action
            self.finished_episodes[self.next_memory] = finished

        self.next_memory = (self.next_memory + 1) % self.max_memory

    # Returning data randomly sampled proportionally to self.probs.
    # Exploits the fact that the next state is in order and the next state to the last state is never used.
    def get_random_data(self, number):
        chosen = np.random.choice(np.arange(self.actions.shape[0]), size=number, replace=True,
                                  p=self.probs / self.probs.sum())
        next = (chosen + 1) % min(self.max_memory, self.actions.shape[0])

        chosen_states = [self.states[chosen[0]]]
        chosen_next = [self.states[next[0]]]
        for i in range(1, chosen.shape[0]):
            chosen_states.append(self.states[chosen[i]])
            chosen_next.append(self.states[next[i]])
        chosen_states = np.array(chosen_states)
        chosen_next = np.array(chosen_next)

        return [chosen_states, np.take(self.actions, chosen), np.take(self.rewards, chosen), chosen_next, \
               np.take(self.finished_episodes, chosen)]

    def update_priority(self, num, pri):
        self.probs[num] = pri ** self.alpha


class DQNWrapper:
    def __init__(self, agent, data_obj):
        self.default_priority = 0.0001

        self.data_obj = data_obj
        self.agent = agent
        self._init_lists()

    def _init_lists(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.finished_episodes = []

    @staticmethod
    def preprocess(image):
        image = np.array(image)
        # print('image', image)
        gray = cv2.cvtColor(image[30:490, 0:460], cv2.COLOR_BGR2GRAY)
        gray = gray.reshape(1, 460, 460, 1)
        # gray1=gray.reshape(460, 460)
        # PIL.Image.fromarray(gray1).show()
        return gray

    # Pre-processing the state, calculating the action, stores the action, state and reward, returns the action.
    def act_store(self, state, reward, done):
        prep_state = DQNWrapper.preprocess(state)
        self.states.append(prep_state)
        action = agent.act(prep_state)
        self.actions.append(action)
        self.add_rewards(reward, done)
        return action

    def add_rewards(self, reward, finished):
        self.rewards.append(reward)
        self.finished_episodes.append(finished)

    # Calculating priorities and stores the sequences in data
    def store_and_reset(self):
        """assuming that it is dead in the last state"""
        for i in range(len(self.states)):
            self.data_obj.add_element(self.states[i], abs(self.rewards[i]) + self.default_priority,
                                      self.actions[i], self.finished_episodes[i] or i == len(self.states) - 1, 1)
        self._init_lists()


action_size = 2
batch_size = 32
epoch_size = 32 * 8
n_episodes = 5000

agent = DQN.DQNAgent(action_size)
data_store = Data()
agent_wrapper = DQNWrapper(agent, data_store)
# agent.load('model_200.h5')
game = Gamelogic.Gamelogic(new_window=True, rendering=True)
max_fitness = 0

for x in range(n_episodes):
    # Reset game
    game.reset_game(delete_players=True)
    # Run game
    game.add_player(human=False, agent_type="DQN", player=agent_wrapper, render=True)
    max_fitness = max(game.run_game(1, 1, return_image=True), max_fitness)
    # game.reset_game(delete_players=True)
    # agent.store_to_remember()
    agent_wrapper.store_and_reset()

    if data_store.states.shape[0] > batch_size:
        data = data_store.get_random_data(epoch_size)
        agent.replay(batch_size, data)

    if x % 20 == 0:
        agent.save("Models/model_" + str(x) + '_' + str(max_fitness) + '.h5', agent.DQNetwork)
        max_fitness = 0
