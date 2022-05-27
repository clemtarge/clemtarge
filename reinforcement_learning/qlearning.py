
import numpy as np
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
import gym
import time


class ReplayBuffer:
    def __init__(self, memory_max_size, input_shape, discrete=False):
        self.memory_max_size: int = memory_max_size
        self.memory_counter: int = 0
        self.input_shape: tuple = input_shape
        self.state_memory: np.array = np.zeros((self.memory_max_size, input_shape))
        self.next_state_memory: np.array = np.zeros((self.memory_max_size, input_shape))
        dtype: type = np.int8 if discrete else np.float32
        self.action_memory: np.array = np.zeros(self.memory_max_size, dtype=dtype)
        self.reward_memory: np.array = np.zeros(self.memory_max_size)
        self.done_memory: np.array = np.zeros(self.memory_max_size, dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.memory_counter % self.memory_max_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.done_memory[index] = 1 - int(done)

        self.action_memory[index] = action
        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.memory_counter, self.memory_max_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        dones = self.done_memory[batch]

        return states, actions, rewards, next_states, dones


class Agent:
    def __init__(self, input_dims, batch_size, memory_max_size, n_actions, name="agent"):
        self.input_dims: tuple = input_dims
        self.action_space: list = list(range(n_actions))  # [0, 1, ..., n_actions-1]
        self.n_actions: int = n_actions

        self.epsilon: float = 1.0
        self.epsilon_min: float = 0.01
        self.epsilon_decay: float = 0.96
        self.gamma: float = 0.95
        self.batch_size: int = batch_size
        self.name = name
        self.fname_dqn: str = name + "_dqn_weights.h5"
        self.fname_memory: str = name + "_memory.pkl"

        self.memory: ReplayBuffer = ReplayBuffer(memory_max_size=memory_max_size,
                                                 input_shape=input_dims,
                                                 discrete=True)

        self.TrainingNet: Sequential = self.build_dqn()
        self.TargetNet: Sequential = self.build_dqn()
        self.copy_weights()

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def update_epsilon(self):
        self.epsilon = self.epsilon*self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

    def save_model(self):
        self.TrainingNet.save(self.fname_dqn)

    def load_model(self):
        self.TrainingNet = load_model(self.fname_dqn)

    def save_memory(self):
        pickle.dump(self.memory, open(self.fname_memory, "wb"))

    def load_memory(self):
        self.memory = pickle.load(self.fname_memory)

    def copy_weights(self):
        self.TargetNet.set_weights(self.TrainingNet.get_weights())

    def build_dqn(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(self.input_dims, )))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))

        model.compile(loss='mse', optimizer=SGD(learning_rate=5e-3))
        return model

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.TrainingNet.predict(np.atleast_2d(state))
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)

        q_eval = self.TrainingNet.predict(states)
        q_next = self.TargetNet.predict(next_states)

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, actions] = rewards + self.gamma*np.max(q_next, axis=1)*dones

        self.TrainingNet.fit(states, q_target, verbose=0, epochs=5, batch_size=self.batch_size)

        self.update_epsilon()


def run_episode(agent: Agent, env: gym.Env, remember: bool, render=False, sleep=0.01):

    done = False
    score = 0
    state = env.reset()

    while not done:
        if render:
            env.render()
            time.sleep(sleep)

        action = agent.get_action(state)

        next_state, reward, done, _ = env.step(action)
        score += reward

        if remember: agent.remember(state, action, reward, next_state, done)
        state = next_state

    return score
