# Program 20: Enhancing DQN into DDQN
# Requires: tensorflow, gymnasium
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import gymnasium as gym

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return map(np.array, (states, actions, rewards, next_states, dones))

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size, self.action_size = state_size, action_size
        self.target_update_frequency = 100
        self.dqn = self.build_model()
        self.target_dqn = self.build_model()
        self.target_dqn.set_weights(self.dqn.get_weights())
        self.replay_buffer = ReplayBuffer(2000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.total_steps = 0

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(24, activation="relu"),
            keras.layers.Dense(24, activation="relu"),
            keras.layers.Dense(self.action_size, activation="linear")
        ])
        model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
        return model

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q = self.dqn.predict(np.expand_dims(state, 0), verbose=0)
        return int(np.argmax(q[0]))

    def remember(self, *experience):
        self.replay_buffer.add(experience)

    def train(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        targets = self.dqn.predict(states, verbose=0)
        next_online = self.dqn.predict(next_states, verbose=0)
        next_target = self.target_dqn.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                best_action = np.argmax(next_online[i])
                targets[i, actions[i]] = rewards[i] + self.gamma * next_target[i, best_action]

        self.dqn.fit(states, targets, epochs=1, verbose=0)

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.total_steps += 1

        if self.total_steps % self.target_update_frequency == 0:
            self.target_dqn.set_weights(self.dqn.get_weights())

    def save(self, filename):
        self.dqn.save_weights(filename)

def train_ddqn_agent():
    env = gym.make("CartPole-v1")
    agent = DDQNAgent(env.observation_space.shape[0], env.action_space.n)

    for episode in range(50):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.train()

        if episode % 10 == 0:
            print(f"Episode {episode}, Steps {agent.total_steps}, Epsilon {agent.epsilon:.2f}")

    agent.save("ddqn_model.weights.h5")
    env.close()

if __name__ == "__main__":
    train_ddqn_agent()
