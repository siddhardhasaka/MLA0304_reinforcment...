# Program 21: DDPG for Optimal Policy - Bus Simulation Analogy
# Requires: tensorflow, gymnasium
import tensorflow as tf
import numpy as np
import gymnasium as gym
from collections import deque
import random

class Actor(tf.keras.Model):
    def __init__(self, action_dim, max_action):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(400, activation="relu")
        self.d2 = tf.keras.layers.Dense(300, activation="relu")
        self.out = tf.keras.layers.Dense(action_dim, activation="tanh")
        self.max_action = max_action

    def call(self, state):
        return self.out(self.d2(self.d1(state))) * self.max_action

class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(400, activation="relu")
        self.d2 = tf.keras.layers.Dense(300, activation="relu")
        self.out = tf.keras.layers.Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        return self.out(self.d2(self.d1(x)))

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(action_dim, max_action)
        self.target_actor = Actor(action_dim, max_action)
        self.critic = Critic()
        self.target_critic = Critic()

        self.actor_optimizer = tf.keras.optimizers.Adam(0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(0.002)
        self.memory = deque(maxlen=100000)

        self.batch_size = 64
        self.discount = 0.99
        self.tau = 0.001

        dummy_s = tf.zeros((1, state_dim))
        dummy_a = tf.zeros((1, action_dim))
        self.actor(dummy_s)
        self.target_actor(dummy_s)
        self.critic(dummy_s, dummy_a)
        self.target_critic(dummy_s, dummy_a)

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def select_action(self, state):
        action = self.actor(np.expand_dims(state, 0).astype(np.float32))
        return np.squeeze(action.numpy(), axis=0)

    def remember(self, *experience):
        self.memory.append(experience)

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(
            lambda x: np.asarray(x, dtype=np.float32), zip(*batch)
        )

        target_actions = self.target_actor(next_states)
        target_q = self.target_critic(next_states, target_actions)
        target_q = rewards[:, None] + self.discount * target_q * (1 - dones[:, None])

        with tf.GradientTape() as tape:
            q = self.critic(states, actions)
            critic_loss = tf.reduce_mean(tf.square(target_q - q))
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actor_actions = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic(states, actor_actions))
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        for target, source in zip(self.target_critic.variables, self.critic.variables):
            target.assign(self.tau * source + (1 - self.tau) * target)
        for target, source in zip(self.target_actor.variables, self.actor.variables):
            target.assign(self.tau * source + (1 - self.tau) * target)

        return float(actor_loss), float(critic_loss)

def train_ddpg_agent():
    env = gym.make("Pendulum-v1")
    agent = DDPGAgent(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        env.action_space.high[0]
    )

    for episode in range(20):
        state, _ = env.reset()
        total_reward = 0
        done = False
        actor_loss = critic_loss = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            actor_loss, critic_loss = agent.train()
            total_reward += reward
            state = next_state

        print(f"Episode {episode+1}, Reward {total_reward:.2f}, Actor Loss {actor_loss:.4f}, Critic Loss {critic_loss:.4f}")

    env.close()

if __name__ == "__main__":
    train_ddpg_agent()
