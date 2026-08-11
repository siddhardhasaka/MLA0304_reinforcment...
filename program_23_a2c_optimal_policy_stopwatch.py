# Program 23: A2C for Optimal Policy Design - Stopwatch
# Requires: tensorflow
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input

class A2CAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.99
        self.actor_critic = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam()

    def build_model(self):
        state = Input(shape=(1,))
        x = Dense(32, activation="relu")(state)
        x = Dense(32, activation="relu")(x)
        policy = Dense(60, activation="softmax")(x)
        value = Dense(1)(x)
        return tf.keras.Model(state, [policy, value])

    def select_action(self, state):
        probs, _ = self.actor_critic.predict(np.asarray(state).reshape(1,1), verbose=0)
        return np.random.choice(60, p=probs[0])

    def train_step(self, state, action, reward):
        state = tf.convert_to_tensor(np.asarray(state).reshape(1,1), dtype=tf.float32)
        action = tf.convert_to_tensor([action])
        reward = tf.convert_to_tensor([[reward]], dtype=tf.float32)

        with tf.GradientTape() as tape:
            probs, value = self.actor_critic(state)
            selected = tf.gather(probs[0], action)
            advantage = reward - value
            actor_loss = -tf.math.log(selected + 1e-8) * advantage
            critic_loss = tf.square(advantage)
            loss = tf.reduce_mean(actor_loss + critic_loss)

        grads = tape.gradient(loss, self.actor_critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor_critic.trainable_variables))

class StopwatchEnv:
    def __init__(self):
        self.time_elapsed = 0

    def reset(self):
        self.time_elapsed = 0
        return [self.time_elapsed]

    def step(self, action):
        self.time_elapsed += action
        done = self.time_elapsed >= 60
        if done:
            self.time_elapsed = 0
        return [self.time_elapsed], 1, done

def train(agent, env, episodes=200, max_steps=200):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.train_step(state, action, reward)
            state = next_state
            total_reward += reward
            if done:
                break

        if episode % 20 == 0:
            print(f"Episode {episode+1}/{episodes}, Total reward: {total_reward}")

if __name__ == "__main__":
    train(A2CAgent(1, 60), StopwatchEnv())
