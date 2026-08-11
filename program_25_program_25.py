# Program 25: Vanilla Policy Gradient - Stock Market Portfolio Management
# Requires: tensorflow
# This is the final stock-market VPG exercise from the lab manual.
import numpy as np
import tensorflow as tf

class VPGAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.02):
        self.policy = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(state_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(action_dim, activation="softmax")
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def select_action(self, state):
        probs = self.policy.predict(np.array([state]), verbose=0)[0]
        return np.random.choice(len(probs), p=probs)

    def train(self, states, actions, rewards):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards, dtype=np.float32)

        with tf.GradientTape() as tape:
            probs = self.policy(states)
            masks = tf.one_hot(actions, probs.shape[-1])
            selected = tf.reduce_sum(probs * masks, axis=1)
            loss = -tf.reduce_sum(
                tf.math.log(selected + 1e-8) * rewards
            )

        gradients = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.policy.trainable_variables)
        )

class StockMarketEnv:
    def __init__(self, prices):
        self.prices = prices
        self.initial_balance = 10000

    def reset(self):
        self.step_index = 0
        self.balance = self.initial_balance
        self.stock_units = 0
        return [self.balance, self.stock_units]

    def step(self, action):
        if self.step_index >= len(self.prices) - 1:
            return [self.balance, self.stock_units], 0, True

        price = self.prices[self.step_index]
        next_price = self.prices[self.step_index + 1]

        if action == 1 and self.balance >= price:
            self.stock_units += 1
            self.balance -= price
        elif action == 0 and self.stock_units > 0:
            self.stock_units -= 1
            self.balance += price

        self.step_index += 1
        portfolio_value = self.balance + self.stock_units * next_price
        reward = portfolio_value - self.initial_balance
        done = self.step_index == len(self.prices) - 1

        return [portfolio_value, self.stock_units], reward, done

prices = np.random.uniform(50, 150, 100)
env = StockMarketEnv(prices)
agent = VPGAgent(2, 2)

for episode in range(100):
    state = env.reset()
    states, actions, rewards = [], [], []
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    agent.train(states, actions, rewards)

    if episode % 10 == 0:
        print(f"Episode {episode + 1}, Total Reward: {sum(rewards):.2f}")
