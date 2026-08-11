# Program 24: Vanilla Policy Gradient for Stock Market Portfolio Management
# Requires: tensorflow
import numpy as np
import tensorflow as tf

class VPGAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.02):
        self.policy_network = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(state_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(action_dim, activation="softmax")
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def select_action(self, state):
        probs = self.policy_network.predict(np.asarray([state]), verbose=0)
        return np.random.choice(len(probs[0]), p=probs[0])

    def train(self, states, actions, advantages):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions)
        advantages = np.asarray(advantages, dtype=np.float32)

        with tf.GradientTape() as tape:
            probs = self.policy_network(states)
            masks = tf.one_hot(actions, probs.shape[-1])
            selected = tf.reduce_sum(probs * masks, axis=1)
            loss = -tf.reduce_sum(tf.math.log(selected + 1e-8) * advantages)

        grads = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables))

class StockMarketEnv:
    def __init__(self, price_data):
        self.price_data = price_data
        self.initial_balance = 10000
        self.max_steps = len(price_data) - 1

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.stock_units = 0
        return [self.balance, self.stock_units]

    def step(self, action):
        if self.current_step >= self.max_steps:
            return [self.balance, self.stock_units], 0, True

        current_price = self.price_data[self.current_step]
        next_price = self.price_data[self.current_step + 1]

        if action == 1 and self.balance >= current_price:
            self.stock_units += 1
            self.balance -= current_price

        elif action == 0 and self.stock_units > 0:
            self.stock_units -= 1
            self.balance += current_price

        self.current_step += 1
        portfolio_value = self.balance + self.stock_units * next_price
        reward = portfolio_value - self.initial_balance
        done = self.current_step == self.max_steps

        return [portfolio_value, self.stock_units], reward, done

def train(agent, env, episodes=100):
    for episode in range(episodes):
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

        # Use episode rewards as the simple advantage signal,
        # matching the exercise's reward-maximization objective.
        agent.train(states, actions, rewards)

        if episode % 10 == 0:
            print(f"Episode {episode+1}, Total Reward: {sum(rewards):.2f}")

if __name__ == "__main__":
    price_data = np.random.uniform(50, 150, size=100)
    env = StockMarketEnv(price_data)
    agent = VPGAgent(2, 2)
    train(agent, env)
