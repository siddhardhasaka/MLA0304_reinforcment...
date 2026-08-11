# Program 1: Neural Network Reinforcement Learning Agent
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make("CartPole-v1")

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class Agent:
    def __init__(self, input_size, output_size):
        self.policy_network = PolicyNetwork(input_size, output_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.01)
        self.output_size = output_size

    def select_action(self, state):
        state = torch.from_numpy(np.asarray(state, dtype=np.float32))
        probabilities = self.policy_network(state).detach().numpy()
        return np.random.choice(np.arange(self.output_size), p=probabilities)

agent = Agent(env.observation_space.shape[0], env.action_space.n)

for episode in range(200):
    state, info = env.reset()
    episode_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.optimizer.zero_grad()
        state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32))
        action_tensor = torch.tensor(action)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)

        log_prob = torch.log(agent.policy_network(state_tensor)[action_tensor])
        loss = -log_prob * reward_tensor
        loss.backward()
        agent.optimizer.step()

        episode_reward += reward
        state = next_state

        if done:
            break

    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {episode_reward}")

env.close()
