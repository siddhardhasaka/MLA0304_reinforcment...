# Program 22: PPO for Optimal Routing Policy
# Requires: stable-baselines3, gymnasium
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=20000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"PPO mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

random_env = gym.make("CartPole-v1")
random_rewards = []

for _ in range(10):
    obs, _ = random_env.reset()
    total_reward = 0
    done = False

    while not done:
        action = random_env.action_space.sample()
        obs, reward, terminated, truncated, _ = random_env.step(action)
        done = terminated or truncated
        total_reward += reward

    random_rewards.append(total_reward)

print(f"Random policy mean reward: {np.mean(random_rewards):.2f} +/- {np.std(random_rewards):.2f}")

env.close()
random_env.close()
