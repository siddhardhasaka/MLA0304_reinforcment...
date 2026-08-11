# Program 9: Custom RL Environment - Rewards and Punishments
import numpy as np

class CustomEnvironment:
    def __init__(self):
        self._state = np.array([0.5], dtype=np.float32)

    def reset(self):
        self._state = np.array([0.5], dtype=np.float32)
        return self._state.copy()

    def step(self, action):
        self._state += -0.1 if action == 0 else 0.1

        if self._state <= 0:
            reward, done = -0.5, True
        elif self._state >= 1:
            reward, done = 2.0, True
        else:
            reward, done = 5.0, False

        return self._state.copy(), reward, done

environment = CustomEnvironment()
state = environment.reset()
cumulative_reward = 0.0

for _ in range(10):
    action = np.random.randint(2)
    next_state, reward, done = environment.step(action)
    print(f"Action: {action}, Next State: {next_state}, Reward: {reward}")
    cumulative_reward += reward
    state = next_state
    if done:
        break

print("Cumulative Reward:", cumulative_reward)
