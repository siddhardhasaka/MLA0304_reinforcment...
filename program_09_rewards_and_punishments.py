# Program 9: Custom RL Environment - Rewards and Punishments

import numpy as np


class CustomEnvironment:

    def __init__(self):
        self._state = np.array(
            [0.5],
            dtype=np.float32
        )

    def reset(self):
        self._state = np.array(
            [0.5],
            dtype=np.float32
        )

        return self._state.copy()

    def step(self, action):

        if action == 0:
            # Move left
            self._state -= 0.1

        else:
            # Move right
            self._state += 0.1

        done = False

        if self._state <= 0:
            # Punishment for going too far left
            reward = -0.5
            done = True

        elif self._state >= 1:
            # Reward for reaching the goal
            reward = 2.0
            done = True

        else:
            # Neutral step
            reward = 5.0

        return (
            self._state.copy(),
            reward,
            done
        )


# Create environment
environment = CustomEnvironment()

# Reset environment
state = environment.reset()

cumulative_reward = 0.0


# Run the environment
for _ in range(10):

    action = np.random.randint(2)

    next_state, reward, done = environment.step(
        action
    )

    print(
        f"Action: {action}, "
        f"Next State: {next_state}, "
        f"Reward: {reward}"
    )

    cumulative_reward += reward

    state = next_state

    if done:
        break


print(
    f"Cumulative Reward: {cumulative_reward}"
)
