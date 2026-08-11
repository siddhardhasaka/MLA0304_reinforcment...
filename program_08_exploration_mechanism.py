# Program 8: Exploration Mechanism on a 6 x 4 Grid

import numpy as np
import random

n_rows, n_cols = 6, 4

actions = [
    (0, 1),    # Right
    (0, -1),   # Left
    (1, 0),    # Down
    (-1, 0)    # Up
]

epsilon = 0.2

state_values = np.zeros((n_rows, n_cols))


def within_bounds(state):
    row, col = state

    return (
        0 <= row < n_rows
        and 0 <= col < n_cols
    )


def choose_action(state):

    if random.uniform(0, 1) < epsilon:
        # Exploration
        return random.choice(
            range(len(actions))
        )

    else:
        # Exploitation
        valid_actions = []

        for action in actions:

            next_state = (
                state[0] + action[0],
                state[1] + action[1]
            )

            if within_bounds(next_state):
                valid_actions.append(
                    state_values[next_state]
                )
            else:
                valid_actions.append(
                    float('-inf')
                )

        return np.argmax(valid_actions)


num_episodes = 1000

max_steps_per_episode = 200


for _ in range(num_episodes):

    current_state = (0, 0)

    for _step in range(max_steps_per_episode):

        action = choose_action(current_state)

        move = actions[action]

        next_state = (
            current_state[0] + move[0],
            current_state[1] + move[1]
        )

        if next_state == (5, 3):
            reward = 1
        else:
            reward = 0

        if within_bounds(next_state):

            state_values[current_state] += 0.1 * (
                reward
                + 0.9 * state_values[next_state]
                - state_values[current_state]
            )

            current_state = next_state

            if current_state == (5, 3):
                break

        else:
            break


print("State Values with Exploration:")
print(state_values)
