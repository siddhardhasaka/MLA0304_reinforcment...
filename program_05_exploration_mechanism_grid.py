# Program 5: Exploration Mechanism on a 6 x 4 Grid
import numpy as np
import random

n_rows, n_cols = 6, 4
actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
epsilon = 0.2
state_values = np.zeros((n_rows, n_cols))

def within_bounds(state):
    r, c = state
    return 0 <= r < n_rows and 0 <= c < n_cols

def choose_action(state):
    if random.random() < epsilon:
        return random.randrange(len(actions))

    values = []
    for dr, dc in actions:
        ns = (state[0] + dr, state[1] + dc)
        values.append(state_values[ns] if within_bounds(ns) else -np.inf)
    return int(np.argmax(values))

for _ in range(1000):
    current_state = (0, 0)

    for _ in range(200):
        action = choose_action(current_state)
        move = actions[action]
        next_state = (current_state[0] + move[0], current_state[1] + move[1])

        reward = 1 if next_state == (5, 3) else 0

        if within_bounds(next_state):
            state_values[current_state] += 0.1 * (
                reward + 0.9 * state_values[next_state]
                - state_values[current_state]
            )
            current_state = next_state

            if current_state == (5, 3):
                break
        else:
            break

print("State Values with Exploration:")
print(state_values)
