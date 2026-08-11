# Program 3: Value Iteration in Reinforcement Learning

import numpy as np

# Define the grid world
n_rows, n_cols = 2, 5

grid_world = np.zeros((n_rows, n_cols))

# Define rewards
rewards = {
    (1, 4): 1,    # Maximum Reward
    (1, 3): -1    # Fire state
}

# Discount factor
gamma = 0.9

# Actions
actions = [
    (0, 1),    # Right
    (0, -1),   # Left
    (1, 0),    # Down
    (-1, 0)    # Up
]

action_names = [
    'Right',
    'Left',
    'Down',
    'Up'
]


# Bellman update function
def bellman_update(i, j, action):

    if (i, j) in rewards:
        return rewards[(i, j)]

    total_reward = 0

    for a, (di, dj) in enumerate(actions):

        next_i = i + di
        next_j = j + dj

        if 0 <= next_i < n_rows and 0 <= next_j < n_cols:
            total_reward += (
                0.25 * grid_world[next_i, next_j] * gamma
            )

    return total_reward


# Perform value iteration
num_iterations = 100

for _ in range(num_iterations):

    new_grid_world = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):

            new_grid_world[i, j] = max(
                [
                    bellman_update(i, j, action)
                    for action in actions
                ]
            )

    grid_world = new_grid_world


# Determine the optimal policy
optimal_policy = np.empty(
    (n_rows, n_cols),
    dtype=object
)

for i in range(n_rows):
    for j in range(n_cols):

        if (i, j) not in rewards:

            optimal_policy[i, j] = action_names[
                np.argmax(
                    [
                        bellman_update(i, j, action)
                        for action in actions
                    ]
                )
            ]

        else:
            optimal_policy[i, j] = None


# Display optimal policy
optimal_policy = np.where(
    optimal_policy != None,
    optimal_policy.astype(str),
    'Reward'
)

print("Optimal Policy:")

for row in optimal_policy:
    print(" | ".join(row))
