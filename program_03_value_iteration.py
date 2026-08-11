# Program 3: Value Iteration
import numpy as np

n_rows, n_cols = 2, 5
grid_world = np.zeros((n_rows, n_cols))

rewards = {
    (1, 4): 1,
    (1, 3): -1
}

gamma = 0.9

actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
action_names = ["Right", "Left", "Down", "Up"]

def bellman_update(i, j):
    if (i, j) in rewards:
        return rewards[(i, j)]

    values = []
    for di, dj in actions:
        ni, nj = i + di, j + dj
        if 0 <= ni < n_rows and 0 <= nj < n_cols:
            values.append(gamma * grid_world[ni, nj])

    return max(values) if values else 0

for _ in range(100):
    new_grid = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            new_grid[i, j] = bellman_update(i, j)
    grid_world = new_grid

optimal_policy = np.empty((n_rows, n_cols), dtype=object)

for i in range(n_rows):
    for j in range(n_cols):
        if (i, j) in rewards:
            optimal_policy[i, j] = "Reward"
        else:
            values = []
            for di, dj in actions:
                ni, nj = i + di, j + dj
                values.append(
                    gamma * grid_world[ni, nj]
                    if 0 <= ni < n_rows and 0 <= nj < n_cols
                    else -np.inf
                )
            optimal_policy[i, j] = action_names[int(np.argmax(values))]

print("Optimal Policy:")
for row in optimal_policy:
    print(" | ".join(row))
