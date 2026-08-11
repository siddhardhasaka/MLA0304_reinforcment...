# Program 15: Temporal-Difference Travel Optimization
import numpy as np
import random

num_states, num_actions = 21, 4
initial_state, destination_state = 0, 20
Q = np.zeros((num_states, num_actions))
epsilon, alpha, gamma = 0.1, 0.1, 0.9

def take_action(state, action):
    if action == 0:
        return max(0, state - 1), -1
    if action == 1:
        return min(num_states - 1, state + 1), -1
    if action == 2:
        return max(0, state - 5), -1
    return min(num_states - 1, state + 5), -1

for _ in range(200):
    state = initial_state

    for _ in range(200):
        action = random.randrange(num_actions) if random.random() < epsilon else int(np.argmax(Q[state]))
        next_state, reward = take_action(state, action)

        Q[state, action] = (1-alpha)*Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state])
        )

        state = next_state
        if state == destination_state:
            break

state = initial_state
optimal_path = [state]

for _ in range(100):
    if state == destination_state:
        break
    action = int(np.argmax(Q[state]))
    state, _ = take_action(state, action)
    optimal_path.append(state)
else:
    print("Destination not reached within the path safety limit.")

print("Optimal Path:", optimal_path)
