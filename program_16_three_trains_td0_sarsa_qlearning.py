# Program 7: Three Trains - TD(0) vs SARSA vs Q-Learning
# Three algorithms learn to reach the destination on a 1-D track.
import numpy as np
import random

TRACK_LENGTH = 15
GOAL = TRACK_LENGTH - 1
ACTIONS = [0, 1]  # 0 = move backward, 1 = move forward
alpha, gamma, epsilon = 0.2, 0.95, 0.1
episodes, max_steps = 300, 100

def step(state, action):
    next_state = min(GOAL, state + 1) if action == 1 else max(0, state - 1)
    reward = 10 if next_state == GOAL else -1
    return next_state, reward, next_state == GOAL

def epsilon_greedy(Q, state):
    return random.choice(ACTIONS) if random.random() < epsilon else int(np.argmax(Q[state]))

def train_td0():
    V = np.zeros(TRACK_LENGTH)
    steps = []
    for _ in range(episodes):
        state, count = 0, 0
        for _ in range(max_steps):
            ns, reward, done = step(state, 1)
            V[state] += alpha * (reward + gamma * V[ns] - V[state])
            state = ns
            count += 1
            if done:
                break
        steps.append(count)
    return V, steps

def train_sarsa():
    Q = np.zeros((TRACK_LENGTH, 2))
    steps = []
    for _ in range(episodes):
        state = 0
        action = epsilon_greedy(Q, state)
        count = 0
        for _ in range(max_steps):
            ns, reward, done = step(state, action)
            na = epsilon_greedy(Q, ns)
            Q[state, action] += alpha * (reward + gamma * Q[ns, na] - Q[state, action])
            state, action = ns, na
            count += 1
            if done:
                break
        steps.append(count)
    return Q, steps

def train_qlearning():
    Q = np.zeros((TRACK_LENGTH, 2))
    steps = []
    for _ in range(episodes):
        state, count = 0, 0
        for _ in range(max_steps):
            action = epsilon_greedy(Q, state)
            ns, reward, done = step(state, action)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[ns]) - Q[state, action])
            state = ns
            count += 1
            if done:
                break
        steps.append(count)
    return Q, steps

_, td0 = train_td0()
_, sarsa = train_sarsa()
_, qlearning = train_qlearning()

results = {
    "TD(0)": np.mean(td0[-20:]),
    "SARSA": np.mean(sarsa[-20:]),
    "Q-Learning": np.mean(qlearning[-20:])
}

print("Average steps in last 20 episodes:")
for name, value in results.items():
    print(f"{name}: {value:.2f}")

print("Most efficient:", min(results, key=results.get))
