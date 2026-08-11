# Program 7: Three Trains - TD(0) vs SARSA vs Q-Learning

import numpy as np
import random

TRACK_LENGTH = 15
GOAL = TRACK_LENGTH - 1

ACTIONS = [0, 1]

# 0 = move backward
# 1 = move forward

alpha = 0.2
gamma = 0.95
epsilon = 0.1

episodes = 300
max_steps = 100


def step(state, action):

    if action == 1:
        next_state = min(GOAL, state + 1)
    else:
        next_state = max(0, state - 1)

    reward = 10 if next_state == GOAL else -1
    done = next_state == GOAL

    return next_state, reward, done


def epsilon_greedy(Q, state):

    if random.uniform(0, 1) < epsilon:
        return random.choice(ACTIONS)

    return int(np.argmax(Q[state]))


# ---------------- TD(0) ----------------

def train_td0():

    V = np.zeros(TRACK_LENGTH)
    steps_per_episode = []

    for _ in range(episodes):

        state = 0
        steps = 0

        for _ in range(max_steps):

            action = 1

            next_state, reward, done = step(
                state,
                action
            )

            V[state] += alpha * (
                reward
                + gamma * V[next_state]
                - V[state]
            )

            state = next_state
            steps += 1

            if done:
                break

        steps_per_episode.append(steps)

    return V, steps_per_episode


# ---------------- SARSA ----------------

def train_sarsa():

    Q = np.zeros(
        (TRACK_LENGTH, len(ACTIONS))
    )

    steps_per_episode = []

    for _ in range(episodes):

        state = 0
        action = epsilon_greedy(Q, state)
        steps = 0

        for _ in range(max_steps):

            next_state, reward, done = step(
                state,
                action
            )

            next_action = epsilon_greedy(
                Q,
                next_state
            )

            Q[state, action] += alpha * (
                reward
                + gamma * Q[next_state, next_action]
                - Q[state, action]
            )

            state = next_state
            action = next_action
            steps += 1

            if done:
                break

        steps_per_episode.append(steps)

    return Q, steps_per_episode


# ---------------- Q-Learning ----------------

def train_qlearning():

    Q = np.zeros(
        (TRACK_LENGTH, len(ACTIONS))
    )

    steps_per_episode = []

    for _ in range(episodes):

        state = 0
        steps = 0

        for _ in range(max_steps):

            action = epsilon_greedy(
                Q,
                state
            )

            next_state, reward, done = step(
                state,
                action
            )

            Q[state, action] += alpha * (
                reward
                + gamma * np.max(Q[next_state])
                - Q[state, action]
            )

            state = next_state
            steps += 1

            if done:
                break

        steps_per_episode.append(steps)

    return Q, steps_per_episode


# Train all three algorithms

_, td0_steps = train_td0()
_, sarsa_steps = train_sarsa()
_, qlearning_steps = train_qlearning()


# Compare the average steps
# over the last 20 episodes

results = {
    "Train A - TD(0)": np.mean(td0_steps[-20:]),
    "Train B - SARSA": np.mean(sarsa_steps[-20:]),
    "Train C - Q-Learning": np.mean(qlearning_steps[-20:])
}


print(
    "Average steps to reach the destination "
    "(last 20 episodes):"
)

for name, avg_steps in results.items():

    print(
        f"{name}: {avg_steps:.2f} steps"
    )


winner = min(
    results,
    key=results.get
)

print(
    f"\nMost efficient: {winner}"
)
