# Program 12: News Recommendation using TD(0)
import numpy as np

n_articles = 10
n_states = 5
Q = np.zeros((n_states, n_articles))
rewards = np.random.rand(n_states, n_articles)

alpha, gamma = 0.1, 0.9
num_episodes = 1000

for _ in range(num_episodes):
    state = np.random.randint(n_states)

    for _ in range(100):
        if np.random.rand() < 0.1:
            action = np.random.randint(n_articles)
        else:
            action = int(np.argmax(Q[state]))

        next_state = np.random.randint(n_states)
        reward = rewards[state, action]

        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

        if np.random.rand() < 0.1:
            break

optimal_policy = np.argmax(Q, axis=1)

print("Optimal Policy:")
print(optimal_policy)
