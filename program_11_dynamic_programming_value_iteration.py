# Program 11: Dynamic Programming - Value Iteration
import numpy as np

n_states, n_actions = 3, 2
P = np.zeros((n_states, n_actions, n_states))

P[0, 0, 0], P[0, 0, 1] = 0.7, 0.3
P[0, 1, 1], P[0, 1, 2] = 0.5, 0.5
P[1, 0, 0], P[1, 0, 1] = 0.4, 0.6
P[1, 1, 0], P[1, 1, 1] = 0.1, 0.9
P[2, 0, 2], P[2, 1, 2] = 1.0, 1.0

R = np.zeros((n_states, n_actions, n_states))
R[0, 0, 0], R[0, 0, 1] = 1.0, 2.0
R[0, 1, 1], R[0, 1, 2] = 3.0, 4.0
R[1, 0, 0], R[1, 0, 1] = 0.0, 2.0
R[1, 1, 0], R[1, 1, 1] = 1.0, 3.0

def value_iteration(P, R, gamma=0.9, epsilon=1e-6):
    V = np.zeros(P.shape[0])

    while True:
        V_new = np.zeros_like(V)

        for s in range(P.shape[0]):
            q = []
            for a in range(P.shape[1]):
                q.append(np.sum(P[s, a] * (R[s, a] + gamma * V)))
            V_new[s] = max(q)

        if np.max(np.abs(V - V_new)) < epsilon:
            return V_new

        V = V_new

optimal_values = value_iteration(P, R)
print("Optimal Values:")
print(optimal_values)
