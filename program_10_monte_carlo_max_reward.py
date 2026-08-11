# Program 10: Monte Carlo Method for Maximum Reward

import numpy as np
import random


class CustomEnv:

    def __init__(self):
        self.num_states = 10
        self.actions = ['left', 'right']
        self.max_steps = 100
        self.current_state = 0
        self.reward = 0
        self.total_reward = 0
        self.steps = 0

    def reset(self):
        self.current_state = 0
        self.reward = 0
        self.total_reward = 0
        self.steps = 0

        return self.current_state

    def step(self, action):

        self.steps += 1

        if action == 'right':
            self.current_state += 1

            self.reward = (
                1
                if self.current_state == self.num_states - 1
                else 0
            )

        else:
            self.current_state -= (
                1
                if self.current_state > 0
                else 0
            )

            self.reward = 0

        self.total_reward += self.reward

        done = (
            self.current_state == self.num_states - 1
            or self.steps >= self.max_steps
        )

        return (
            self.current_state,
            self.reward,
            done,
            {}
        )


def monte_carlo_control(
    env,
    episodes=1000,
    gamma=1.0
):

    returns_sum = {}
    returns_count = {}

    Q = {}
    policy = {}

    for episode in range(episodes):

        states_actions_returns = []

        state = env.reset()

        done = False

        while not done:

            action = random.choice(env.actions)

            next_state, reward, done, _ = env.step(
                action
            )

            states_actions_returns.append(
                (state, action, reward)
            )

            state = next_state

        G = 0

        for i, (
            state,
            action,
            reward
        ) in enumerate(
            reversed(states_actions_returns)
        ):

            G = gamma * G + reward

            visited_pairs = [
                (x[0], x[1])
                for x in states_actions_returns[
                    ::-1
                ][i + 1:]
            ]

            if (state, action) not in visited_pairs:

                if (state, action) in returns_sum:
                    returns_sum[(state, action)] += G
                    returns_count[(state, action)] += 1

                else:
                    returns_sum[(state, action)] = G
                    returns_count[(state, action)] = 1

                Q[(state, action)] = (
                    returns_sum[(state, action)]
                    / returns_count[(state, action)]
                )

        for s in range(env.num_states):

            available_actions = [
                act
                for act in env.actions
                if (s, act) in Q
            ]

            if available_actions:

                best_value = max(
                    Q[(s, a)]
                    for a in available_actions
                )

                best_actions = [
                    a
                    for a in available_actions
                    if Q[(s, a)] == best_value
                ]

                policy[s] = random.choice(
                    best_actions
                )

            else:
                policy[s] = random.choice(
                    env.actions
                )

    return Q, policy


# Create environment
env = CustomEnv()

# Train using Monte Carlo control
Q, optimal_policy = monte_carlo_control(env)


print("Learned Optimal Policy:")

for state, action in optimal_policy.items():
    print(
        f"State: {state}, Action: {action}"
    )
