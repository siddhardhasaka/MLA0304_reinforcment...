# Program 10: Monte Carlo Method for Maximum Reward
import random

class CustomEnv:
    def __init__(self):
        self.num_states = 10
        self.actions = ["left", "right"]
        self.max_steps = 100

    def reset(self):
        self.current_state = 0
        self.steps = 0
        return self.current_state

    def step(self, action):
        self.steps += 1

        if action == "right":
            self.current_state = min(self.current_state + 1, self.num_states - 1)
            reward = 1 if self.current_state == self.num_states - 1 else 0
        else:
            self.current_state = max(self.current_state - 1, 0)
            reward = 0

        done = self.current_state == self.num_states - 1 or self.steps >= self.max_steps
        return self.current_state, reward, done, {}

def monte_carlo_control(env, episodes=1000, gamma=1.0):
    returns_sum, returns_count, Q, policy = {}, {}, {}, {}

    for _ in range(episodes):
        episode = []
        state = env.reset()
        done = False

        while not done:
            action = random.choice(env.actions)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        G = 0
        visited = set()

        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            pair = (state, action)

            if pair not in visited:
                visited.add(pair)
                returns_sum[pair] = returns_sum.get(pair, 0) + G
                returns_count[pair] = returns_count.get(pair, 0) + 1
                Q[pair] = returns_sum[pair] / returns_count[pair]

        for s in range(env.num_states):
            available = [a for a in env.actions if (s, a) in Q]
            if available:
                policy[s] = max(available, key=lambda a: Q[(s, a)])
            else:
                policy[s] = random.choice(env.actions)

    return Q, policy

env = CustomEnv()
Q, optimal_policy = monte_carlo_control(env)

print("Learned Optimal Policy:")
for state, action in optimal_policy.items():
    print(f"State: {state}, Action: {action}")
