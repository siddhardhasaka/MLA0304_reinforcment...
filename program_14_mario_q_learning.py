# Program 14: Mario-style Q-Learning on a 5 x 5 Grid
import numpy as np

class MarioEnvironment:
    def __init__(self):
        self.grid_size = (5, 5)
        self.goal_position = (4, 4)
        self.obstacle_positions = [(1,1), (2,2), (3,3)]
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.Q_values = np.zeros((5, 5, 4))
        self.alpha, self.gamma, self.epsilon = 0.1, 0.9, 0.1
        self.agent_position = (0, 0)

    def valid(self, p):
        return 0 <= p[0] < 5 and 0 <= p[1] < 5

    def reward(self, p):
        if p == self.goal_position:
            return 10
        if p in self.obstacle_positions:
            return -5
        return -1

    def move(self, action):
        r, c = self.agent_position
        moves = {
            "UP": (r-1,c), "DOWN": (r+1,c),
            "LEFT": (r,c-1), "RIGHT": (r,c+1)
        }
        if self.valid(moves[action]):
            self.agent_position = moves[action]

    def q_learning(self, episodes=1000, max_steps=500):
        for episode in range(episodes):
            self.agent_position = (0, 0)
            state = self.agent_position
            total_reward = 0

            for _ in range(max_steps):
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.actions)
                else:
                    action = self.actions[np.argmax(self.Q_values[state])]

                self.move(action)
                new_state = self.agent_position
                reward = self.reward(new_state)
                total_reward += reward

                a = self.actions.index(action)
                self.Q_values[state][a] += self.alpha * (
                    reward + self.gamma * np.max(self.Q_values[new_state])
                    - self.Q_values[state][a]
                )

                state = new_state

                if state == self.goal_position:
                    break

            if episode % 100 == 0:
                print(f"Episode {episode + 1}, Total Reward: {total_reward}")

env = MarioEnvironment()
env.q_learning()
