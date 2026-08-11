# Program 17: Tic-Tac-Toe with Q-Learning
import random

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [" "] * 9
        self.current_player = "X"
        self.winner = None

    def make_move(self, action):
        if action is not None and self.board[action] == " " and not self.winner:
            self.board[action] = self.current_player
            self.check_winner()
            self.current_player = "O" if self.current_player == "X" else "X"

    def check_winner(self):
        combos = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        for a,b,c in combos:
            if self.board[a] == self.board[b] == self.board[c] != " ":
                self.winner = self.board[a]
                break

    def is_game_over(self):
        return " " not in self.board or self.winner is not None

    def get_state(self):
        return tuple(self.board)

class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.epsilon, self.alpha, self.gamma = epsilon, alpha, gamma
        self.q_table = {}

    def choose_action(self, state):
        available = [i for i, x in enumerate(state) if x == " "]
        if not available:
            return None
        if random.random() < self.epsilon or state not in self.q_table:
            return random.choice(available)
        return max(available, key=lambda i: self.q_table[state][i])

    def ensure(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 9

    def update(self, state, action, reward, next_state):
        self.ensure(state)
        self.ensure(next_state)
        if action is not None:
            next_max = max(self.q_table[next_state])
            self.q_table[state][action] += self.alpha * (
                reward + self.gamma * next_max - self.q_table[state][action]
            )

def train(agent, env, episodes=10000):
    for _ in range(episodes):
        env.reset()
        while not env.is_game_over():
            state = env.get_state()
            action = agent.choose_action(state)
            if action is None:
                break

            env.make_move(action)
            next_state = env.get_state()

            reward = 1 if env.winner == "X" else -1 if env.winner == "O" else 0
            agent.update(state, action, reward, next_state)

agent = QLearningAgent()
env = TicTacToe()
train(agent, env)

print("Training completed.")
print("Learned states:", len(agent.q_table))
