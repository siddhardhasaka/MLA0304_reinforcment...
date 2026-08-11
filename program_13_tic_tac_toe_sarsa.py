# Program 13: Tic-Tac-Toe with SARSA
import random

class TicTacToe:
    def __init__(self):
        self.winning_combos = [
            [0,1,2], [3,4,5], [6,7,8],
            [0,3,6], [1,4,7], [2,5,8],
            [0,4,8], [2,4,6]
        ]
        self.reset()

    def reset(self):
        self.board = [" "] * 9
        self.current_player = "X"

    def is_winner(self, player):
        return any(all(self.board[i] == player for i in combo)
                   for combo in self.winning_combos)

    def is_draw(self):
        return " " not in self.board

    def is_game_over(self):
        return self.is_winner("X") or self.is_winner("O") or self.is_draw()

    def available_moves(self):
        return [i for i, mark in enumerate(self.board) if mark == " "]

    def make_move(self, move):
        self.board[move] = self.current_player
        self.current_player = "O" if self.current_player == "X" else "X"

Q = {}
alpha, gamma, epsilon = 0.1, 0.9, 0.1
env = TicTacToe()

def get_action(state):
    moves = env.available_moves()
    if not moves:
        return None
    if random.random() < epsilon:
        return random.choice(moves)
    return max(moves, key=lambda x: Q.get((state, x), 0))

def update_Q(state, action, reward, next_state, next_action):
    Q[(state, action)] = Q.get((state, action), 0)
    next_value = Q.get((next_state, next_action), 0) if next_action is not None else 0
    Q[(state, action)] += alpha * (
        reward + gamma * next_value - Q[(state, action)]
    )

for _ in range(10000):
    env.reset()
    state = tuple(env.board)
    action = get_action(state)

    while not env.is_game_over() and action is not None:
        env.make_move(action)
        next_state = tuple(env.board)

        if env.is_winner("X"):
            reward = 1
        elif env.is_winner("O"):
            reward = -1
        else:
            reward = 0

        next_action = get_action(next_state)
        update_Q(state, action, reward, next_state, next_action)

        state, action = next_state, next_action

print("SARSA training completed.")
print("Learned state-action pairs:", len(Q))
