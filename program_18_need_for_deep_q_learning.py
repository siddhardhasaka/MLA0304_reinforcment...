# Program 18: Demonstrate the Need for Deep Q-Learning
import numpy as np

class ChessEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.array([
            ["R","N","B","Q","K","B","N","R"],
            ["P","P","P","P","P","P","P","P"],
            [" "," "," "," "," "," "," "," "],
            [" "," "," "," "," "," "," "," "],
            [" "," "," "," "," "," "," "," "],
            [" "," "," "," "," "," "," "," "],
            ["p","p","p","p","p","p","p","p"],
            ["r","n","b","q","k","b","n","r"]
        ])
        self.current_player = "white"

    def is_checkmate(self, player):
        king = "K" if player == "white" else "k"
        return not np.any(self.board == king)

    def make_move(self, move):
        if move is not None:
            r1,c1,r2,c2 = move
            self.board[r2,c2] = self.board[r1,c1]
            self.board[r1,c1] = " "
            self.current_player = "black" if self.current_player == "white" else "white"

class DQLAgent:
    def choose_move(self, state):
        return (0,0,1,0) if np.random.rand() < 0.5 else None

env = ChessEnvironment()
agent = DQLAgent()

for episode in range(10):
    env.reset()
    moves = 0

    while not env.is_checkmate(env.current_player) and moves < 200:
        move = agent.choose_move(env.board)
        env.make_move(move)
        moves += 1

    status = "checkmate" if env.is_checkmate(env.current_player) else "step cap reached"
    print(f"Episode {episode + 1}: ended after {moves} moves ({status})")

print("\nThis demonstrates why a learned neural Q-function is useful in large state spaces.")
