# Program 19: Deep Q-Learning to Win Chess Faster
# Simplified demonstration based on the chess environment in the manual.
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
        r1,c1,r2,c2 = move
        self.board[r2,c2] = self.board[r1,c1]
        self.board[r1,c1] = " "
        self.current_player = "black" if self.current_player == "white" else "white"

class DQLAgent:
    def __init__(self):
        self.possible_moves = [
            (r1,c1,r2,c2)
            for r1 in range(8) for c1 in range(8)
            for r2 in range(8) for c2 in range(8)
        ]

    def choose_move(self, state):
        return self.possible_moves[np.random.randint(len(self.possible_moves))]

env = ChessEnvironment()
agent = DQLAgent()

for episode in range(10):
    env.reset()
    moves = 0

    while not env.is_checkmate(env.current_player) and moves < 500:
        move = agent.choose_move(env.board)
        env.make_move(move)
        moves += 1

        if moves % 50 == 0:
            print(f"Episode {episode+1}, move {moves}, player: {env.current_player}")

    outcome = "checkmate" if env.is_checkmate(env.current_player) else "step cap reached"
    print(f"Episode {episode+1} finished after {moves} moves ({outcome})")
