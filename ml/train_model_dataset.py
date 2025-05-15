import os
import copy
import random
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from connect4 import create_board, drop_disc, is_winning_move, is_draw
from connect4 import create_board, drop_disc, is_winning_move, is_draw
from agents.smart_agent import smart_move
from agents.minimax_agent import minimax_move

def board_to_input(board):
    flat = []
    for row in board:
        for cell in row:
            if cell == '●':
                flat.append(1)
            elif cell == '○':
                flat.append(-1)
            else:
                flat.append(0)
    return flat

def simulate_training_data(n=10000):
    X = []
    Y = []
    for _ in tqdm(range(n), desc="Simulating games", unit="game", dynamic_ncols=True):
        board = create_board()
        player_turn = random.choice([0, 1])
        discs = ['●', '○']
        agents = [minimax_move, smart_move]
        states = []

        while True:
            disc = discs[player_turn]
            result = agents[player_turn](copy.deepcopy(board), disc, discs[1 - player_turn])
            move = result if isinstance(result, int) else result[0]
            if move is None or board[0][move] != ' ':
                outcome = "loss" if player_turn == 0 else "win"
                break

            row, _ = drop_disc(board, move, disc)
            states.append(board_to_input(board))

            if is_winning_move(board, row, move, disc):
                outcome = "win" if player_turn == 0 else "loss"
                break
            if is_draw(board):
                outcome = "draw"
                break

            player_turn = 1 - player_turn

        label = {"win": 2, "draw": 1, "loss": 0}[outcome]
        X.extend(states)
        Y.extend([label] * len(states))
    return X, Y

def train_and_save_model(X, Y, model_path="ml/model_from_sim.joblib"):
    df = pd.DataFrame(X, columns=[f"pos_{i}" for i in range(42)])
    df["outcome"] = Y

    X = df.drop("outcome", axis=1)
    y = df["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy on test set: {acc:.4f}")

    os.makedirs("ml", exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    X, Y = simulate_training_data(10000)
    train_and_save_model(X, Y)