import joblib

import copy
import numpy as np
import pandas as pd
from connect4 import get_valid_moves, drop_disc
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Load the latest trained model

model_path = os.path.join(os.path.dirname(__file__), "..", "ml", "model_PLEASE.joblib")
with open(model_path, "rb") as f:
    model = joblib.load(f)

def flip_perspective(vec):
    return [-v for v in vec]

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

def ml_move(board, my_disc='○', opp_disc='●'):
   # print(f"🔍 ML is making a move as {my_disc}")
    valid_moves = get_valid_moves(board)
    best_col = None
    best_score = -1
    log_lines = []

    for col in valid_moves:
        temp_board = copy.deepcopy(board)
        row, _ = drop_disc(temp_board, col, my_disc)
        input_vec = board_to_input(temp_board)

        # Flip perspective so the AI always thinks it's '●' in the model
        if my_disc == '○':
            input_vec = flip_perspective(input_vec)

        input_df = pd.DataFrame([input_vec], columns=[f"pos_{i}" for i in range(42)])
        outcome = model.predict(input_df)[0]
        score = {"win": 2, "draw": 1, "loss": 0}.get(outcome, 0)
        if col == 3:
            score += 0.25  # Slight bias for the center column

        log_lines.append(f"Evaluating move in column {col}: outcome={outcome}, score={score}")
        if score > best_score:
            best_score = score
            best_col = col
        log_lines.append(f"ML considered column {col} | Best so far: {best_col} (score {best_score})")

    os.makedirs("ml", exist_ok=True)
    with open("ml/ml_log.txt", "a") as log_file:
        log_file.write("\n--- New Game Turn ---\n")
        for line in log_lines:
            log_file.write(line + "\n")

    return best_col
