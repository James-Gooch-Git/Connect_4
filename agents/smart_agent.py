import random
from connect4 import get_valid_moves, drop_disc, is_winning_move
import copy

import random
import copy
from connect4 import get_valid_moves, drop_disc, is_winning_move
from agents.heuristics import evaluate_board, evaluate_opponent_threat  # you'll write these

def smart_move(board, my_disc, opp_disc=None):
    valid_moves = get_valid_moves(board)

    best_score = float('-inf')
    best_col = None
    best_heuristic = 0

    for col in valid_moves:
        temp_board = copy.deepcopy(board)
        row, _ = drop_disc(temp_board, col, my_disc)

        if row is None:
            continue

        if is_winning_move(temp_board, row, col, my_disc):
            # Immediate win
            return col, 10000

        # If opponent could win next, block
        opponent_threats = evaluate_opponent_threat(temp_board, my_disc, opp_disc)

        heuristic_score = evaluate_board(temp_board, my_disc)
        total_score = heuristic_score - (opponent_threats * 100)  # Penalize threats

        if total_score > best_score:
            best_score = total_score
            best_col = col
            best_heuristic = heuristic_score

    # If no best found, fallback
    if best_col is None and valid_moves:
        best_col = random.choice(valid_moves)
        best_heuristic = 0

    return best_col, best_heuristic

