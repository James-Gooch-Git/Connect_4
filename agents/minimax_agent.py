import math
import random
import copy
from connect4 import get_valid_moves, drop_disc, is_winning_move, is_draw, ROWS, COLS
from agents.heuristics import evaluate_board  # Only use the external one

def minimax(board, depth, alpha, beta, maximizing, disc, opp_disc, metrics, initial_depth):
    metrics['nodes_expanded'][0] += 1
    current_depth = initial_depth - depth
    if current_depth > metrics['max_depth'][0]:
        metrics['max_depth'][0] = current_depth

    valid_moves = get_valid_moves(board)
    metrics['total_valid_moves'][0] += len(valid_moves)

    is_terminal = (
        is_draw(board) or
        any(is_winning_move(board, r, c, disc) for c in valid_moves for r, _ in [drop_disc(copy.deepcopy(board), c, disc)]) or
        any(is_winning_move(board, r, c, opp_disc) for c in valid_moves for r, _ in [drop_disc(copy.deepcopy(board), c, opp_disc)]) or
        depth == 0
    )

    if is_terminal:
        if depth == 0:
            return None, evaluate_board(board, disc)
        if is_draw(board):
            return None, 0
        for col in valid_moves:
            temp_board = copy.deepcopy(board)
            row, _ = drop_disc(temp_board, col, disc if maximizing else opp_disc)
            if row is not None and is_winning_move(temp_board, row, col, disc if maximizing else opp_disc):
                return col, (1000000 if maximizing else -1000000)

    best_col = None

    if maximizing:
        max_eval = -math.inf
        random.shuffle(valid_moves)
        for col in valid_moves:
            temp_board = copy.deepcopy(board)
            row, success = drop_disc(temp_board, col, disc)
            if not success:
                continue
            if is_winning_move(temp_board, row, col, disc):
                return col, 1000000
            _, eval = minimax(temp_board, depth - 1, alpha, beta, False, disc, opp_disc, metrics, initial_depth)
            if eval > max_eval:
                max_eval = eval
                best_col = col
            alpha = max(alpha, eval)
            if beta <= alpha:
                metrics['total_nodes_pruned'][0] += len(valid_moves) - valid_moves.index(col) - 1
                break
        return best_col, max_eval

    else:
        min_eval = math.inf
        random.shuffle(valid_moves)
        for col in valid_moves:
            temp_board = copy.deepcopy(board)
            row, success = drop_disc(temp_board, col, opp_disc)
            if not success:
                continue
            if is_winning_move(temp_board, row, col, opp_disc):
                return col, -1000000
            _, eval = minimax(temp_board, depth - 1, alpha, beta, True, disc, opp_disc, metrics, initial_depth)
            if eval < min_eval:
                min_eval = eval
                best_col = col
            beta = min(beta, eval)
            if beta <= alpha:
                metrics['total_nodes_pruned'][0] += len(valid_moves) - valid_moves.index(col) - 1
                break
        return best_col, min_eval

def minimax_move(board, my_disc, opp_disc, depth=4):
    metrics = {
        'nodes_expanded': [0],
        'max_depth': [0],
        'total_valid_moves': [0],
        'total_nodes_pruned': [0]
    }

    col, _ = minimax(board, depth, -math.inf, math.inf, True, my_disc, opp_disc, metrics, depth)

    if col is None:
        valid = get_valid_moves(board)
        col = random.choice(valid) if valid else 0  # fallback

    avg_branching_factor = (metrics['total_valid_moves'][0] / max(1, metrics['nodes_expanded'][0])
                            if metrics['nodes_expanded'][0] > 0 else 0)

    # Heuristic evaluation for the chosen move
    heuristic_score = 0
    if col is not None:
        temp_board = copy.deepcopy(board)
        row, _ = drop_disc(temp_board, col, my_disc)
        if row is not None:
            heuristic_score = evaluate_board(temp_board, my_disc)

    return col, metrics['nodes_expanded'][0], metrics['max_depth'][0], avg_branching_factor, metrics['total_nodes_pruned'][0], heuristic_score