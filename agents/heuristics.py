import copy
from connect4 import is_winning_move, ROWS, COLS

def evaluate_board(board, disc):
    score = 0
    center_col = [row[len(board[0]) // 2] for row in board]
    score += center_col.count(disc) * 3
    return score

def evaluate_opponent_threat(board, current_disc, opponent_disc):
    threats = 0
    for col in range(len(board[0])):
        temp_board = simulate_move(board, col, opponent_disc)
        if temp_board:
            row = get_drop_row(temp_board, col)
            if row is not None and is_winning_move(temp_board, row, col, opponent_disc):
                threats += 1
    return threats

def simulate_move(board, col, disc):
    temp_board = copy.deepcopy(board)
    for row in reversed(range(ROWS)):
        if temp_board[row][col] == ' ':
            temp_board[row][col] = disc
            return temp_board
    return None

def get_drop_row(board, col):
    for row in reversed(range(ROWS)):
        if board[row][col] == ' ':
            return row
    return None
