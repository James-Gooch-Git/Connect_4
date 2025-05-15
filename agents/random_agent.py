import random
from connect4 import get_valid_moves

def random_move(board, my_disc=None, opp_disc=None):
    valid_moves = get_valid_moves(board)
    return random.choice(valid_moves) if valid_moves else None
