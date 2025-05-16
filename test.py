from connect4 import create_board, drop_disc, print_board
from agents.ml_agent import ml_move

board = create_board()
drop_disc(board, 3, '●')  # ML agent's disc
drop_disc(board, 3, '●')
drop_disc(board, 3, '●')

print_board(board)

# ML agent is always passed '●' as my_disc to match training
move = ml_move(board, my_disc='●', opp_disc='○')
print("Predicted move:", move)