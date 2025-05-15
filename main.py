from connect4 import create_board, print_board, drop_disc, is_winning_move, is_draw, get_valid_moves
from agents.random_agent import random_move
from agents.smart_agent import smart_move  # Add others later
from agents.minimax_agent import minimax_move
from ml.replay_logger import save_failed_game

import random 

def play_game(agent_name="random"):
    board = create_board()
    print_board(board)

    player_turn = random.choice([0, 1])
    print(f"{'Human' if player_turn == 0 else 'AI'} goes first!\n")

    while True:
        if player_turn == 0:
            try:
                col = int(input("Your move (0-6): "))
            except ValueError:
                print("Invalid input. Try again.")
                continue
        else:
            if agent_name == "random":
                col = random_move(board)
            elif agent_name == "smart":
                col = smart_move(board, '○', '●')
            elif agent_name == "minimax":
                col, *_ = minimax_move(board, '○', '●')

            elif agent_name == "ml":
                col = ml_move(board, '○')
            else:
                print("Unknown agent selected.")
                return
            print(f"AI ({agent_name}) chooses column {col}")

        if col not in range(7):
            print("Column out of range.")
            continue
        if board[0][col] != ' ':
            print("Column is full.")
            continue

        disc = '●' if player_turn == 0 else '○'
        row, _ = drop_disc(board, col, disc)
        print_board(board)

        if is_winning_move(board, row, col, disc):
            print("You win!" if player_turn == 0 else "AI wins!")
            break
        if is_draw(board):
            print("It's a draw!")
            break

        player_turn = 1 - player_turn

if __name__ == "__main__":
    print("Choose the AI agent to play against:")
    print("1 - Random Agent")
    print("2 - Smart Agent")
    print("3 - Minimax Agent")
    print("4 - ML Agent")
    
    choice = input("Enter choice (1-4): ")
    agent_map = {
        "1": "random",
        "2": "smart",
        "3": "minimax",
        "4": "ml"
    }

    selected_agent = agent_map.get(choice, "random")
    print(f"\nStarting game against: {selected_agent} agent\n")
    play_game(agent_name=selected_agent)
