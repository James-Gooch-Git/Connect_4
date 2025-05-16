import os
import sys
import json
from tqdm import tqdm

# Ensure agent functions and game logic are importable
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from agents.ml_agent import ml_move, board_to_input, flip_perspective
from agents.smart_agent import smart_move
from agents.minimax_agent import minimax_move
from connect4 import create_board, drop_disc, get_valid_moves, is_draw, is_winning_move

# Storage
all_games = []

# Matchups: each pair will generate games_per_matchup games
matchups = [
    (ml_move, minimax_move),
    (minimax_move, ml_move),
    (ml_move, smart_move),
    (smart_move, ml_move),
    (smart_move, minimax_move),
    (minimax_move, smart_move)
]

games_per_matchup = 2000  # Total = 6 × 2000 = 12,000 games
SAVE_PATH = "ml/full_training_data.json"

# --- Custom simulation with board state logging
def simulate_and_collect(agent1, agent2):
    board = create_board()
    agents = [agent1, agent2]
    discs = ['●', '○']
    player_turn = 0
    states = []
    game_over = False
    move_count = 0
    outcome = "draw"

    while not game_over:
        agent = agents[player_turn]
        my_disc = discs[player_turn]
        opp_disc = discs[1 - player_turn]

        valid_moves = get_valid_moves(board)
        if not valid_moves:
            break

        move = agent(board, my_disc, opp_disc)
        col = move if isinstance(move, int) else move[0]
        row, success = drop_disc(board, col, my_disc)
        if not success:
            break

        state = board_to_input(board)
        # Flip if ML played this move (always want ML's perspective)
        if my_disc == '○':
            state = flip_perspective(state)
        states.append(state)

        if is_winning_move(board, row, col, my_disc):
            outcome = "win" if agent == ml_move else "loss" if (agent1 == ml_move or agent2 == ml_move) else "neutral"
            game_over = True
        elif is_draw(board):
            outcome = "draw"
            game_over = True

        player_turn = 1 - player_turn
        move_count += 1
        if move_count > 42:
            break

    return {
        "states": states,
        "outcome": outcome if game_over else "draw",
        "agent1": agent1.__name__,
        "agent2": agent2.__name__
    }

# --- Run simulations
print("🎮 Generating AI vs AI games...\n")
for agent1, agent2 in matchups:
    print(f"⚔️ {agent1.__name__} vs {agent2.__name__}")
    for _ in tqdm(range(games_per_matchup)):
        game_data = simulate_and_collect(agent1, agent2)
        all_games.append(game_data)

# --- Save dataset
os.makedirs("ml", exist_ok=True)
with open(SAVE_PATH, "w") as f:
    json.dump(all_games, f)

print(f"\n✅ Dataset complete: {len(all_games)} games saved to {SAVE_PATH}")