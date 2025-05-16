import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import random
import math # Make sure math is imported for minimax
import time 
import tracemalloc
import subprocess 
import random
import time
import copy
import psutil
import sys
import subprocess

from connect4 import (
    create_board, print_board, get_valid_moves,
    drop_disc, is_draw, is_winning_move
)
from agents.ml_agent import ml_move, board_to_input
from ml.replay_logger import save_failed_game

from connect4 import (
    create_board,
    drop_disc,
    is_winning_move,
    is_draw,
    get_valid_moves,
    print_board,
    ROWS,  # Import ROWS
    COLS   # Import COLS
)
from agents.heuristics import (
    evaluate_board,
    evaluate_opponent_threat,
    simulate_move,
    get_drop_row
)
from agents.smart_agent import smart_move
from agents.ml_agent import ml_move
from agents.random_agent import random_move
from agents.minimax_agent import minimax_move # Make sure minimax_move is imported
from ml.replay_logger import save_failed_game


def random_move_wrapped(board, my_disc, opp_disc):
    return random_move(board, my_disc, opp_disc)



# def simulate_game(agent1, agent2):
#     board = create_board()
#     ml_board_states = []
    
#     # Determine which agent is the ML agent (if any)
#     ml_agent_idx = None
#     if agent1 == ml_move:
#         ml_agent_idx = 0
#     elif agent2 == ml_move:
#         ml_agent_idx = 1
    
#     # Randomly choose who goes first
#     player_turn = random.choice([0, 1])
#     first_player = player_turn
#     agents = [agent1, agent2]
    
#     # Always assign circle ('○') to ML agent if it's playing
#     if ml_agent_idx is not None:
#         discs = ['●', '●']  # Default both to filled
#         discs[ml_agent_idx] = '○'  # ML agent gets circle
#     else:
#         discs = ['●', '○']  # Default assignment if no ML agent
    
#     move_count = 0

#     while True:
#         current_agent = agents[player_turn]
#         col = current_agent(board, discs[player_turn], discs[1 - player_turn])
#         current_agent = agents[player_turn]
#         col = current_agent(board, discs[player_turn], discs[1 - player_turn])
#         if col is None or board[0][col] != ' ':
#             print(f"⚠️ Invalid move by agent {player_turn} ({'Agent 1' if player_turn == 0 else 'Agent 2'})")
#             print(f"Returned column: {col}")
#             print_board(board)
#             winner = 1 - player_turn
#             return f"{winner}-wins-{first_player}", False

#         if (current_agent == ml_move) and (discs[player_turn] == '○'):
#             from agents.ml_agent import board_to_input
#             ml_board_states.append(board_to_input(board))

#         row, _ = drop_disc(board, col, discs[player_turn])

#         if is_winning_move(board, row, col, discs[player_turn]):
#             # ML agent *just lost* if it is NOT the winning player
#             if ml_agent_idx is not None:
#                 # Determine if ML agent lost or won
#                 outcome = "loss" if player_turn != ml_agent_idx else "win"
#                 if ml_board_states:
#                     save_failed_game(ml_board_states, outcome)
#             ml_lost = (ml_agent_idx is not None and player_turn != ml_agent_idx)
#             return f"{player_turn}-wins-{first_player}", ml_lost

#         if is_draw(board):
#             # ML agent played to a draw
#             if ml_agent_idx is not None and ml_board_states:
#                 save_failed_game(ml_board_states, "draw")
#             return f"draw-{first_player}", False


#         player_turn = 1 - player_turn

# Keep all your imports and other functions above this
def random_move_wrapped(board, my_disc, opp_disc):
    return random_move(board, my_disc, opp_disc)



def simulate_game(agent1, agent2, verbose=False):
    board = create_board()
    move_durations = {0: [], 1: [], 'memory_usage': {0: [], 1: []}}
    heuristic_scores = {0: [], 1: []}
    threats_faced = {0: 0, 1: 0}
    threats_blocked = {0: 0, 1: 0}
    ml_board_states = []

    # Determine which player is ML
    ml_agent_idx = None
    if agent1 == ml_move:
        ml_agent_idx = 0
    elif agent2 == ml_move:
        ml_agent_idx = 1

    # Force ML agent to always play as ○ for consistency
    discs = ['○', '●'] if ml_agent_idx == 0 else ['●', '○']

    player_turn = random.choice([0, 1])
    first_player = player_turn
    agents = [agent1, agent2]

    game_over = False
    winner_idx = None
    win_type = None
    total_moves = 0

    metrics = {'nodes_expanded': [], 'max_depth': [], 'branching_factor': [], 'nodes_pruned': []}

    while not game_over:
        agent = agents[player_turn]
        disc = discs[player_turn]

        valid_moves = get_valid_moves(board)

        if not valid_moves:
            break

        # Track memory before move
        # Track memory before move
        proc = psutil.Process()
        mem_before = proc.memory_info().rss / 1024

        start = time.time()
        try:
            move_result = agent(board, disc, discs[1 - player_turn])
        except TypeError as e:
            raise RuntimeError(f"{agent.__name__}() failed: {e}")
        duration = time.time() - start


        mem_after = proc.memory_info().rss / 1024
        move_durations[player_turn].append(duration)
        move_durations['memory_usage'][player_turn].append(mem_after - mem_before)


        # Log ML board state
        if ml_agent_idx == player_turn:
            ml_board_states.append(board_to_input(board))

        col = move_result if isinstance(move_result, int) else move_result[0]
        row, success = drop_disc(board, col, disc)
        if not success:
            break

        # Heuristic stub: use simple piece count for now
        heuristic_scores[player_turn].append(sum(row.count(disc) for row in board))

        # Blocked threats (rudimentary version)
        for threat_col in get_valid_moves(board):
            temp_board = copy.deepcopy(board)
            r, _ = drop_disc(temp_board, threat_col, discs[1 - player_turn])
            if r is not None and is_winning_move(temp_board, r, threat_col, discs[1 - player_turn]):
                threats_faced[player_turn] += 1
                if col == threat_col:
                    threats_blocked[player_turn] += 1

        total_moves += 1
        if is_winning_move(board, row, col, disc):
            game_over = True
            winner_idx = player_turn
            win_type = is_winning_move(board, row, col, disc, return_type=True)
        elif is_draw(board):
            game_over = True
            win_type = "draw"

        player_turn = 1 - player_turn

    # Log game outcome for ML agent
    if ml_agent_idx is not None and ml_board_states:
        if winner_idx is None:
            outcome = "draw"
        else:
            outcome = "win" if winner_idx == ml_agent_idx else "loss"
        save_failed_game(ml_board_states, outcome)

        if outcome == "loss":
            simulate_game.ml_losses += 1
            if simulate_game.ml_losses % 20 == 0:
                print(f"🔁 Retraining ML model after {simulate_game.ml_losses} ML losses...")
                try:
                    subprocess.run([sys.executable, "ml/train_model.py"], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Retraining failed: {e}")

    result_str = (
        f"{winner_idx}-wins-{first_player}"
        if winner_idx is not None
        else f"draw-{first_player}"
    )

    return (
        result_str,
        winner_idx is not None and winner_idx == ml_agent_idx,
        total_moves,
        metrics,
        move_durations,
        win_type,
        heuristic_scores,
        threats_faced,
        threats_blocked
    )

# Track ML losses across simulations
simulate_game.ml_losses = 0
def plot_memory_usage(memory_usage_data, agent_labels, save_path="evaluation/memory_usage.png"):
    import matplotlib.pyplot as plt
    import numpy as np

    avg_mem_usage = [
        np.mean(memory_usage_data[0]) if memory_usage_data[0] else 0,
        np.mean(memory_usage_data[1]) if memory_usage_data[1] else 0
    ]

    plt.figure(figsize=(8, 6))
    plt.bar(agent_labels, avg_mem_usage, color=['skyblue', 'salmon'])
    plt.title("Average Peak Memory Usage per Move")
    plt.ylabel("Memory Usage (KB)")
    plt.xlabel("Agent")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Memory usage plot saved to {save_path}")

def extract_memory_usage_stats(aggregated_agent_move_durations):
    """s
    Extract memory usage data for plotting from the aggregated structure.
    Returns a tuple: (agent0_memory_list, agent1_memory_list)
    """
    return (
        aggregated_agent_move_durations['memory_usage'][0],
        aggregated_agent_move_durations['memory_usage'][1]
    )



def run_simulations(agent1, agent2, n=500):
    
    results_counts = {
        'agent1_wins_first': 0,
        'agent2_wins_first': 0,
        'agent1_wins_second': 0,
        'agent2_wins_second': 0,
        'draws_first': 0,
        'draws_second': 0,
        'total_games': n
    }

    win_type_outcomes = []

    aggregated_minimax_metrics = {
        'nodes_expanded': [],
        'max_depth': [],
        'branching_factor': [],
        'nodes_pruned': []
    }

    aggregated_memory_usage = {
    0: [],
    1: []
    }


    aggregated_agent_move_durations = {
        0: [],
        1: []
    }

    aggregated_heuristic_scores = {
        0: [],
        1: []
    }

    aggregated_threats_faced = {
        0: 0,
        1: 0
    }

    aggregated_threats_blocked = {
        0: 0,
        1: 0
    }

    ml_losses = 0

    for game_idx in tqdm(range(n), desc="Simulating games", dynamic_ncols=False):
        try:
            result, ml_lost, game_length, minimax_metrics, game_move_durations, win_type, heuristic_scores, threats_faced, threats_blocked = simulate_game(agent1, agent2)
        except Exception as e:
            print(f"Error during simulate_game execution for game {game_idx}: {e}. Skipping outcome tracking.")
            continue

        # Aggregate Minimax metrics if used
        if (agent1 == minimax_move or agent2 == minimax_move) and minimax_metrics['nodes_expanded']:
            aggregated_minimax_metrics['nodes_expanded'].extend(minimax_metrics['nodes_expanded'])
            aggregated_minimax_metrics['max_depth'].extend(minimax_metrics['max_depth'])
            aggregated_minimax_metrics['branching_factor'].extend(minimax_metrics['branching_factor'])
            aggregated_minimax_metrics['nodes_pruned'].extend(minimax_metrics['nodes_pruned'])

        aggregated_memory_usage[0].extend(game_move_durations['memory_usage'][0])
        aggregated_memory_usage[1].extend(game_move_durations['memory_usage'][1])


        # Aggregate move durations
        aggregated_agent_move_durations[0].extend(game_move_durations[0])
        aggregated_agent_move_durations[1].extend(game_move_durations[1])

        # Aggregate heuristic scores
        aggregated_heuristic_scores[0].extend(heuristic_scores[0])
        aggregated_heuristic_scores[1].extend(heuristic_scores[1])

        # Aggregate threat stats
        aggregated_threats_faced[0] += threats_faced[0]
        aggregated_threats_faced[1] += threats_faced[1]
        aggregated_threats_blocked[0] += threats_blocked[0]
        aggregated_threats_blocked[1] += threats_blocked[1]

        if win_type:
            win_type_outcomes.append(win_type)

        # Parse result string
        parts = result.split('-')
        if len(parts) == 3:
            winner_idx = int(parts[0])
            first_player_idx = int(parts[2])

            if winner_idx == 0:
                if first_player_idx == 0:
                    results_counts['agent1_wins_first'] += 1
                else:
                    results_counts['agent1_wins_second'] += 1
            else:
                if first_player_idx == 0:
                    results_counts['agent2_wins_first'] += 1
                else:
                    results_counts['agent2_wins_second'] += 1

        elif len(parts) == 2 and parts[0] == "draw":
            first_player_idx = int(parts[1])
            if first_player_idx == 0:
                results_counts['draws_first'] += 1
            else:
                results_counts['draws_second'] += 1
        else:
            print(f"⚠️ Unexpected game result format: {result}")

                # ♻️ Retrain if needed
        
        if ml_lost:
            ml_losses += 1
            # if ml_losses % 20 == 0:
            #     print("♻️ Retraining model after 20 ML losses...")
            #     os.system("python ml/train_model.py") # Keep commented


    return (
    results_counts,
    aggregated_minimax_metrics,
    aggregated_agent_move_durations,
    win_type_outcomes,
    aggregated_heuristic_scores,
    aggregated_threats_faced,
    aggregated_threats_blocked,
    aggregated_memory_usage
    )


def plot_threats_faced_vs_blocked(threats_faced, threats_blocked, agent_labels, filename):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(agent_labels))
    width = 0.35

    faced_values = [threats_faced[i] for i in x]
    blocked_values = [threats_blocked[i] for i in x]

    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, faced_values, width, label='Threats Faced', color='orange')
    plt.bar(x + width/2, blocked_values, width, label='Threats Blocked', color='green')

    plt.xticks(x, agent_labels)
    plt.ylabel("Count")
    plt.title("Threats Faced vs Blocked")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_heuristic_score_distribution(heuristic_scores, agent_labels, filename):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for idx, scores in heuristic_scores.items():
        plt.hist(scores, bins=20, alpha=0.6, label=agent_labels[idx])
    plt.title("Heuristic Score Distribution")
    plt.xlabel("Heuristic Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



# --------------------------------------------------------------------------------------
# REVISED save_results function to display/save average move duration per agent
# --------------------------------------------------------------------------------------
# save_results now accepts 6 arguments: name1, name2, results_counts, aggregated_minimax_metrics, agent1, agent2
# We need to add aggregated_agent_move_durations as a 7th argument
def save_results(
    name1,
    name2,
    results_counts,
    aggregated_minimax_metrics,
    aggregated_agent_move_durations,
    win_type_outcomes,
    aggregated_heuristic_scores,
    aggregated_threats_faced,
    aggregated_threats_blocked,
    aggregated_memory_usage
):
    import csv
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter

    total_games = results_counts['total_games']
    games_starting_first = total_games / 2
    games_starting_second = total_games - games_starting_first

    agent1_win_rate_first = results_counts['agent1_wins_first'] / max(1, games_starting_first)
    agent1_win_rate_second = results_counts['agent1_wins_second'] / max(1, games_starting_second)
    agent2_win_rate_first = results_counts['agent2_wins_first'] / max(1, games_starting_first)
    agent2_win_rate_second = results_counts['agent2_wins_second'] / max(1, games_starting_second)

    draw_rate_first = results_counts['draws_first'] / max(1, games_starting_first)
    draw_rate_second = results_counts['draws_second'] / max(1, games_starting_second)

    print(f"\n--- {name1} vs {name2} Simulation Results ({total_games} games) ---")
    print(f"{name1} (First): Win Rate: {agent1_win_rate_first:.2%}, Draw Rate: {draw_rate_first:.2%}, Loss Rate: {agent2_win_rate_first:.2%}")
    print(f"{name1} (Second): Win Rate: {agent1_win_rate_second:.2%}, Draw Rate: {draw_rate_second:.2%}, Loss Rate: {agent2_win_rate_second:.2%}")
    print(f"{name2} (First): Win Rate: {agent2_win_rate_first:.2%}, Draw Rate: {draw_rate_first:.2%}, Loss Rate: {agent1_win_rate_first:.2%}")
    print(f"{name2} (Second): Win Rate: {agent2_win_rate_second:.2%}, Draw Rate: {draw_rate_second:.2%}, Loss Rate: {agent1_win_rate_second:.2%}")
    print("-" * 60)

    if aggregated_minimax_metrics['nodes_expanded']:
        print("\n--- Minimax Metrics ---")
        avg_nodes = np.mean(aggregated_minimax_metrics['nodes_expanded'])
        avg_depth = np.mean(aggregated_minimax_metrics['max_depth'])
        avg_branching = np.mean(aggregated_minimax_metrics['branching_factor'])
        total_expanded = sum(aggregated_minimax_metrics['nodes_expanded'])
        total_pruned = sum(aggregated_minimax_metrics['nodes_pruned'])
        pruning_efficiency = (total_pruned / max(1, total_expanded + total_pruned)) * 100
        print(f"Avg Nodes: {avg_nodes:.2f}, Avg Depth: {avg_depth:.2f}, Avg Branching: {avg_branching:.2f}")
        print(f"Total Expanded: {total_expanded}, Total Pruned: {total_pruned}, Pruning Effectiveness: {pruning_efficiency:.2f}%")

    print("\n--- Move Durations ---")
    avg_time1 = np.mean(aggregated_agent_move_durations[0]) if aggregated_agent_move_durations[0] else 0
    avg_time2 = np.mean(aggregated_agent_move_durations[1]) if aggregated_agent_move_durations[1] else 0
    print(f"{name1}: {avg_time1:.6f} sec | {name2}: {avg_time2:.6f} sec")

    print("\n--- Win Type Distribution ---")
    win_type_counter = Counter(win_type_outcomes)
    for k, v in win_type_counter.items():
        print(f"{k}: {v}")

    print("\n--- Memory Usage ---")
    avg_mem1 = np.mean(aggregated_memory_usage[0]) if aggregated_memory_usage[0] else 0
    avg_mem2 = np.mean(aggregated_memory_usage[1]) if aggregated_memory_usage[1] else 0
    print(f"{name1}: {avg_mem1:.2f} KB | {name2}: {avg_mem2:.2f} KB")

    print("\n--- Heuristic Scores ---")
    for i, name in enumerate([name1, name2]):
        scores = aggregated_heuristic_scores[i]
        if scores:
            print(f"{name}: Mean={np.mean(scores):.2f}, Min={np.min(scores):.2f}, Max={np.max(scores):.2f}, StdDev={np.std(scores):.2f}")
        else:
            print(f"{name}: No scores")

    print("\n--- Threat Counterplay ---")
    for i, name in enumerate([name1, name2]):
        faced = aggregated_threats_faced[i]
        blocked = aggregated_threats_blocked[i]
        block_rate = (blocked / max(1, faced)) * 100
        print(f"{name}: Faced={faced}, Blocked={blocked}, Block Rate={block_rate:.2f}%")

    # CSV Save
    csv_file = f"evaluation/results_{name1.lower()}_vs_{name2.lower()}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Outcome', 'Count'])
        writer.writerow([f"{name1} Wins (First)", results_counts['agent1_wins_first']])
        writer.writerow([f"{name1} Wins (Second)", results_counts['agent1_wins_second']])
        writer.writerow([f"{name2} Wins (First)", results_counts['agent2_wins_first']])
        writer.writerow([f"{name2} Wins (Second)", results_counts['agent2_wins_second']])
        writer.writerow([f"Draws ({name1} First)", results_counts['draws_first']])
        writer.writerow([f"Draws ({name2} First)", results_counts['draws_second']])

        writer.writerow([])
        writer.writerow(['Metric', 'Value'])
        writer.writerow([f"{name1} Avg Move Time (s)", avg_time1])
        writer.writerow([f"{name2} Avg Move Time (s)", avg_time2])
        writer.writerow([f"{name1} Avg Peak Memory (KB)", avg_mem1])
        writer.writerow([f"{name2} Avg Peak Memory (KB)", avg_mem2])

        for i, name in enumerate([name1, name2]):
            scores = aggregated_heuristic_scores[i]
            if scores:
                writer.writerow([f"{name} Heuristic Mean", np.mean(scores)])
                writer.writerow([f"{name} Heuristic Min", np.min(scores)])
                writer.writerow([f"{name} Heuristic Max", np.max(scores)])
                writer.writerow([f"{name} Heuristic StdDev", np.std(scores)])

        for i, name in enumerate([name1, name2]):
            faced = aggregated_threats_faced[i]
            blocked = aggregated_threats_blocked[i]
            block_rate = (blocked / max(1, faced)) * 100
            writer.writerow([f"{name} Threats Faced", faced])
            writer.writerow([f"{name} Threats Blocked", blocked])
            writer.writerow([f"{name} Threat Block Rate (%)", block_rate])

        if win_type_counter:
            writer.writerow([])
            writer.writerow(['Win Type', 'Count'])
            for k, v in win_type_counter.items():
                writer.writerow([k, v])
    # --- Win/Draw/Loss Rates by Turn Order ---
    turn_outcomes = {
        "First - Win": results_counts['agent1_wins_first'],
        "First - Loss": results_counts['agent2_wins_first'],
        "First - Draw": results_counts['draws_first'],
        "Second - Win": results_counts['agent1_wins_second'],
        "Second - Loss": results_counts['agent2_wins_second'],
        "Second - Draw": results_counts['draws_second'],
    }

    # Normalize to percentages
    total_first = results_counts['agent1_wins_first'] + results_counts['agent2_wins_first'] + results_counts['draws_first']
    total_second = results_counts['agent1_wins_second'] + results_counts['agent2_wins_second'] + results_counts['draws_second']

    rate_turn_outcomes = {
        "First - Win": (results_counts['agent1_wins_first'] / total_first * 100) if total_first else 0,
        "First - Loss": (results_counts['agent2_wins_first'] / total_first * 100) if total_first else 0,
        "First - Draw": (results_counts['draws_first'] / total_first * 100) if total_first else 0,
        "Second - Win": (results_counts['agent1_wins_second'] / total_second * 100) if total_second else 0,
        "Second - Loss": (results_counts['agent2_wins_second'] / total_second * 100) if total_second else 0,
        "Second - Draw": (results_counts['draws_second'] / total_second * 100) if total_second else 0,
    }

    print(f"📁 CSV saved to {csv_file}")

    # Visualizations
    def plot_bar(data, title, path, ylabel="Count", color='orchid'):
        plt.figure(figsize=(8, 5))
        keys, values = zip(*data.items())
        bars = plt.bar(keys, values, color=color)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(int(height)), ha='center')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"📊 Plot saved to {path}")

    # Win Types
    if win_type_counter:
        plot_bar(win_type_counter, f"{name1} vs {name2} Win Types", f"evaluation/results_{name1.lower()}_vs_{name2.lower()}_win_types.png")

    # Heuristic Distribution
    for i, name in enumerate([name1, name2]):
        scores = aggregated_heuristic_scores[i]
        if scores:
            plt.figure(figsize=(8, 5))
            plt.hist(scores, bins=20, color='mediumseagreen', edgecolor='black')
            plt.title(f"{name} Heuristic Score Distribution")
            plt.xlabel("Heuristic Score")
            plt.ylabel("Frequency")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"evaluation/results_{name.lower()}_heuristic_distribution.png")
            plt.close()
            print(f"📊 Saved heuristic plot for {name}")

    # Threats Faced vs Blocked
    labels = [name1, name2]
    faced = [aggregated_threats_faced[0], aggregated_threats_faced[1]]
    blocked = [aggregated_threats_blocked[0], aggregated_threats_blocked[1]]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, faced, width, label='Threats Faced', color='coral')
    plt.bar(x + width/2, blocked, width, label='Threats Blocked', color='steelblue')
    plt.xticks(x, labels)
    plt.ylabel("Count")
    plt.title("Threats Faced vs Blocked")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"evaluation/results_{name1.lower()}_vs_{name2.lower()}_threats.png")
    plt.close()
    print(f"📊 Saved threat analysis plot")

        
    # --- Turn Order Outcome Rates Plot ---
    labels = list(rate_turn_outcomes.keys())
    values = list(rate_turn_outcomes.values())

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=['mediumseagreen', 'salmon', 'lightgray', 'mediumseagreen', 'salmon', 'lightgray'])
    plt.ylabel("Percentage (%)")
    plt.title(f"Outcome Rates by Turn Order: {name1} vs {name2}")
    plt.ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1, f"{height:.1f}%", ha='center')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    turn_plot_path = f"evaluation/results_{name1.lower()}_vs_{name2.lower()}_turn_order_outcomes.png"
    plt.savefig(turn_plot_path)
    plt.close()
    print(f"📊 Saved turn order outcome plot to {turn_plot_path}")

        # --- Outcome Summary Plot (Wins/Draws by Role & Turn) ---
    outcome_data = {
        f"{name1} Wins (First)": results_counts['agent1_wins_first'],
        f"{name1} Wins (Second)": results_counts['agent1_wins_second'],
        f"{name2} Wins (First)": results_counts['agent2_wins_first'],
        f"{name2} Wins (Second)": results_counts['agent2_wins_second'],
        f"Draws ({name1} First)": results_counts['draws_first'],
        f"Draws ({name2} First)": results_counts['draws_second'],
    }

    # Plot outcome bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(outcome_data.keys(), outcome_data.values(), color='mediumorchid')
    plt.title(f"Game Outcomes: {name1} vs {name2}")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate values on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, str(int(height)), ha='center', fontsize=9)

    plt.tight_layout()
    plot_path = f"evaluation/results_{name1.lower()}_vs_{name2.lower()}_outcome_summary.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Saved outcome summary plot: {plot_path}")




# --------------------------------------------------------------------------------------
# REVISED run_selected_vs_all to correctly unpack results and pass to save_results
# --------------------------------------------------------------------------------------
# run_selected_vs_all now unpacks 3 return values from run_simulations
# and passes 7 arguments to save_results
def run_selected_vs_all(n=500):
    agents = {
        "Random": random_move_wrapped,
        "Smart": smart_move,
        "Minimax": minimax_move,
        "ML": ml_move
    }

    agent_names = list(agents.keys())

    print("Simulation Mode:")
    print("1. Choose one agent vs one other")
    print("2. Choose one agent vs all others")
    print("3. Run all vs all matchups")
    mode = input("Enter 1, 2 or 3: ").strip()

    if mode == "3":
        for i, name1 in enumerate(agent_names):
            for j in range(i + 1, len(agent_names)):
                name2 = agent_names[j]

                print(f"\nSimulating: {name1} vs {name2} ({n} games)")
                results_counts, aggregated_minimax_metrics, aggregated_agent_move_durations, win_type_outcomes, aggregated_heuristic_scores, aggregated_threats_faced, aggregated_threats_blocked, aggregated_memory_usage = run_simulations(agents[name1], agents[name2], n)

                save_results(
                    name1,
                    name2,
                    results_counts,
                    aggregated_minimax_metrics,
                    aggregated_agent_move_durations,
                    win_type_outcomes,
                    aggregated_heuristic_scores,
                    aggregated_threats_faced,
                    aggregated_threats_blocked,
                    aggregated_memory_usage
                )

        return

    print("\nSelect the primary agent:")
    for i, name in enumerate(agent_names, 1):
        print(f"{i}. {name}")
    try:
        idx1 = int(input("Enter number: ")) - 1
        name1 = agent_names[idx1]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    if mode == "2":
        for name2 in agent_names:
            if name1 != name2:
                print(f"\n▶️ Simulating: {name1} vs {name2} ({n} games)")
                results_counts, aggregated_minimax_metrics, aggregated_agent_move_durations, win_type_outcomes, aggregated_heuristic_scores, aggregated_threats_faced, aggregated_threats_blocked, aggregated_memory_usage = run_simulations(agents[name1], agents[name2], n)

                save_results(
                    name1,
                    name2,
                    results_counts,
                    aggregated_minimax_metrics,
                    aggregated_agent_move_durations,
                    win_type_outcomes,
                    aggregated_heuristic_scores,
                    aggregated_threats_faced,
                    aggregated_threats_blocked,
                    aggregated_memory_usage
                )
                return

    elif mode == "1":
        print("\nSelect the opponent agent:")
        opponents = [name for name in agent_names if name != name1]
        for i, name in enumerate(opponents, 1):
            print(f"{i}. {name}")
        try:
            idx2 = int(input("Enter number: ")) - 1
            name2 = opponents[idx2]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return

        print(f"\n▶️ Simulating: {name1} vs {name2} ({n} games)")
        results_counts, aggregated_minimax_metrics, aggregated_agent_move_durations, win_type_outcomes, aggregated_heuristic_scores, aggregated_threats_faced, aggregated_threats_blocked, aggregated_memory_usage = run_simulations(agents[name1], agents[name2], n)

        save_results(
            name1,
            name2,
            results_counts,
            aggregated_minimax_metrics,
            aggregated_agent_move_durations,
            win_type_outcomes,
            aggregated_heuristic_scores,
            aggregated_threats_faced,
            aggregated_threats_blocked,
            aggregated_memory_usage
        )
# Run the main simulation function
if __name__ == "__main__":
    run_selected_vs_all(n=500)

# ----------------------------------------------
# 💡 Optional: Single matchup mode — enable if needed
# ----------------------------------------------
# if __name__ == "__main__":
#     results = run_simulations(agent1=ml_move, agent2=minimax_move, n=100)
#     for k, v in results.items():
#         print(f"{k}: {v}")
#
#     categories = [
#         "ML Wins (First)", "ML Wins (Second)",
#         "Minimax Wins (First)", "Minimax Wins (Second)",
#         "Draws (ML First)", "Draws (Minimax First)"
#     ]
#     values = [
#         results['agent1_wins_first'],
#         results['agent1_wins_second'],
#         results['agent2_wins_first'],
#         results['agent2_wins_second'],
#         results['draws_first'],
#         results['draws_second']1
#     ]
#
#     plt.figure(figsize=(10, 6))
#     plt.bar(categories, values)
#     plt.title("ML vs Minimax Outcomes")
#     plt.ylabel("Games")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.savefig("evaluation/game_results_ml_vs_minimax.png")
#     plt.show()
#
#     with open("evaluation/game_results_ml_vs_minimax.csv", 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Outcome', 'Count'])
#         for label, value in zip(categories, values):
#             writer.writerow([label, value])
#     print("✅ Results saved to CSV and PNG.")
