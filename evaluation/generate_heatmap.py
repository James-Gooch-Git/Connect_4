import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from simulate_games import simulate_game, random_move_wrapped, smart_move, minimax_move, ml_move

def evaluate_agents_heatmap(n=50):
    agents = {
        "Random": random_move_wrapped,
        "Smart": smart_move,
        "Minimax": minimax_move,
        "ML": ml_move
    }
    names = list(agents.keys())
    win_first = pd.DataFrame(np.zeros((len(names), len(names))), index=names, columns=names)
    win_second = pd.DataFrame(np.zeros((len(names), len(names))), index=names, columns=names)
    draw_first = pd.DataFrame(np.zeros((len(names), len(names))), index=names, columns=names)
    draw_second = pd.DataFrame(np.zeros((len(names), len(names))), index=names, columns=names)
    avg_len_first = pd.DataFrame(np.zeros((len(names), len(names))), index=names, columns=names)
    avg_len_second = pd.DataFrame(np.zeros((len(names), len(names))), index=names, columns=names)

    progress = tqdm(total=len(names)*len(names), desc="Simulating agent matchups")

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if name1 == name2:
                for mat in [win_first, win_second, draw_first, draw_second, avg_len_first, avg_len_second]:
                    mat.iloc[i, j] = np.nan
                progress.update(1)
                continue

            wins_first = draws_first = total_len_first = 0
            wins_second = draws_second = total_len_second = 0

            # Agent1 as first
            for _ in range(n):
                result, _, moves = simulate_game(agents[name1], agents[name2])
                if result.startswith("0-wins-0"):
                    wins_first += 1
                elif result.startswith("draw-0"):
                    draws_first += 1
                total_len_first += moves

            # Agent1 as second (so agent2 is first)
            for _ in range(n):
                result, _, moves = simulate_game(agents[name1], agents[name2])
                if result.startswith("0-wins-1"):
                    wins_second += 1
                elif result.startswith("draw-1"):
                    draws_second += 1
                total_len_second += moves

            win_first.iloc[i, j] = wins_first / n
            win_second.iloc[i, j] = wins_second / n
            draw_first.iloc[i, j] = draws_first / n
            draw_second.iloc[i, j] = draws_second / n
            avg_len_first.iloc[i, j] = total_len_first / n
            avg_len_second.iloc[i, j] = total_len_second / n

            progress.update(1)

    progress.close()
    return win_first, win_second, draw_first, draw_second, avg_len_first, avg_len_second

def save_heatmap(matrix, title, filename, cmap="Blues"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap=cmap, linewidths=0.5, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_csv(matrix, filename):
    matrix.to_csv(filename)

if __name__ == "__main__":
    os.makedirs("evaluation", exist_ok=True)
    n_games = 50  # Change this for more robust stats!

    (win_first, win_second,
     draw_first, draw_second,
     avg_len_first, avg_len_second) = evaluate_agents_heatmap(n=n_games)

    heatmaps = [
        (win_first, "Win Rate (First Player)", "evaluation/winrate_first_heatmap.png", "Blues"),
        (win_second, "Win Rate (Second Player)", "evaluation/winrate_second_heatmap.png", "Purples"),
        (draw_first, "Draw Rate (First Player)", "evaluation/drawrate_first_heatmap.png", "Greens"),
        (draw_second, "Draw Rate (Second Player)", "evaluation/drawrate_second_heatmap.png", "Oranges"),
        (avg_len_first, "Average Game Length (First Player)", "evaluation/avglength_first_heatmap.png", "YlGnBu"),
        (avg_len_second, "Average Game Length (Second Player)", "evaluation/avglength_second_heatmap.png", "YlOrRd"),
    ]

    for matrix, title, filename, cmap in heatmaps:
        save_heatmap(matrix, title, filename, cmap)
        csv_filename = filename.replace("_heatmap.png", ".csv")
        save_csv(matrix, csv_filename)

    print("All heatmaps and CSVs saved in the 'evaluation' folder.")
