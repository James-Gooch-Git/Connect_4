import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulate_games import simulate_game, smart_move, minimax_move, ml_move, random_move_wrapped

# Disable replay logging during evaluation
import agents.ml_agent
def dummy_save_failed_game(*args, **kwargs):
    pass
agents.ml_agent.save_failed_game = dummy_save_failed_game

agents_dict = {
    "Random": random_move_wrapped,
    "Smart": smart_move,
    "Minimax": minimax_move,
    "ML": ml_move
}

def run_test_vs_agent(ml_agent, opponent_name, agents, n=100):
    agent = agents[opponent_name]
    ml_first_wins = draws_first = 0
    for _ in range(n):
        result, _, _ = simulate_game(ml_agent, agent, False)
        if result.startswith("0-wins-0"):
            ml_first_wins += 1
        elif result.startswith("draw-0"):
            draws_first += 1

    ml_second_wins = draws_second = 0
    for _ in range(n):
        result, _, _ = simulate_game(agent, ml_agent)
        if result.startswith("1-wins-1"):
            ml_second_wins += 1
        elif result.startswith("draw-1"):
            draws_second += 1

    print(f"\nML Agent vs {opponent_name} ({n} games each way):")
    print(f"ML as first: {ml_first_wins} wins, {draws_first} draws, win rate: {ml_first_wins/n:.2f}")
    print(f"ML as second: {ml_second_wins} wins, {draws_second} draws, win rate: {ml_second_wins/n:.2f}")

def run_test_vs_all(ml_agent, agents, n=100):
    for name in agents:
        if agents[name] == ml_agent:
            continue
        run_test_vs_agent(ml_agent, name, agents, n)

if __name__ == "__main__":
    n_games = 100
    agent_names = [name for name in agents_dict if name != "ML"]

    print("Choose an agent to play against the ML agent:")
    for i, name in enumerate(agent_names, 1):
        print(f"{i}. {name}")
    print(f"{len(agent_names)+1}. All agents")

    try:
        choice = int(input("Enter your choice (number): "))
    except Exception:
        print("Invalid input.")
        sys.exit(1)

    if 1 <= choice <= len(agent_names):
        opponent_name = agent_names[choice-1]
        run_test_vs_agent(ml_move, opponent_name, agents_dict, n=n_games)
    elif choice == len(agent_names) + 1:
        run_test_vs_all(ml_move, agents_dict, n=n_games)
    else:
        print("Invalid choice.")
