import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from simulate_games import run_simulations, random_move_wrapped, smart_move, minimax_move, ml_move

agents = {
    "Random": random_move_wrapped,
    "Smart": smart_move,
    "Minimax": minimax_move,
    "ML": ml_move
}

def run_simulation():
    agent1_name = agent1_var.get()
    agent2_name = agent2_var.get()
    num_games = int(num_games_var.get())

    if agent1_name == agent2_name:
        messagebox.showerror("Selection Error", "Choose two different agents!")
        return

    results = run_simulations(agents[agent1_name], agents[agent2_name], n=num_games)

    categories = [
        f"{agent1_name} Wins (First)",
        f"{agent1_name} Wins (Second)",
        f"{agent2_name} Wins (First)",
        f"{agent2_name} Wins (Second)",
        f"Draws ({agent1_name} First)",
        f"Draws ({agent2_name} First)"
    ]
    values = [
        results['agent1_wins_first'],
        results['agent1_wins_second'],
        results['agent2_wins_first'],
        results['agent2_wins_second'],
        results['draws_first'],
        results['draws_second']
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(categories, values, color="skyblue")
    ax.set_ylabel("Games")
    ax.set_title("Simulation Results")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # If there's a previous chart, clear it
    for widget in chart_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Tkinter Window
root = tk.Tk()
root.title("Connect 4 Agent Simulation Dashboard")

tk.Label(root, text="Agent 1:").grid(row=0, column=0, padx=5, pady=5)
tk.Label(root, text="Agent 2:").grid(row=1, column=0, padx=5, pady=5)
tk.Label(root, text="Number of games:").grid(row=2, column=0, padx=5, pady=5)

agent1_var = tk.StringVar(value="Random")
agent2_var = tk.StringVar(value="Smart")
num_games_var = tk.StringVar(value="100")

agent1_menu = ttk.Combobox(root, textvariable=agent1_var, values=list(agents.keys()), state="readonly")
agent1_menu.grid(row=0, column=1, padx=5, pady=5)

agent2_menu = ttk.Combobox(root, textvariable=agent2_var, values=list(agents.keys()), state="readonly")
agent2_menu.grid(row=1, column=1, padx=5, pady=5)

num_games_entry = ttk.Entry(root, textvariable=num_games_var, width=8)
num_games_entry.grid(row=2, column=1, padx=5, pady=5)

run_btn = ttk.Button(root, text="Run Simulation", command=run_simulation)
run_btn.grid(row=3, column=0, columnspan=2, pady=10)

chart_frame = tk.Frame(root)
chart_frame.grid(row=4, column=0, columnspan=2, pady=10)

root.mainloop()
