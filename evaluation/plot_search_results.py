# evaluation/plot_search_metrics.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv # Import the csv module for manual parsing

def plot_minimax_search_metrics(opponent_names, minimax_agent_name="Minimax", evaluation_dir="evaluation"):
    """
    Reads average search metrics for Minimax against different opponents
    from CSV files and generates bar charts.

    Args:
        opponent_names (list): A list of strings with the names of the opponents Minimax played against.
        minimax_agent_name (str): The name used for the Minimax agent (case-insensitive match for filename).
        evaluation_dir (str): The directory where simulation results CSVs are saved.
    """
    metrics_data = {
        'Opponent': [],
        'Avg Nodes Expanded': [],
        'Avg Max Depth': [],
        'Avg Branching Factor': []
    }

    # Define the exact strings expected in the 'Metric' column of the CSV
    # These must match exactly what save_results writes
    metric_names_in_csv = {
        'Avg Nodes Expanded': 'Minimax Avg Nodes Expanded per move',
        'Avg Max Depth': 'Minimax Avg Max Depth per move',
        'Avg Branching Factor': 'Minimax Avg Branching Factor per move'
    }


    for opp_name in opponent_names:
        # Construct the expected filename for the matchup (handling both naming orders)
        # Based on save_results, the filename is results_[name1.lower()]_vs_[name2.lower()].csv
        # Let's assume Minimax is usually name1 or name2 based on selection order
        filename_option1 = os.path.join(evaluation_dir, f"results_{minimax_agent_name.lower()}_vs_{opp_name.lower()}.csv")
        filename_option2 = os.path.join(evaluation_dir, f"results_{opp_name.lower()}_vs_{minimax_agent_name.lower()}.csv")

        csv_file = None
        if os.path.exists(filename_option1):
            csv_file = filename_option1
        elif os.path.exists(filename_option2):
            csv_file = filename_option2
        else:
            print(f"Warning: Could not find results CSV for {minimax_agent_name} vs {opp_name}. Looked for {filename_option1} and {filename_option2}")
            continue # Skip this opponent if no data file is found

        print(f"Processing: {csv_file}") # Debug print


        # --- Manual CSV reading to find metrics ---
        found_metrics_section = False
        current_game_metrics = {} # Store metrics for the current game before appending

        try:
            with open(csv_file, 'r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Skip empty rows
                    if not row:
                        continue

                    # Check if this is the header row for the metrics section
                    # Use strip() to handle potential leading/trailing whitespace
                    if row[0].strip() == 'Metric' and (len(row) > 1 and row[1].strip() == 'Value'):
                         found_metrics_section = True
                         continue # Skip the header row itself

                    # If we are in the metrics section, extract the data
                    if found_metrics_section:
                        # Ensure the row has at least two columns before accessing index 1
                        if len(row) >= 2:
                            metric_name = row[0].strip()
                            metric_value_str = row[1].strip()
                            try:
                                metric_value = float(metric_value_str)
                                current_game_metrics[metric_name] = metric_value
                            except ValueError:
                                 print(f"Warning: Could not convert value to float for metric '{metric_name}' in {csv_file}. Value: '{metric_value_str}'")
                            except Exception as e:
                                 print(f"Error processing metric row '{row}' in {csv_file}: {e}")
                        # Optionally handle rows in the metrics section with less than 2 columns if they can occur
                        # else:
                        #      print(f"Warning: Skipping row in metrics section with less than 2 columns: {row} in {csv_file}")


        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_file}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred reading {csv_file}: {e}")
            continue # Skip this file due to read error


        # Check if we found all expected metrics for this game
        # Use .get() with None and check for None to handle missing keys gracefully
        avg_nodes = current_game_metrics.get(metric_names_in_csv['Avg Nodes Expanded'])
        avg_depth = current_game_metrics.get(metric_names_in_csv['Avg Max Depth'])
        avg_branching = current_game_metrics.get(metric_names_in_csv['Avg Branching Factor'])


        if avg_nodes is not None and avg_depth is not None and avg_branching is not None:
             metrics_data['Opponent'].append(opp_name)
             metrics_data['Avg Nodes Expanded'].append(avg_nodes) # Values are already float
             metrics_data['Avg Max Depth'].append(avg_depth)
             metrics_data['Avg Branching Factor'].append(avg_branching)
             print(f"Successfully extracted metrics for {minimax_agent_name} vs {opp_name}") # Debug print
        else:
             print(f"Warning: Not all expected search metrics found in {csv_file}.") # Debug print
             print(f"Expected: {list(metric_names_in_csv.values())}")
             print(f"Found keys: {list(current_game_metrics.keys())}")
             print("Please ensure Minimax played and its search metrics were recorded in this file.")


    # Convert to DataFrame for easier plotting
    metrics_df = pd.DataFrame(metrics_data)

    if metrics_df.empty:
        print("No Minimax search metrics data found across the specified opponents to plot.")
        return

    # Sort by opponent name for consistent plotting order
    metrics_df = metrics_df.sort_values('Opponent')

    # --- Create Plots ---

    # Plot 1: Average Nodes Expanded
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_df['Opponent'], metrics_df['Avg Nodes Expanded'], color='skyblue')
    plt.title(f'Average Nodes Expanded per move for {minimax_agent_name}')
    plt.ylabel('Average Nodes Expanded')
    plt.xlabel('Opponent')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_file_nodes = os.path.join(evaluation_dir, f'plot_{minimax_agent_name.lower()}_avg_nodes_expanded.png')
    plt.savefig(plot_file_nodes)
    plt.close()
    print(f"📊 Saved plot: {plot_file_nodes}")

    # Plot 2: Average Max Depth
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_df['Opponent'], metrics_df['Avg Max Depth'], color='lightcoral')
    plt.title(f'Average Max Depth per move for {minimax_agent_name}')
    plt.ylabel('Average Max Depth')
    plt.xlabel('Opponent')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_file_depth = os.path.join(evaluation_dir, f'plot_{minimax_agent_name.lower()}_avg_max_depth.png')
    plt.savefig(plot_file_depth)
    plt.close()
    print(f"📊 Saved plot: {plot_file_depth}")


    # Plot 3: Average Branching Factor
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_df['Opponent'], metrics_df['Avg Branching Factor'], color='lightgreen')
    plt.title(f'Average Branching Factor per move for {minimax_agent_name}')
    plt.ylabel('Average Branching Factor')
    plt.xlabel('Opponent')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_file_branching = os.path.join(evaluation_dir, f'plot_{minimax_agent_name.lower()}_avg_branching_factor.png')
    plt.savefig(plot_file_branching)
    plt.close()
    print(f"📊 Saved plot: {plot_file_branching}")


if __name__ == "__main__":
    # Define the names of the opponents Minimax played against
    # Make sure these names match the keys in the agents dictionary in simulate_games.py
    # and that the corresponding CSV files exist and contain the metrics.
    opponent_agent_names = ["Random", "Smart", "ML"] # Assuming you want to compare vs these 3

    # Call the plotting function
    plot_minimax_search_metrics(opponent_agent_names)

    # Example if you only want to plot vs a subset:
    # plot_minimax_search_metrics(["Random", "Smart"])