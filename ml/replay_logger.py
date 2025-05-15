# ml/replay_logger.py
import os, json

# def save_game(states, outcome, filename="ml/replay_buffer.json"):
#     os.makedirs("ml", exist_ok=True)
#     if os.path.exists(filename):
#         with open(filename, "r") as f:
#             existing = json.load(f)
#     else:
#         existing = []
#     entry = {"states": states, "outcome": outcome}
#     existing.append(entry)
#     with open(filename, "w") as f:
#         json.dump(existing, f)

def save_failed_game(states, outcome="loss", filename="ml/replay_buffer.json"):
    os.makedirs("ml", exist_ok=True)
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    # Now save as object with both states and outcome
    existing.append({"states": states, "outcome": outcome})

    with open(filename, "w") as f:
        json.dump(existing, f)
    print(f"📝 Logged {len(states)} board states with outcome={outcome} to replay buffer.")
