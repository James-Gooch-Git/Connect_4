import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import datetime
import json
import sys

# Make sure we can import flip_perspective from the agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.ml_agent import flip_perspective

# --- Load Dataset ---
data_path = "ml/connect-4.data"  # Must exist, previously decompressed
print(f"📂 Loading dataset from: {data_path}")
df_raw = pd.read_csv(data_path, header=None)

# --- Assign Features & Target ---
X = df_raw.iloc[:, :-1]
X.columns = [f"pos_{i}" for i in range(42)]
y = df_raw.iloc[:, -1].values

# --- Flip Board Perspective So 'x' is always the ML player ---
mask_flip = (y == 'win') | (y == 'loss')
X_flipped = X.loc[mask_flip].replace({'x': 'o', 'o': 'x'})
X.loc[mask_flip] = X_flipped
y = ['win' if label == 'loss' else 'loss' if label == 'win' else label for label in y]

# --- Convert to Numeric Format ---
X.replace({'x': 1, 'o': -1, 'b': 0}, inplace=True)
X = X.infer_objects(copy=False)
df = X.copy()
df["outcome"] = y

# --- Load Replay Buffer if Available ---
replay_file = os.path.join("ml", "replay_buffer.json")
if os.path.exists(replay_file):
    print("📂 Found replay buffer, loading...")
    with open(replay_file, "r") as f:
        replay_data = json.load(f)

    replay_X = []
    replay_Y = []

    for game in replay_data:
        outcome = game.get("outcome", "draw")
        for state in game.get("states", []):
            if isinstance(state, list) and len(state) == 42:
                flipped = flip_perspective(state)
                replay_X.append(flipped)
                replay_Y.append(outcome)

    if replay_X:
        df_replay = pd.DataFrame(replay_X, columns=[f"pos_{i}" for i in range(42)])
        df_replay["outcome"] = replay_Y
        df = pd.concat([df, df_replay], ignore_index=True)
        print(f"✅ Appended {len(replay_X)} replay states.")
    else:
        print("⚠️ Replay buffer was empty or invalid.")

# --- Train/Test Split ---
X = df.drop("outcome", axis=1)
y = df["outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Accuracy on test set: {acc:.4f}")

# --- Save Model ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("ml", exist_ok=True)

versioned_path = f"ml/model_{timestamp}.joblib"
latest_path = "ml/model_latest.joblib"

joblib.dump(model, versioned_path)
joblib.dump(model, latest_path)

print(f"💾 Model saved to:\n- {versioned_path}\n- {latest_path} (latest)")