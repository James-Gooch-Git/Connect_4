from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import datetime 
import json

# ✅ Load from UCI
connect_4 = fetch_ucirepo(id=26)
X = connect_4.data.features.copy()
X.columns = [f"pos_{i}" for i in range(42)]

y = connect_4.data.targets.values.ravel()

# ✅ Convert board state
X.replace({'x': 1, 'o': -1, 'b': 0}, inplace=True)
X = X.infer_objects(copy=False)

df = X.copy()
df["outcome"] = y

# Load replay data
replay_file = "ml/replay_buffer.json"
if os.path.exists(replay_file):
    with open(replay_file, "r") as f:
        replay_data = json.load(f)
    # Flatten states and outcomes
    replay_X = [state for game in replay_data for state in game["states"]]
    replay_Y = [game["outcome"] for game in replay_data for _ in game["states"]]
    df_replay = pd.DataFrame(replay_X, columns=[f"pos_{i}" for i in range(42)])
    df_replay["outcome"] = replay_Y

    # Append to main dataset
    df_full = pd.concat([df, df_replay], ignore_index=True)
else:
    df_full = df

# ✅ Train/test split
X = df_full.drop("outcome", axis=1)
y = df_full["outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Evaluate
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Accuracy on test set: {acc:.4f}")

# ✅ Save model
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"ml/model_{timestamp}.joblib"
joblib.dump(model, model_filename)
print(f"💾 Model saved to: {model_filename}")

# Optionally, still overwrite the default for "current"
joblib.dump(model, "ml/model_latest.joblib")
