import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import json
import datetime
import sys

# Allow import of flip_perspective
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.ml_agent import flip_perspective

# --- Load UCI dataset
uci_path = "ml/connect-4.data"
print(f"📂 Loading UCI dataset from {uci_path}")
df_uci = pd.read_csv(uci_path, header=None)

X = df_uci.iloc[:, :-1]
X.columns = [f"pos_{i}" for i in range(42)]
y = df_uci.iloc[:, -1].values

# Normalize UCI perspective so 'x' is always ML
mask = (y == 'win') | (y == 'loss')
X.loc[mask] = X.loc[mask].replace({'x': 'o', 'o': 'x'})
y = ['win' if label == 'loss' else 'loss' if label == 'win' else label for label in y]
X.replace({'x': 1, 'o': -1, 'b': 0}, inplace=True)
uci_df = X.copy()
uci_df["outcome"] = y

# --- Load AI-generated games
generated_path = "ml/full_training_data.json"
if os.path.exists(generated_path):
    print(f"📂 Loading generated data from {generated_path}")
    with open(generated_path, "r") as f:
        game_data = json.load(f)

    gen_X, gen_Y = [], []
    for game in game_data:
        if game["outcome"] == "neutral":
            continue  # Optional: ignore non-ML matchups
        for state in game["states"]:
            if isinstance(state, list) and len(state) == 42:
                gen_X.append(state)
                gen_Y.append(game["outcome"])

    if gen_X:
        df_gen = pd.DataFrame(gen_X, columns=[f"pos_{i}" for i in range(42)])
        df_gen["outcome"] = gen_Y
        print(f"✅ Loaded {len(gen_X)} board states from generated games")
        df_full = pd.concat([uci_df, df_gen], ignore_index=True)
    else:
        print("⚠️ No valid states found in generated dataset.")
        df_full = uci_df
else:
    print("⚠️ No generated dataset found, training on UCI only.")
    df_full = uci_df

# --- Train/test split
X = df_full.drop("outcome", axis=1)
y = df_full["outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Accuracy on test set: {acc:.4f}")

# --- Save model
os.makedirs("ml", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
versioned_path = f"ml/model_{timestamp}.joblib"
latest_path = "ml/model_PLEASE.joblib"

joblib.dump(model, versioned_path)
joblib.dump(model, latest_path)
print(f"💾 Model saved to:\n- {versioned_path}\n- {latest_path} (latest)")