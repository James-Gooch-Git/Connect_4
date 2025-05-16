import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_move_model(csv_path="ml/move_data_from_replay.csv", model_path="ml/move_model_from_replay.joblib"):
    df = pd.read_csv(csv_path)
    X = df.drop(["best_move"], axis=1)
    y = df["best_move"]
 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train, sample_weight=w_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Weighted accuracy on move prediction: {acc:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Weighted move prediction model saved to: {model_path}")

if __name__ == "__main__":
    train_move_model()