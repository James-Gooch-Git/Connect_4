import pandas as pd

def load_data(path="ml/connect-4.data"):

    col_names = [f"pos_{i}" for i in range(42)] + ["outcome"]
    df = pd.read_csv(path, names=col_names)
    df.replace({'x': 1, 'o': -1, 'b': 0}, inplace=True)
    df = df.infer_objects(copy=False)
    print("🔍 Outcome distribution:\n", df["outcome"].value_counts())

    return df
