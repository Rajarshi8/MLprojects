import pandas as pd
import os

def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "iris.csv")

    df = pd.read_csv(file_path)

    X = df.drop("species", axis=1)
    y = df["species"]

    return X, y