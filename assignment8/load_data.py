import pandas as pd
import os

def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "bank-additional-full.csv")

    df = pd.read_csv(file_path, sep=';')

    # Convert target
    df['y'] = df['y'].map({'yes': 1, 'no': 0})

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('y', axis=1)
    y = df['y']

    return X, y