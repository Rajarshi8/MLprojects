import pandas as pd

def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    
    df = pd.read_csv(url, header=None)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y