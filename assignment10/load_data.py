from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data():
    data = load_digits()

    X = data.data.astype(float)
    y = data.target.astype(int)

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # One-hot encoding
    num_classes = len(np.unique(y))
    y_onehot = np.zeros((y.size, num_classes))
    y_onehot[np.arange(y.size), y] = 1

    return train_test_split(X, y, y_onehot, test_size=0.2, random_state=42)