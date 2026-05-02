from load_data import load_energy_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

df = load_energy_data()

X = df.drop("Heating_Load", axis=1)
y = df["Heating_Load"]

degrees = [1, 2, 3, 4, 5]

for d in degrees:
    if d == 1:
        model = LinearRegression()
        X_input = X
    else:
        poly = PolynomialFeatures(degree=d)
        X_input = poly.fit_transform(X)
        model = LinearRegression()

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_input, y, cv=5, scoring="neg_mean_squared_error"
    )

    train_error = -train_scores.mean(axis=1)
    val_error = -val_scores.mean(axis=1)

    plt.plot(train_sizes, train_error, label=f"Train d={d}")
    plt.plot(train_sizes, val_error, linestyle="--", label=f"Val d={d}")

plt.xlabel("Training Size")
plt.ylabel("Error (MSE)")
plt.title("Learning Curves")
plt.legend()
plt.show()