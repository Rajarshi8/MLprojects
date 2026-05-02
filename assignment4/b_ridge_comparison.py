from load_data import load_data
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

df = load_data()

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# LINEAR REGRESSION
# -------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_pred)

# -------------------------
# RIDGE REGRESSION
# -------------------------
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)

# -------------------------
# RESULTS
# -------------------------
print("\nLinear Regression MSE:", lr_mse)
print("Ridge Regression MSE:", ridge_mse)

if ridge_mse < lr_mse:
    print("Ridge performs better (less overfitting)")
else:
    print("Linear Regression performs better")