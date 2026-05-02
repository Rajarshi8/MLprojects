from load_data import load_data
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = load_data()
df = df.select_dtypes(include='number').dropna()

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------
# RIDGE TUNING
# -------------------------
ridge_params = {"alpha": [0.01, 0.1, 1, 10, 100]}

ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5)
ridge_grid.fit(X_train, y_train)

print("Best Ridge alpha:", ridge_grid.best_params_)

# -------------------------
# LASSO TUNING
# -------------------------
lasso_params = {"alpha": [0.001, 0.01, 0.1, 1, 10]}

lasso_grid = GridSearchCV(Lasso(max_iter=5000), lasso_params, cv=5)
lasso_grid.fit(X_train, y_train)

print("Best Lasso alpha:", lasso_grid.best_params_)