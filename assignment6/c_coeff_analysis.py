from load_data import load_data
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = load_data()
df = df.select_dtypes(include='number').dropna()

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

ridge.fit(X_scaled, y)
lasso.fit(X_scaled, y)

# Coefficients
ridge_coef = pd.Series(ridge.coef_, index=X.columns)
lasso_coef = pd.Series(lasso.coef_, index=X.columns)

# -------------------------
# PLOT
# -------------------------
plt.figure(figsize=(12,6))
ridge_coef.plot(label="Ridge")
lasso_coef.plot(label="Lasso")
plt.legend()
plt.title("Coefficient Comparison")
plt.show()

# -------------------------
# FEATURE ELIMINATION
# -------------------------
zero_features = (lasso_coef == 0).sum()

print("Features eliminated by Lasso:", zero_features)