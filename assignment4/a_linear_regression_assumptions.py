from load_data import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro

df = load_data()

# -------------------------
# FEATURES & TARGET
# -------------------------
X = df.drop("target", axis=1)
y = df["target"]

# -------------------------
# LINEAR REGRESSION
# -------------------------
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
residuals = y - y_pred

# -------------------------
# VIF (Multicollinearity)
# -------------------------
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(X.shape[1])]

print("\nVIF Values:\n", vif_data)

# -------------------------
# RESIDUAL PLOT
# -------------------------
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# -------------------------
# HISTOGRAM OF RESIDUALS
# -------------------------
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()

# -------------------------
# NORMALITY TEST (Shapiro)
# -------------------------
stat, p = shapiro(residuals.sample(5000))  # sample for speed

print("\nShapiro Test:")
print("Statistic:", stat)
print("p-value:", p)

if p > 0.05:
    print("Residuals are normally distributed")
else:
    print("Residuals are NOT normally distributed")