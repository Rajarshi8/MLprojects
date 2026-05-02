from load_data import load_energy_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

df = load_energy_data()

# Features & target
X = df.drop("Heating_Load", axis=1)
y = df["Heating_Load"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = []

# -------------------------
# Linear Regression
# -------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

results.append([
    "Linear",
    mean_squared_error(y_test, y_pred),
    r2_score(y_test, y_pred)
])

# -------------------------
# Polynomial Regression
# -------------------------
for d in range(2, 6):
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    results.append([
        f"Poly_{d}",
        mean_squared_error(y_test, y_pred),
        r2_score(y_test, y_pred)
    ])

# Save results
df_results = pd.DataFrame(results, columns=["Model", "MSE", "R2"])
print(df_results)

df_results.to_csv("results.csv", index=False)