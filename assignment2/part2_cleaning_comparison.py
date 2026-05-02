from load_data import load_adult_data
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = load_adult_data()

# -------------------------
# CLEANING STRATEGIES
# -------------------------

# 1️⃣ Drop missing
df_drop = df.dropna()

# 2️⃣ Mean/Mode
df_mean = df.copy()
for col in df_mean.columns:
    if df_mean[col].dtype == 'object':
        df_mean[col].fillna(df_mean[col].mode()[0], inplace=True)
    else:
        df_mean[col].fillna(df_mean[col].mean(), inplace=True)

# 3️⃣ Forward Fill
df_ffill = df.fillna(method='ffill')

datasets = {
    "Drop": df_drop,
    "Mean/Mode": df_mean,
    "Forward Fill": df_ffill
}

# -------------------------
# ENCODING FUNCTION
# -------------------------
def preprocess(df):
    df = df.copy()
    le = LabelEncoder()

    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop("income", axis=1)
    y = df["income"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# EVALUATION
# -------------------------
results = {}

for name, dataset in datasets.items():
    X_train, X_test, y_train, y_test = preprocess(dataset)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    results[name] = acc

    print(f"\n{name}")
    print("Shape:", dataset.shape)
    print("Accuracy:", acc)
    print("Mean (age):", dataset['age'].mean())
    print("Variance (age):", dataset['age'].var())
    print("Class Distribution:\n", dataset['income'].value_counts())

# -------------------------
# BEST STRATEGY
# -------------------------
best = max(results, key=results.get)
print("\nBest Cleaning Strategy:", best)