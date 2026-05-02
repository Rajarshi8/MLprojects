from load_data import load_adult_data
import pandas as pd
import numpy as np

df = load_adult_data()

print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)

# Missing values
print("\nMissing Values:\n", df.isnull().sum())

# Descriptive statistics
print("\nSummary Stats:\n", df.describe())

# -------------------------
# OUTLIERS USING IQR
# -------------------------
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return len(outliers)

print("\nIQR Outliers:")
for col in df.select_dtypes(include=np.number).columns:
    print(col, detect_outliers_iqr(df, col))

# -------------------------
# OUTLIERS USING Z-SCORE
# -------------------------
from scipy.stats import zscore

print("\nZ-score Outliers:")
for col in df.select_dtypes(include=np.number).columns:
    z = np.abs(zscore(df[col].dropna()))
    print(col, (z > 3).sum())
    

with open("data_quality_report.txt", "w") as f:
    f.write("DATA QUALITY REPORT\n\n")

    f.write(f"Shape: {df.shape}\n\n")
    f.write("Missing Values:\n")
    f.write(str(df.isnull().sum()) + "\n\n")

    f.write("Data Types:\n")
    f.write(str(df.dtypes) + "\n\n")

    f.write("Summary Statistics:\n")
    f.write(str(df.describe()) + "\n\n")