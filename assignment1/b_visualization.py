from load_data import load_wine_data
import matplotlib.pyplot as plt
import seaborn as sns

df = load_wine_data()

# Histogram
sns.histplot(df['quality'], bins=10, kde=True)
plt.title("Wine Quality Distribution")
plt.show()

# Boxplots
features = df.columns[:-2]

for col in features:
    plt.figure()
    sns.boxplot(x=df['quality'], y=df[col])
    plt.title(f"{col} vs Quality")
    plt.show()

# Heatmap
corr = df.corr(numeric_only=True)

plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Correlation insights
quality_corr = corr['quality'].sort_values(ascending=False)

print("Top Positive Correlations:\n", quality_corr.head())
print("\nTop Negative Correlations:\n", quality_corr.tail())