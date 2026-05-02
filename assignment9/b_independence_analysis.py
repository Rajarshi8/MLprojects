from load_data import load_data
import seaborn as sns
import matplotlib.pyplot as plt

X, y = load_data()

corr = X.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Count highly correlated features
high_corr = ((corr.abs() > 0.7).sum().sum()) - len(corr)
print("Highly correlated feature pairs:", high_corr)