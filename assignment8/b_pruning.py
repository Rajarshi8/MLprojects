from load_data import load_data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# DEFINE MODEL FIRST
model = DecisionTreeClassifier(random_state=42)

# Get pruning path
path = model.cost_complexity_pruning_path(X_train, y_train)

ccp_alphas = path.ccp_alphas[::10]   # optimized (skip values)
accuracies = []

for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(
        ccp_alpha=alpha,
        max_depth=10,
        min_samples_split=20,
        random_state=42
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Plot
plt.plot(ccp_alphas, accuracies)
plt.xlabel("ccp_alpha")
plt.ylabel("Accuracy")
plt.title("Pruning Effect")
plt.show()

# Best alpha
best_alpha = ccp_alphas[accuracies.index(max(accuracies))]
print("Best alpha:", best_alpha)