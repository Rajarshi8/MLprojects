from load_data import load_data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Find misclassified samples
misclassified = X_test[y_pred != y_test]

print("Number of misclassified samples:", len(misclassified))

# Show few samples
print(misclassified.head())