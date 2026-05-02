from load_data import load_data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

X, y = load_data()

# Add random noisy features
for i in range(20):
    X[f"noise_{i}"] = np.random.rand(len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy after adding noise features:",
      accuracy_score(y_test, y_pred))