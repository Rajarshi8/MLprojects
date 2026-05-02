from load_data import load_data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

metrics = {
    "Euclidean": KNeighborsClassifier(metric='euclidean'),
    "Manhattan": KNeighborsClassifier(metric='manhattan'),
    "Minkowski": KNeighborsClassifier(metric='minkowski', p=3)
}

for name, model in metrics.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name} Accuracy:", accuracy_score(y_test, y_pred))