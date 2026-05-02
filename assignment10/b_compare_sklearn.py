from load_data import load_data
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test, _, _ = load_data()

model = MLPClassifier(
    hidden_layer_sizes=(32,),
    max_iter=300,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Sklearn MLP Accuracy:", accuracy_score(y_test, y_pred))  