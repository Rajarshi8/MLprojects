import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    eps = 1e-9
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))


# Load data
X_train, X_test, y_train, y_test, y_train_oh, y_test_oh = load_data()

# Network structure
input_size = X_train.shape[1]
hidden_size = 32
output_size = y_train_oh.shape[1]

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Hyperparameters
lr = 0.1
epochs = 100
losses = []

for epoch in range(epochs):

    # Forward pass
    Z1 = X_train @ W1 + b1
    A1 = sigmoid(Z1)

    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    # Loss
    loss = cross_entropy(y_train_oh, A2)
    losses.append(loss)

    # Backpropagation
    dZ2 = A2 - y_train_oh
    dW2 = (A1.T @ dZ2) / X_train.shape[0]
    db2 = np.sum(dZ2, axis=0, keepdims=True) / X_train.shape[0]

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = (X_train.T @ dZ1) / X_train.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True) / X_train.shape[0]

    # Update weights
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Plot loss
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.show()

# Prediction
def predict(X):
    A1 = sigmoid(X @ W1 + b1)
    A2 = softmax(A1 @ W2 + b2)
    return np.argmax(A2, axis=1)

y_pred = predict(X_test)
accuracy = np.mean(y_pred == y_test)

print("Scratch ANN Accuracy:", accuracy)