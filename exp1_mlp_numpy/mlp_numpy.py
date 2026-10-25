"""
DL Exp1: Multilayer Perceptron for binary classification (XOR) using NumPy.

Implements a simple feedforward neural network (1 hidden layer) trained with
backpropagation and gradient descent, purely in NumPy, to learn the XOR
function (a classic linearly-inseparable binary classification problem).
"""

import numpy as np

np.random.seed(42)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(a):
    # a is sigmoid(x) already
    return a * (1 - a)


class MLP:
    def __init__(self, n_input, n_hidden, n_output, lr=0.1):
        self.lr = lr
        # He/Xavier-ish small random init
        self.W1 = np.random.randn(n_input, n_hidden) * 0.5
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * 0.5
        self.b2 = np.zeros((1, n_output))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]
        output = self.a2

        # Binary cross-entropy derivative w.r.t z2 simplifies to (output - y)
        dz2 = (output - y)
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = dz2 @ self.W2.T
        dz1 = da1 * sigmoid_derivative(self.a1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def compute_loss(self, y_pred, y_true):
        eps = 1e-9
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

    def train(self, X, y, epochs=10000, print_every=1000):
        history = []
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(X, y)
            history.append(loss)
            if epoch % print_every == 0 or epoch == 1:
                acc = self.accuracy(X, y)
                print(f"Epoch {epoch:5d} | Loss: {loss:.6f} | Accuracy: {acc * 100:.2f}%")
        return history

    def predict(self, X):
        probs = self.forward(X)
        return (probs > 0.5).astype(int)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


def main():
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    print("=" * 60)
    print("DL Exp1: MLP for XOR binary classification (NumPy only)")
    print("=" * 60)
    print("Training data (XOR):")
    for xi, yi in zip(X, y):
        print(f"  Input: {xi} -> Target: {yi[0]:.0f}")
    print()

    mlp = MLP(n_input=2, n_hidden=4, n_output=1, lr=0.5)
    mlp.train(X, y, epochs=10000, print_every=1000)

    print()
    print("Final predictions on XOR inputs:")
    probs = mlp.forward(X)
    preds = mlp.predict(X)
    for xi, p, pred, yi in zip(X, probs, preds, y):
        print(f"  Input: {xi} -> Predicted prob: {p[0]:.4f} -> Predicted class: {pred[0]} | True: {int(yi[0])}")

    final_acc = mlp.accuracy(X, y)
    print(f"\nFinal training accuracy: {final_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
