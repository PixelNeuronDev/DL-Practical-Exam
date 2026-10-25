"""
DL Exp2(a): Regression model (house price prediction) using PyTorch.

Uses the California Housing dataset (via scikit-learn) and trains a small
feedforward neural network to predict median house value.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

torch.manual_seed(42)
np.random.seed(42)


class RegressionNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    print("=" * 60)
    print("DL Exp2(a): House Price Regression using PyTorch")
    print("=" * 60)

    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names
    print(f"Dataset shape: {X.shape}, Target shape: {y.shape}")
    print(f"Features: {feature_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_X = StandardScaler().fit(X_train)
    X_train_s = scaler_X.transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    model = RegressionNet(n_features=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 100
    batch_size = 256
    n_samples = X_train_t.shape[0]

    print("\nTraining...")
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= n_samples
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Train MSE Loss: {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).numpy().flatten()

    mse = mean_squared_error(y_test, test_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, test_preds)

    print("\nEvaluation on test set:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R^2:  {r2:.4f}")

    print("\nSample predictions (price in $100,000s):")
    for i in range(5):
        print(f"  Actual: {y_test[i]:.3f} | Predicted: {test_preds[i]:.3f}")


if __name__ == "__main__":
    main()
