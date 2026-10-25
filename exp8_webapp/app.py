"""
DL Exp8: Simple web application for model predictions.

A Flask web app that lets a user input house feature values via a form and
get a live price prediction from a PyTorch regression model (trained on the
California Housing dataset, same model type as Exp2(a)).

On first run it trains (or loads a cached) model, then serves a small HTML
form at http://127.0.0.1:5000/ for interactive predictions.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

torch.manual_seed(42)
np.random.seed(42)

FEATURE_NAMES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
FEATURE_LABELS = {
    "MedInc": "Median Income (10k USD)",
    "HouseAge": "House Age (years)",
    "AveRooms": "Average Rooms per household",
    "AveBedrms": "Average Bedrooms per household",
    "Population": "Block Population",
    "AveOccup": "Average Occupancy",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}
DEFAULT_VALUES = {
    "MedInc": 5.0, "HouseAge": 25.0, "AveRooms": 6.0, "AveBedrms": 1.0,
    "Population": 1200.0, "AveOccup": 3.0, "Latitude": 34.0, "Longitude": -118.0,
}


class RegressionNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_and_save_model():
    print("Training house price regression model (first run)...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    model = RegressionNet(n_features=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    n_samples = X_train_t.shape[0]
    batch_size = 256
    for epoch in range(1, 61):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            print(f"  epoch {epoch}: loss={loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved scaler -> {SCALER_PATH}")
    return model, scaler


def load_or_train_model():
    model = RegressionNet(n_features=len(FEATURE_NAMES))
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        scaler = joblib.load(SCALER_PATH)
        print("Loaded cached model and scaler.")
        return model, scaler
    model, scaler = train_and_save_model()
    model.eval()
    return model, scaler


app = Flask(__name__)
MODEL, SCALER = load_or_train_model()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_values = DEFAULT_VALUES.copy()

    if request.method == "POST":
        try:
            input_values = {name: float(request.form.get(name, DEFAULT_VALUES[name])) for name in FEATURE_NAMES}
            x = np.array([[input_values[name] for name in FEATURE_NAMES]], dtype=float)
            x_scaled = SCALER.transform(x)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
            with torch.no_grad():
                pred = MODEL(x_tensor).item()
            prediction = round(pred * 100000, 2)  # dataset target is in $100,000s
        except ValueError:
            prediction = "Invalid input, please enter numeric values."

    return render_template(
        "index.html",
        feature_names=FEATURE_NAMES,
        feature_labels=FEATURE_LABELS,
        values=input_values,
        prediction=prediction,
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
