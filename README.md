# Deep Learning (DL) Practical Exams

Solutions for the DL practical exam experiments, implemented in Python
(NumPy / PyTorch / Flask).

## Experiments

| # | Title | Folder | Tech |
|---|-------|--------|------|
| Exp1 | Multilayer Perceptron for binary classification (XOR) | `exp1_mlp_numpy/` | NumPy (from scratch) |
| Exp2(a) | Regression — House Price Prediction | `exp2a_regression/` | PyTorch |
| Exp2(b) | Classification — MNIST Digit Classification | `exp2b_mnist_classification/` | PyTorch (CNN) |
| Exp5 | Autoencoder for image denoising / dimensionality reduction | `exp5_autoencoder/` | PyTorch |
| Exp8 | Simple web app for model predictions | `exp8_webapp/` | Flask + PyTorch |

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
# Exp1 - MLP XOR classifier (NumPy)
python exp1_mlp_numpy/mlp_numpy.py

# Exp2(a) - House price regression (PyTorch)
python exp2a_regression/regression_pytorch.py

# Exp2(b) - MNIST digit classification (PyTorch CNN)
python exp2b_mnist_classification/mnist_cnn_pytorch.py

# Exp5 - Autoencoder denoising (PyTorch)
python exp5_autoencoder/autoencoder_mnist.py

# Exp8 - Web app (Flask)
python exp8_webapp/app.py
# then open http://127.0.0.1:5000/ in a browser
```

## Notes

- Exp2(b) and Exp5 download the MNIST dataset automatically via `torchvision`
  into a local `data/` folder on first run.
- Exp8 trains a small regression model on first run and caches it
  (`house_price_model.pt`, `scaler.pkl`) for subsequent runs.
- Screenshots of runs/outputs are in the `screenshots/` folder.
