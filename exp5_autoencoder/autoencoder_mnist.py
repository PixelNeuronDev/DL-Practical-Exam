"""
DL Exp5: Autoencoder for MNIST image denoising / dimensionality reduction.

Trains a fully-connected autoencoder that:
  1. Compresses 28x28 MNIST images into a small latent dimension (dim. reduction)
  2. Learns to reconstruct clean images from noisy inputs (denoising)

Saves a comparison plot of noisy input / reconstruction / original clean image.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

torch.manual_seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "autoencoder.pt")
NOISE_FACTOR = 0.4
LATENT_DIM = 32


class Autoencoder(nn.Module):
    def __init__(self, input_dim=28 * 28, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


def add_noise(images, factor=NOISE_FACTOR):
    noisy = images + factor * torch.randn_like(images)
    return torch.clamp(noisy, 0.0, 1.0)


def main():
    print("=" * 60)
    print("DL Exp5: Autoencoder for MNIST Denoising / Dim. Reduction")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, latent dim: {LATENT_DIM}")

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 8
    print("\nTraining...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total = 0.0, 0
        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(device)
            noisy_images = add_noise(images)

            optimizer.zero_grad()
            reconstructed, _ = model(noisy_images)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

        print(f"Epoch {epoch}/{epochs} | Reconstruction MSE Loss: {total_loss / total:.6f}")

    model.eval()
    with torch.no_grad():
        test_images, _ = next(iter(test_loader))
        test_images_flat = test_images.view(test_images.size(0), -1).to(device)
        noisy_test = add_noise(test_images_flat)
        reconstructed, latent = model(noisy_test)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Latent representation shape per image: {latent.shape[1]} (reduced from 784)")

    # Visual comparison: original / noisy / reconstructed
    n_show = 8
    fig, axes = plt.subplots(3, n_show, figsize=(2 * n_show, 6))
    for i in range(n_show):
        orig = test_images_flat[i].cpu().view(28, 28).numpy()
        noisy = noisy_test[i].cpu().view(28, 28).numpy()
        recon = reconstructed[i].cpu().view(28, 28).numpy()

        axes[0, i].imshow(orig, cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(noisy, cmap="gray")
        axes[1, i].axis("off")
        axes[2, i].imshow(recon, cmap="gray")
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[0, 0].text(-10, 14, "Original", rotation=90, va="center", fontsize=10)
    axes[1, 0].text(-10, 14, "Noisy", rotation=90, va="center", fontsize=10)
    axes[2, 0].text(-10, 14, "Denoised", rotation=90, va="center", fontsize=10)
    plt.suptitle("Autoencoder: Original vs Noisy vs Reconstructed (MNIST)")
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "denoising_results.png")
    plt.savefig(out_path, dpi=120)
    print(f"Comparison plot saved to: {out_path}")


if __name__ == "__main__":
    main()
