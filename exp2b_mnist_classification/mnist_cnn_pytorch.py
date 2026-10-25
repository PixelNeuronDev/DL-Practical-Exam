"""
DL Exp2(b): MNIST digit classification using PyTorch.

Trains a small CNN on the MNIST dataset (downloaded via torchvision) and
reports test accuracy, plus a visual grid of sample predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mnist_cnn.pt")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))   # 14x14 -> 7x7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def main():
    print("=" * 60)
    print("DL Exp2(b): MNIST Digit Classification using PyTorch")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    print("\nTraining...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"Epoch {epoch}/{epochs} | Loss: {total_loss / total:.4f} | Train Acc: {train_acc * 100:.2f}%")

    model.eval()
    correct, total = 0, 0
    sample_images, sample_preds, sample_labels = None, None, None
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if sample_images is None:
                sample_images = images[:10].cpu()
                sample_preds = preds[:10].cpu()
                sample_labels = labels[:10].cpu()

    test_acc = correct / total
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    # Save a visual grid of sample predictions (for screenshot / report)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        img = sample_images[i].squeeze().numpy()
        ax.imshow(img, cmap="gray")
        color = "green" if sample_preds[i] == sample_labels[i] else "red"
        ax.set_title(f"Pred: {sample_preds[i].item()} (True: {sample_labels[i].item()})", color=color, fontsize=9)
        ax.axis("off")
    plt.suptitle(f"MNIST CNN Predictions (Test Acc: {test_acc*100:.2f}%)")
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "sample_predictions.png")
    plt.savefig(out_path, dpi=120)
    print(f"Sample predictions plot saved to: {out_path}")


if __name__ == "__main__":
    main()
