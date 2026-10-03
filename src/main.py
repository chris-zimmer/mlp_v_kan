import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.kan import build_kan
from models.mlp import MLP

DATA_ROOT = Path(__file__).resolve().parent / "data"
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10


def pick_device() -> torch.device:
    """Return the best available torch device, preferring CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mnist_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Build normalized MNIST train and test DataLoaders at full 28x28 resolution."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(
        root=str(DATA_ROOT), train=True, transform=transform, download=False
    )
    test_set = datasets.MNIST(
        root=str(DATA_ROOT), train=False, transform=transform, download=False
    )
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy of model over loader on the given device."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.view(-1, INPUT_SIZE).to(device)
            target = target.to(device)
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total


def train_and_report(
    name: str, model: nn.Module, device: torch.device, epochs: int = 1
) -> None:
    """Train model on full MNIST with Adam + cross-entropy and print params, train time, and test accuracy."""
    print(f"\n=== {name} ===")
    train_loader, test_loader = mnist_loaders(batch_size=64)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    start = time.time()
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, INPUT_SIZE).to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            if batch_idx % 200 == 0:
                print(
                    f"  epoch {epoch} batch {batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )
    elapsed = time.time() - start
    acc = evaluate(model, test_loader, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name} params: {n_params}")
    print(f"{name} train time: {elapsed:.1f}s")
    print(f"{name} test accuracy: {acc:.4f}")


def main() -> None:
    device = pick_device()
    print(f"Using device: {device}")
    train_and_report("MLP", MLP(INPUT_SIZE, 128, NUM_CLASSES), device)
    # pykan is faster on CPU than MPS at this size, and its ops are more reliable there.
    train_and_report(
        "KAN",
        build_kan(input_size=INPUT_SIZE, hidden_size=10, output_size=NUM_CLASSES),
        torch.device("cpu"),
    )


if __name__ == "__main__":
    main()
