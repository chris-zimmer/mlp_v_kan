import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def pick_device() -> torch.device:
    """Return the best available torch device, preferring CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (targets, predicted classes) for every example in loader, on CPU."""
    model.eval()
    targets: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for data, target in loader:
            data = data.flatten(start_dim=1).to(device)
            preds.append(model(data).argmax(dim=1).cpu())
            targets.append(target)
    return torch.cat(targets), torch.cat(preds)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy of model over loader on the given device."""
    targets, preds = predict(model, loader, device)
    return (preds == targets).float().mean().item()


def train(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
) -> float:
    """Train model with Adam + cross-entropy and return the wall time in seconds."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    model.train()
    for _ in range(epochs):
        for data, target in loader:
            data = data.flatten(start_dim=1).to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    return time.time() - start
