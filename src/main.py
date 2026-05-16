import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.kan import build_kan
from models.mlp import MLP

DATA_ROOT = Path(__file__).resolve().parent / "data"
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
KAN_INPUT_SIZE = 8 * 8
KAN_TRAIN_SAMPLES = 1000
KAN_TEST_SAMPLES = 1000


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mnist_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
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


def mnist_tensors(
    n_train: int, n_test: int, image_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_set = Subset(
        datasets.MNIST(
            root=str(DATA_ROOT), train=True, transform=transform, download=False
        ),
        range(n_train),
    )
    test_set = Subset(
        datasets.MNIST(
            root=str(DATA_ROOT), train=False, transform=transform, download=False
        ),
        range(n_test),
    )

    def stack(ds: Subset) -> tuple[torch.Tensor, torch.Tensor]:
        xs = torch.stack([ds[i][0].view(-1) for i in range(len(ds))])
        ys = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.long)
        return xs, ys

    train_x, train_y = stack(train_set)
    test_x, test_y = stack(test_set)
    return train_x, train_y, test_x, test_y


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
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


def run_mlp(device: torch.device, epochs: int = 1) -> None:
    print("\n=== MLP ===")
    train_loader, test_loader = mnist_loaders(batch_size=64)
    model = MLP(INPUT_SIZE, 128, NUM_CLASSES).to(device)
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
    print(f"MLP params: {n_params}")
    print(f"MLP train time: {elapsed:.1f}s")
    print(f"MLP test accuracy: {acc:.4f}")


def run_kan(device: torch.device, steps: int = 20) -> None:
    print("\n=== KAN ===")
    # KAN training is expensive; use downsampled images and a subset.
    train_x, train_y, test_x, test_y = mnist_tensors(
        n_train=KAN_TRAIN_SAMPLES,
        n_test=KAN_TEST_SAMPLES,
        image_size=int(KAN_INPUT_SIZE**0.5),
    )

    # pykan's internal ops are most reliable on CPU; force CPU here.
    kan_device = torch.device("cpu")
    train_x = train_x.to(kan_device)
    train_y = train_y.to(kan_device)
    test_x = test_x.to(kan_device)
    test_y = test_y.to(kan_device)

    model = build_kan(
        input_size=KAN_INPUT_SIZE, hidden_size=10, output_size=NUM_CLASSES
    ).to(kan_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=10, history_size=10
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(model(train_x), train_y)
        loss.backward()
        return loss

    start = time.time()
    model.train()
    for step in range(steps):
        loss = optimizer.step(closure)
        print(f"  step {step + 1}/{steps} loss={loss.item():.4f}")
    elapsed = time.time() - start

    model.eval()
    with torch.no_grad():
        preds = model(test_x).argmax(dim=1)
        acc = (preds == test_y).float().mean().item()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"KAN params: {n_params}")
    print(f"KAN train time: {elapsed:.1f}s")
    print(
        f"KAN test accuracy: {acc:.4f} "
        f"(downsampled {int(KAN_INPUT_SIZE**0.5)}x{int(KAN_INPUT_SIZE**0.5)}, "
        f"{KAN_TRAIN_SAMPLES} train / {KAN_TEST_SAMPLES} test)"
    )


def main() -> None:
    device = pick_device()
    print(f"Using device: {device}")
    run_mlp(device)
    run_kan(device)


if __name__ == "__main__":
    main()
