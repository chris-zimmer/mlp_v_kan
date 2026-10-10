import argparse
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from models.kan import build_kan
from models.mlp import MLP

DATA_ROOT = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
VAL_SIZE = 5_000
SEED = 0

ModelFactory = Callable[[optuna.trial.BaseTrial], nn.Module]


def pick_device() -> torch.device:
    """Return the best available torch device, preferring CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mnist_datasets() -> tuple[Dataset, Dataset]:
    """Load normalized MNIST train and test sets at full 28x28 resolution."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(
        root=str(DATA_ROOT), train=True, transform=transform, download=False
    )
    test_set = datasets.MNIST(
        root=str(DATA_ROOT), train=False, transform=transform, download=False
    )
    return train_set, test_set


def split_train_val(train_set: Dataset) -> tuple[Dataset, Dataset]:
    """Hold out VAL_SIZE training examples for tuning so the test set stays untouched."""
    generator = torch.Generator().manual_seed(SEED)
    train_size = len(train_set) - VAL_SIZE  # type: ignore[arg-type]
    fit_set, val_set = random_split(train_set, [train_size, VAL_SIZE], generator)
    return fit_set, val_set


def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (targets, predicted classes) for every example in loader, on CPU."""
    model.eval()
    targets: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for data, target in loader:
            data = data.view(-1, INPUT_SIZE).to(device)
            preds.append(model(data).argmax(dim=1).cpu())
            targets.append(target)
    return torch.cat(targets), torch.cat(preds)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy of model over loader on the given device."""
    targets, preds = predict(model, loader, device)
    return (preds == targets).float().mean().item()


def save_confusion_matrix(
    name: str, targets: torch.Tensor, preds: torch.Tensor
) -> Path:
    """Plot the confusion matrix of preds against targets and save it as a PNG."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name.lower()}_confusion_matrix.png"
    display = ConfusionMatrixDisplay.from_predictions(targets.numpy(), preds.numpy())
    display.ax_.set_title(f"{name} test confusion matrix")
    display.figure_.savefig(path, bbox_inches="tight")
    plt.close(display.figure_)
    return path


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
            data = data.view(-1, INPUT_SIZE).to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    return time.time() - start


def suggest_training(trial: optuna.trial.BaseTrial) -> tuple[float, int]:
    """Suggest the optimizer and batching hyperparameters shared by every model."""
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    return lr, batch_size


def build_mlp(trial: optuna.trial.BaseTrial) -> nn.Module:
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
    return MLP(INPUT_SIZE, hidden_size, NUM_CLASSES)


def build_tuned_kan(trial: optuna.trial.BaseTrial) -> nn.Module:
    return build_kan(
        input_size=INPUT_SIZE,
        hidden_size=trial.suggest_categorical("hidden_size", [5, 10, 20]),
        output_size=NUM_CLASSES,
        grid=trial.suggest_categorical("grid", [3, 5, 8]),
        k=trial.suggest_categorical("k", [2, 3]),
    )


def tune(
    name: str,
    build_model: ModelFactory,
    train_set: Dataset,
    device: torch.device,
    n_trials: int,
    epochs: int,
) -> dict[str, int | float | str]:
    """Search hyperparameters by validation accuracy and return the best set found."""
    fit_set, val_set = split_train_val(train_set)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

    def objective(trial: optuna.Trial) -> float:
        model = build_model(trial)
        lr, batch_size = suggest_training(trial)
        fit_loader = DataLoader(fit_set, batch_size=batch_size, shuffle=True)
        train(model, fit_loader, device, lr, epochs)
        return evaluate(model, val_loader, device)

    study = optuna.create_study(
        study_name=name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials)
    print(f"{name} best val accuracy: {study.best_value:.4f}")
    print(f"{name} best params: {study.best_params}")
    return study.best_params


def train_and_report(
    name: str,
    build_model: ModelFactory,
    params: dict[str, int | float | str],
    train_set: Dataset,
    test_set: Dataset,
    device: torch.device,
    epochs: int,
) -> None:
    """Retrain with params on the full training set, print params, train time, and test accuracy, and save a confusion matrix."""
    trial = optuna.trial.FixedTrial(params)
    model = build_model(trial)
    lr, batch_size = suggest_training(trial)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    elapsed = train(model, train_loader, device, lr, epochs)
    targets, preds = predict(model, test_loader, device)
    acc = (preds == targets).float().mean().item()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name} params: {n_params}")
    print(f"{name} train time: {elapsed:.1f}s")
    print(f"{name} test accuracy: {acc:.4f}")
    print(f"{name} confusion matrix: {save_confusion_matrix(name, targets, preds)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials", type=int, default=20, help="tuning trials per model"
    )
    parser.add_argument("--epochs", type=int, default=1, help="epochs per training run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    print(f"Using device: {device}")
    train_set, test_set = mnist_datasets()
    # pykan is faster on CPU than MPS at this size, and its ops are more reliable there.
    runs: list[tuple[str, ModelFactory, torch.device]] = [
        ("MLP", build_mlp, device),
        ("KAN", build_tuned_kan, torch.device("cpu")),
    ]
    for name, build_model, model_device in runs:
        print(f"\n=== {name} ===")
        params = tune(
            name, build_model, train_set, model_device, args.trials, args.epochs
        )
        train_and_report(
            name, build_model, params, train_set, test_set, model_device, args.epochs
        )


if __name__ == "__main__":
    main()
