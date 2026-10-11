from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader, Dataset

from tuning.search_spaces import ModelFactory, suggest_training
from training.trainer import predict, train


def save_confusion_matrix(
    name: str, targets: torch.Tensor, preds: torch.Tensor, results_dir: Path
) -> Path:
    """Plot the confusion matrix of preds against targets and save it as a PNG."""
    results_dir.mkdir(exist_ok=True)
    path = results_dir / f"{name.lower()}_confusion_matrix.png"
    display = ConfusionMatrixDisplay.from_predictions(targets.numpy(), preds.numpy())
    display.ax_.set_title(f"{name} test confusion matrix")
    display.figure_.savefig(path, bbox_inches="tight")
    plt.close(display.figure_)
    return path


def train_and_report(
    name: str,
    build_model: ModelFactory,
    params: dict[str, int | float | str],
    train_set: Dataset,
    test_set: Dataset,
    device: torch.device,
    epochs: int,
    results_dir: Path,
) -> None:
    """Retrain with params on the full training set, print params, train time, test accuracy, and a classification report, and save a confusion matrix."""
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
    print(
        f"{name} confusion matrix: {save_confusion_matrix(name, targets, preds, results_dir)}"
    )
    print(f"{name} classification report:")
    print(classification_report(targets.numpy(), preds.numpy(), digits=4))
