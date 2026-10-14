import uuid
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader, Dataset

from tuning.search_spaces import ModelFactory, suggest_training
from training.trainer import predict, train


def make_run_dir(results_dir: Path) -> Path:
    """Create a uniquely named results/<date>_<short uuid> directory for this run."""
    run_dir = results_dir / f"{date.today().isoformat()}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True)
    return run_dir


def save_confusion_matrix(
    name: str, targets: torch.Tensor, preds: torch.Tensor, run_dir: Path
) -> Path:
    """Plot the confusion matrix of preds against targets and save it as a PNG."""
    path = run_dir / f"{name.lower()}_confusion_matrix.png"
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
    run_dir: Path | None,
) -> None:
    """Retrain with params on the full training set, print params, train time, test accuracy, and a classification report; if run_dir is set, save the report and a confusion matrix there."""
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
    report = classification_report(targets.numpy(), preds.numpy(), digits=4)
    print(f"{name} classification report:")
    print(report)
    if run_dir is None:
        return
    report_path = run_dir / f"{name.lower()}_classification_report.txt"
    report_path.write_text(report)
    print(f"{name} classification report saved: {report_path}")
    print(
        f"{name} confusion matrix saved: {save_confusion_matrix(name, targets, preds, run_dir)}"
    )
