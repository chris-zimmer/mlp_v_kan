import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader, Dataset

from tuning.search_spaces import ModelFactory, suggest_training
from training.trainer import predict, train

PAGE_SIZE = (8.5, 11)


@dataclass(frozen=True)
class ModelResult:
    name: str
    hyperparams: dict[str, int | float | str]
    n_params: int
    train_time: float
    accuracy: float
    targets: torch.Tensor
    preds: torch.Tensor


def make_report_path(results_dir: Path, dataset: str) -> Path:
    """Return a uniquely named results/<date>_<dataset>_<short uuid>.pdf path for this run."""
    results_dir.mkdir(parents=True, exist_ok=True)
    return (
        results_dir / f"{date.today().isoformat()}_{dataset}_{uuid.uuid4().hex[:8]}.pdf"
    )


def train_and_report(
    name: str,
    build_model: ModelFactory,
    params: dict[str, int | float | str],
    train_set: Dataset,
    test_set: Dataset,
    device: torch.device,
    epochs: int,
) -> ModelResult:
    """Retrain with params on the full training set, print params, train time, test accuracy, and a classification report, and return the results."""
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
    print(f"{name} classification report:")
    print(classification_report(targets.numpy(), preds.numpy(), digits=4))
    return ModelResult(name, params, n_params, elapsed, acc, targets, preds)


def _format_value(value: int | float | str) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _summary_page(results: list[ModelResult], dataset_title: str) -> Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.suptitle(f"MLP vs KAN on {dataset_title}", fontsize=16, y=0.95)
    ax = fig.add_axes((0.1, 0.55, 0.8, 0.3))
    ax.axis("off")
    ax.table(
        cellText=[
            [r.name, f"{r.accuracy:.4f}", f"{r.n_params:,}", f"{r.train_time:.1f}"]
            for r in results
        ],
        colLabels=["Model", "Test accuracy", "Parameters", "Train time (s)"],
        loc="upper center",
        cellLoc="center",
    ).scale(1, 2)
    return fig


def _model_page(result: ModelResult, dataset_title: str) -> Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.suptitle(f"{result.name} on {dataset_title}", fontsize=16, y=0.97)

    hyperparams = ", ".join(
        f"{key}={_format_value(value)}" for key, value in result.hyperparams.items()
    )
    fig.text(
        0.5,
        0.93,
        f"Test accuracy {result.accuracy:.4f} · {result.n_params:,} parameters · "
        f"trained in {result.train_time:.1f}s\n{hyperparams}",
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )

    report = classification_report(
        result.targets.numpy(), result.preds.numpy(), digits=4, output_dict=True
    )
    rows = [
        [
            label,
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1-score']:.4f}",
            f"{int(m['support'])}",
        ]
        for label, m in report.items()
        if isinstance(m, dict)
    ]
    table_ax = fig.add_axes((0.1, 0.52, 0.8, 0.36))
    table_ax.axis("off")
    table_ax.table(
        cellText=rows,
        colLabels=["Class", "Precision", "Recall", "F1", "Support"],
        loc="upper center",
        cellLoc="center",
    ).scale(1, 1.3)

    matrix_ax = fig.add_axes((0.15, 0.05, 0.7, 0.42))
    ConfusionMatrixDisplay.from_predictions(
        result.targets.numpy(), result.preds.numpy(), ax=matrix_ax, colorbar=False
    )
    matrix_ax.set_title("Test confusion matrix")
    return fig


def write_pdf_report(
    results: list[ModelResult], dataset_title: str, path: Path
) -> None:
    """Write a PDF with a model comparison page followed by one page per model."""
    with PdfPages(path) as pdf:
        for fig in [
            _summary_page(results, dataset_title),
            *(_model_page(result, dataset_title) for result in results),
        ]:
            pdf.savefig(fig)
            plt.close(fig)
