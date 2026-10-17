import optuna
import torch
from torch.utils.data import DataLoader, Dataset

from data.loaders import split_train_val
from tuning.search_spaces import ModelFactory, suggest_training
from training.trainer import evaluate, train


def tune(
    name: str,
    build_model: ModelFactory,
    train_set: Dataset,
    device: torch.device,
    n_trials: int,
    epochs: int,
    val_size: int,
    seed: int,
) -> dict[str, int | float | str]:
    """Search hyperparameters by validation accuracy and return the best set found."""
    fit_set, val_set = split_train_val(train_set, val_size, seed)
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
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    print(f"{name} best val accuracy: {study.best_value:.4f}")
    print(f"{name} best params: {study.best_params}")
    return study.best_params
