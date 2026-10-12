from collections.abc import Callable

import optuna
import torch.nn as nn

from models.kan import build_kan
from models.mlp import MLP

ModelFactory = Callable[[optuna.trial.BaseTrial], nn.Module]


def suggest_training(trial: optuna.trial.BaseTrial) -> tuple[float, int]:
    """Suggest the optimizer and batching hyperparameters shared by every model."""
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    return lr, batch_size


def build_mlp(
    trial: optuna.trial.BaseTrial, input_size: int, num_classes: int
) -> nn.Module:
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
    return MLP(input_size, hidden_size, num_classes)


def build_tuned_kan(
    trial: optuna.trial.BaseTrial, input_size: int, num_classes: int
) -> nn.Module:
    return build_kan(
        input_size=input_size,
        hidden_size=trial.suggest_categorical("hidden_size", [5, 10, 20]),
        output_size=num_classes,
        grid=trial.suggest_categorical("grid", [3, 5, 8]),
        k=trial.suggest_categorical("k", [2, 3]),
    )
