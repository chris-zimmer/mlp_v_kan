import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import yaml

from data.mnist import mnist_datasets
from evaluation.report import make_run_dir, train_and_report
from training.trainer import pick_device
from tuning.search_spaces import ModelFactory, build_mlp, build_tuned_kan
from tuning.study import tune

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass(frozen=True)
class Config:
    data_root: Path
    results_dir: Path
    input_size: int
    num_classes: int
    val_size: int
    seed: int
    save_results: bool


def load_config(path: Path) -> Config:
    """Load config.yaml, resolving its paths against the repository root."""
    with path.open() as f:
        raw = yaml.safe_load(f)
    return Config(
        data_root=PROJECT_ROOT / raw["data_root"],
        results_dir=PROJECT_ROOT / raw["results_dir"],
        input_size=raw["input_size"],
        num_classes=raw["num_classes"],
        val_size=raw["val_size"],
        seed=raw["seed"],
        save_results=raw["save_results"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials", type=int, default=20, help="tuning trials per model"
    )
    parser.add_argument("--epochs", type=int, default=1, help="epochs per training run")
    return parser.parse_args()


def main(config: Config) -> None:
    args = parse_args()
    device = pick_device()
    print(f"Using device: {device}")
    train_set, test_set = mnist_datasets(config.data_root)
    run_dir = make_run_dir(config.results_dir) if config.save_results else None
    # pykan is faster on CPU than MPS at this size, and its ops are more reliable there.
    mlp = partial(
        build_mlp, input_size=config.input_size, num_classes=config.num_classes
    )
    kan = partial(
        build_tuned_kan, input_size=config.input_size, num_classes=config.num_classes
    )
    runs: list[tuple[str, ModelFactory, torch.device]] = [
        ("MLP", mlp, device),
        ("KAN", kan, torch.device("cpu")),
    ]
    for name, build_model, model_device in runs:
        print(f"\n=== {name} ===")
        params = tune(
            name,
            build_model,
            train_set,
            model_device,
            args.trials,
            args.epochs,
            config.val_size,
            config.seed,
        )
        train_and_report(
            name,
            build_model,
            params,
            train_set,
            test_set,
            model_device,
            args.epochs,
            run_dir,
        )


if __name__ == "__main__":
    main(load_config(CONFIG_PATH))
