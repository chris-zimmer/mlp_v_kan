import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import yaml

from data.loaders import load_dataset
from evaluation.report import (
    ModelResult,
    make_report_path,
    train_and_report,
    write_pdf_report,
)
from training.trainer import pick_device
from tuning.search_spaces import ModelFactory, build_mlp, build_tuned_kan
from tuning.study import tune

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass(frozen=True)
class Config:
    dataset: str
    data_root: Path
    results_dir: Path
    val_size: int
    seed: int
    save_results: bool


def load_config(path: Path) -> Config:
    """Load config.yaml, resolving its paths against the repository root."""
    with path.open() as f:
        raw = yaml.safe_load(f)
    return Config(
        dataset=raw["dataset"],
        data_root=PROJECT_ROOT / raw["data_root"],
        results_dir=PROJECT_ROOT / raw["results_dir"],
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
    data = load_dataset(config.dataset, config.data_root, config.seed)
    print(f"Dataset: {data.title}")
    # pykan is faster on CPU than MPS at this size, and its ops are more reliable there.
    mlp = partial(build_mlp, input_size=data.input_size, num_classes=data.num_classes)
    kan = partial(
        build_tuned_kan, input_size=data.input_size, num_classes=data.num_classes
    )
    runs: list[tuple[str, ModelFactory, torch.device]] = [
        ("MLP", mlp, device),
        ("KAN", kan, torch.device("cpu")),
    ]
    results: list[ModelResult] = []
    for name, build_model, model_device in runs:
        print(f"\n=== {name} ===")
        params = tune(
            name,
            build_model,
            data.train,
            model_device,
            args.trials,
            args.epochs,
            config.val_size,
            config.seed,
        )
        results.append(
            train_and_report(
                name,
                build_model,
                params,
                data.train,
                data.test,
                model_device,
                args.epochs,
            )
        )
    if config.save_results:
        report_path = make_report_path(config.results_dir, config.dataset)
        write_pdf_report(results, data.title, report_path)
        print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main(load_config(CONFIG_PATH))
