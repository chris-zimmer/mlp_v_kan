from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, TensorDataset, random_split
from torchvision import datasets, transforms

COVTYPE_CONTINUOUS_FEATURES = 10
COVTYPE_TEST_FRACTION = 0.2


@dataclass(frozen=True)
class DatasetSplits:
    title: str
    train: Dataset
    test: Dataset
    input_size: int
    num_classes: int


def _torchvision_splits(
    dataset_class: type[datasets.MNIST],
    title: str,
    mean: float,
    std: float,
    data_root: Path,
) -> DatasetSplits:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
    )
    train_set = dataset_class(
        root=str(data_root), train=True, transform=transform, download=True
    )
    test_set = dataset_class(
        root=str(data_root), train=False, transform=transform, download=True
    )
    return DatasetSplits(title, train_set, test_set, input_size=784, num_classes=10)


def mnist_splits(data_root: Path, seed: int) -> DatasetSplits:
    """Load normalized MNIST train and test sets at full 28x28 resolution."""
    return _torchvision_splits(datasets.MNIST, "MNIST", 0.1307, 0.3081, data_root)


def fashion_mnist_splits(data_root: Path, seed: int) -> DatasetSplits:
    """Load normalized Fashion-MNIST train and test sets at full 28x28 resolution."""
    return _torchvision_splits(
        datasets.FashionMNIST, "Fashion-MNIST", 0.2860, 0.3530, data_root
    )


def covtype_splits(data_root: Path, seed: int) -> DatasetSplits:
    """Load Covertype with a stratified test split, standardizing only the continuous features."""
    features, labels = fetch_covtype(data_home=str(data_root), return_X_y=True)
    labels = labels - 1
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=COVTYPE_TEST_FRACTION,
        stratify=labels,
        random_state=seed,
    )
    scaler = StandardScaler().fit(train_x[:, :COVTYPE_CONTINUOUS_FEATURES])
    train_x[:, :COVTYPE_CONTINUOUS_FEATURES] = scaler.transform(
        train_x[:, :COVTYPE_CONTINUOUS_FEATURES]
    )
    test_x[:, :COVTYPE_CONTINUOUS_FEATURES] = scaler.transform(
        test_x[:, :COVTYPE_CONTINUOUS_FEATURES]
    )
    return DatasetSplits(
        "Covertype",
        _tensor_dataset(train_x, train_y),
        _tensor_dataset(test_x, test_y),
        input_size=features.shape[1],
        num_classes=int(labels.max()) + 1,
    )


def _tensor_dataset(features: np.ndarray, labels: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(features).float(), torch.from_numpy(labels).long()
    )


DATASETS: dict[str, Callable[[Path, int], DatasetSplits]] = {
    "mnist": mnist_splits,
    "fashion_mnist": fashion_mnist_splits,
    "covtype": covtype_splits,
}


def load_dataset(name: str, data_root: Path, seed: int) -> DatasetSplits:
    """Load the named dataset's train and test splits, downloading it if missing."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[name](data_root, seed)


def split_train_val(
    train_set: Dataset, val_size: int, seed: int
) -> tuple[Dataset, Dataset]:
    """Hold out val_size training examples for tuning so the test set stays untouched."""
    generator = torch.Generator().manual_seed(seed)
    train_size = len(train_set) - val_size  # type: ignore[arg-type]
    fit_set, val_set = random_split(train_set, [train_size, val_size], generator)
    return fit_set, val_set
