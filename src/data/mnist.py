from pathlib import Path

import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms


def mnist_datasets(data_root: Path) -> tuple[Dataset, Dataset]:
    """Load normalized MNIST train and test sets at full 28x28 resolution."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(
        root=str(data_root), train=True, transform=transform, download=False
    )
    test_set = datasets.MNIST(
        root=str(data_root), train=False, transform=transform, download=False
    )
    return train_set, test_set


def split_train_val(
    train_set: Dataset, val_size: int, seed: int
) -> tuple[Dataset, Dataset]:
    """Hold out val_size training examples for tuning so the test set stays untouched."""
    generator = torch.Generator().manual_seed(seed)
    train_size = len(train_set) - val_size  # type: ignore[arg-type]
    fit_set, val_set = random_split(train_set, [train_size, val_size], generator)
    return fit_set, val_set
