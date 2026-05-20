from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DEFAULT_TRAIN_BATCH_SIZE = 64
DEFAULT_TEST_BATCH_SIZE = 1000


def load_mnist(
    data_dir: str | Path | None = None,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    test_batch_size: int = DEFAULT_TEST_BATCH_SIZE,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, datasets.MNIST, datasets.MNIST]:
    """Load MNIST with standard normalization and deterministic train shuffling."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data"
    else:
        data_dir = Path(data_dir)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)

    # Seeded generator ensures deterministic DataLoader shuffle order across runs.
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader, train_dataset, test_dataset


def get_small_batch(loader: DataLoader, n: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exactly n samples from a loader by concatenating batches as needed."""
    if n <= 0:
        raise ValueError("n must be a positive integer")

    xs = []
    ys = []
    collected = 0

    for batch_x, batch_y in loader:
        needed = n - collected
        if needed <= 0:
            break
        xs.append(batch_x[:needed])
        ys.append(batch_y[:needed])
        collected += min(batch_x.shape[0], needed)
        if collected >= n:
            break

    if not xs:
        raise RuntimeError("Unable to collect samples from loader")

    x_small = torch.cat(xs, dim=0)
    y_small = torch.cat(ys, dim=0)

    if x_small.shape[0] < n:
        raise RuntimeError(f"Requested {n} samples but only collected {x_small.shape[0]}")

    return x_small[:n], y_small[:n]
