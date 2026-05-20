from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def save_results(results_dict: dict, filename: str) -> Path:
    """Save dictionary as JSON in the results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(results_dict), f, indent=2)
    return path


def load_results(filename: str) -> dict:
    """Load JSON results from the results directory."""
    path = RESULTS_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_metrics(metrics_dict: dict) -> str:
    """Pretty-print metrics as key=value pairs."""
    parts = []
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            parts.append(f"{key}: {value:.4f}")
        else:
            parts.append(f"{key}: {value}")
    return " | ".join(parts)


class EarlyStopping:
    """Early stopping utility that monitors validation/test loss."""

    def __init__(self, patience: int = 10) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, test_loss: float) -> bool:
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.counter = 0
            self.should_stop = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def __call__(self, test_loss: float) -> bool:
        return self.step(test_loss)

    def reset(self) -> None:
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False
