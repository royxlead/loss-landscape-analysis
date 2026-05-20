from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_mnist
from src.losses import CrossEntropyLoss, MSELoss
from src.model import get_fresh_model
from src.trainer import Trainer
from src.utils import save_results, set_seed
from visualizations.plots import plot_lr_sensitivity


LEARNING_RATES = [0.0001, 0.001, 0.01, 0.1, 0.5]


def _run_setting(loss_cls, lr: float, device: str, epochs: int = 15) -> dict:
    set_seed(42)
    train_loader, test_loader, _, _ = load_mnist(seed=42)

    model = get_fresh_model().to(device)
    loss_fn = loss_cls()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)

    final_train_loss = float("nan")
    final_test_loss = float("nan")
    final_test_acc = float("nan")
    diverged = False

    for epoch in range(1, epochs + 1):
        train_metrics = trainer.train_epoch(train_loader)
        test_metrics = trainer.evaluate(test_loader)

        final_train_loss = train_metrics["loss"]
        final_test_loss = test_metrics["loss"]
        final_test_acc = test_metrics["accuracy"]

        if (
            np.isnan(final_train_loss)
            or np.isnan(final_test_loss)
            or final_train_loss > 100
            or final_test_loss > 100
        ):
            diverged = True
            break

    status = "diverged" if diverged else "ok"
    return {
        "train_loss": float(final_train_loss),
        "test_loss": float(final_test_loss),
        "loss": float(final_test_loss),
        "acc": float(final_test_acc),
        "diverged": bool(diverged),
        "status": status,
    }


def main() -> dict:
    print("Experiment 3: Learning Rate Sensitivity")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = {}

    for lr in LEARNING_RATES:
        print(f"Testing LR={lr:g} with MSE...")
        mse_result = _run_setting(MSELoss, lr=lr, device=device, epochs=15)

        print(f"Testing LR={lr:g} with CE...")
        ce_result = _run_setting(CrossEntropyLoss, lr=lr, device=device, epochs=15)

        results[str(lr)] = {"mse": mse_result, "ce": ce_result}

    save_results(results, "exp3_lr_sensitivity.json")
    plot_lr_sensitivity(results)

    print("LR     | MSE Loss | MSE Acc | CE Loss | CE Acc | MSE Status | CE Status")
    for lr in LEARNING_RATES:
        row = results[str(lr)]
        print(
            f"{lr:<6g} | "
            f"{row['mse']['loss']:<8.4f} | "
            f"{row['mse']['acc']:<7.4f} | "
            f"{row['ce']['loss']:<7.4f} | "
            f"{row['ce']['acc']:<6.4f} | "
            f"{row['mse']['status']:<10} | "
            f"{row['ce']['status']}"
        )

    return results


if __name__ == "__main__":
    main()
