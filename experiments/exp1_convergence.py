from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_mnist
from src.losses import CrossEntropyLoss, MSELoss
from src.model import get_fresh_model
from src.trainer import Trainer
from src.utils import save_results, set_seed
from visualizations.plots import plot_convergence


def _run_training(loss_cls, device: str, epochs: int = 30) -> dict:
    set_seed(42)
    train_loader, test_loader, _, _ = load_mnist(seed=42)

    model = get_fresh_model().to(device)
    loss_fn = loss_cls()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)

    history = trainer.train(train_loader=train_loader, test_loader=test_loader, epochs=epochs)
    return history


def _first_epoch_below(values: list[float], threshold: float) -> int | None:
    for idx, value in enumerate(values, start=1):
        if value < threshold:
            return idx
    return None


def main() -> dict:
    print("Experiment 1: Convergence Comparison (MSE vs CE)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mse_history = _run_training(MSELoss, device=device, epochs=30)
    ce_history = _run_training(CrossEntropyLoss, device=device, epochs=30)

    save_results(mse_history, "exp1_mse_history.json")
    save_results(ce_history, "exp1_ce_history.json")

    plot_convergence(mse_history, ce_history)

    for idx, epoch in enumerate(mse_history["epochs"]):
        mse_loss = mse_history["test_loss"][idx]
        ce_loss = ce_history["test_loss"][idx]
        print(f"Epoch {epoch:02d} | MSE Loss: {mse_loss:.4f} | CE Loss: {ce_loss:.4f}")

    threshold = 0.5
    mse_hit = _first_epoch_below(mse_history["test_loss"], threshold)
    ce_hit = _first_epoch_below(ce_history["test_loss"], threshold)

    expectation_holds = ce_hit is not None and (mse_hit is None or ce_hit < mse_hit)
    print(f"MSE first epoch with test loss < {threshold}: {mse_hit}")
    print(f"CE first epoch with test loss < {threshold}: {ce_hit}")
    print(f"Expectation holds (CE faster than MSE): {expectation_holds}")
    try:
        assert expectation_holds
        print("Assertion result: SUCCESS")
    except AssertionError:
        print("Assertion result: FAILED")

    return {
        "mse_threshold_epoch": mse_hit,
        "ce_threshold_epoch": ce_hit,
        "expectation_holds": expectation_holds,
    }


if __name__ == "__main__":
    main()
