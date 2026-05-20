from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_mnist
from src.losses import CrossEntropyLoss, MSELoss
from src.model import get_fresh_model
from src.trainer import Trainer
from src.utils import save_results, set_seed
from visualizations.plots import plot_generalization


def _build_small_train_loader(train_dataset, seed: int = 42) -> DataLoader:
    subset = Subset(train_dataset, list(range(1000)))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(subset, batch_size=64, shuffle=True, num_workers=0, generator=generator)


def _train_small_data(loss_cls, device: str, epochs: int = 50) -> dict:
    set_seed(42)
    _, test_loader, train_dataset, _ = load_mnist(seed=42)
    train_loader = _build_small_train_loader(train_dataset, seed=42)

    model = get_fresh_model().to(device)
    loss_fn = loss_cls()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)

    return trainer.train(train_loader=train_loader, test_loader=test_loader, epochs=epochs)


def main() -> dict:
    print("Experiment 4: Generalization with 1000 training samples")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mse_history = _train_small_data(MSELoss, device=device, epochs=50)
    ce_history = _train_small_data(CrossEntropyLoss, device=device, epochs=50)

    mse_gap = [test - train for test, train in zip(mse_history["test_loss"], mse_history["train_loss"])]
    ce_gap = [test - train for test, train in zip(ce_history["test_loss"], ce_history["train_loss"])]

    final_mse_gap = float(mse_gap[-1])
    final_ce_gap = float(ce_gap[-1])

    better = "CE" if final_ce_gap < final_mse_gap else "MSE"

    payload = {
        "mse_history": mse_history,
        "ce_history": ce_history,
        "mse_generalization_gap": mse_gap,
        "ce_generalization_gap": ce_gap,
        "final_mse_gap": final_mse_gap,
        "final_ce_gap": final_ce_gap,
        "better_generalization": better,
    }

    save_results(payload, "exp4_generalization.json")
    plot_generalization(mse_history, ce_history)

    print(f"MSE generalization gap at epoch 50: {final_mse_gap:.4f}")
    print(f"CE generalization gap at epoch 50: {final_ce_gap:.4f}")
    print(f"Which loss function generalizes better: {better}")

    return payload


if __name__ == "__main__":
    main()
