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
from src.trainer import compute_gradient_norm
from src.utils import save_results, set_seed
from visualizations.plots import plot_gradient_norms


def _run_gradient_tracking(loss_cls, loss_name: str, device: str, epochs: int = 20) -> dict:
    set_seed(42)
    train_loader, _, _, _ = load_mnist(seed=42)

    model = get_fresh_model().to(device)
    loss_fn = loss_cls()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    data = {
        "loss_name": loss_name,
        "epochs": [],
        "mean_grad_norm": [],
        "std_grad_norm": [],
        "min_grad_norm": [],
        "max_grad_norm": [],
        "train_loss": [],
        "train_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        batch_grad_norms = []
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        optimizer.zero_grad()

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()

            grad_norm = compute_gradient_norm(model)
            batch_grad_norms.append(grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            preds = torch.argmax(logits, dim=1)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size

        grad_array = np.array(batch_grad_norms, dtype=np.float64)
        mean_grad = float(np.mean(grad_array))
        std_grad = float(np.std(grad_array))
        min_grad = float(np.min(grad_array))
        max_grad = float(np.max(grad_array))

        data["epochs"].append(epoch)
        data["mean_grad_norm"].append(mean_grad)
        data["std_grad_norm"].append(std_grad)
        data["min_grad_norm"].append(min_grad)
        data["max_grad_norm"].append(max_grad)
        data["train_loss"].append(float(total_loss / total_samples))
        data["train_accuracy"].append(float(total_correct / total_samples))

        print(
            f"{loss_name} Epoch {epoch:02d}/{epochs} "
            f"| Mean Grad: {mean_grad:.6f} "
            f"| Std: {std_grad:.6f} "
            f"| Min: {min_grad:.6f} "
            f"| Max: {max_grad:.6f}"
        )

    return data


def main() -> dict:
    print("Experiment 2: Gradient Magnitude Analysis")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mse_data = _run_gradient_tracking(MSELoss, "MSE", device=device, epochs=20)
    ce_data = _run_gradient_tracking(CrossEntropyLoss, "CE", device=device, epochs=20)

    save_results(mse_data, "exp2_mse_gradients.json")
    save_results(ce_data, "exp2_ce_gradients.json")

    plot_gradient_norms(mse_data, ce_data)

    ratio_epoch5 = ce_data["mean_grad_norm"][4] / max(mse_data["mean_grad_norm"][4], 1e-12)
    ratio_epoch20 = ce_data["mean_grad_norm"][19] / max(mse_data["mean_grad_norm"][19], 1e-12)

    mse_decreases = mse_data["mean_grad_norm"][-1] < mse_data["mean_grad_norm"][0]
    ce_relative_change = abs(ce_data["mean_grad_norm"][-1] - ce_data["mean_grad_norm"][0]) / max(
        ce_data["mean_grad_norm"][0], 1e-12
    )
    ce_stable = ce_relative_change < 0.5

    print(f"Mean grad norm ratio CE/MSE at epoch 5: {ratio_epoch5:.2f}")
    print(f"Mean grad norm ratio CE/MSE at epoch 20: {ratio_epoch20:.2f}")
    print(f"Does MSE gradient norm decrease over time (saturation)? {mse_decreases}")
    print(f"Does CE gradient norm stay more stable? {ce_stable}")

    return {
        "ratio_epoch5": ratio_epoch5,
        "ratio_epoch20": ratio_epoch20,
        "mse_decreases": mse_decreases,
        "ce_stable": ce_stable,
    }


if __name__ == "__main__":
    main()
