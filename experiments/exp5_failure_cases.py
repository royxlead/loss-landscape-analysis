from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_mnist
from src.losses import CrossEntropyLoss, MSELoss, softmax
from src.model import get_fresh_model
from src.trainer import Trainer, compute_gradient_norm
from src.utils import save_results, set_seed
from visualizations.plots import plot_failure_cases


def _train_underfit_model(device: str) -> tuple[torch.nn.Module, torch.utils.data.DataLoader]:
    set_seed(42)
    train_loader, test_loader, _, _ = load_mnist(seed=42)

    model = get_fresh_model().to(device)
    loss_fn = MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)

    print("Training underfit MSE model for 5 epochs...")
    trainer.train(train_loader=train_loader, test_loader=test_loader, epochs=5)
    return model, test_loader


def _collect_confidently_wrong(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    threshold: float = 0.8,
    target_count: int = 50,
) -> dict:
    model.eval()
    strict_samples = []
    all_wrong_samples = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            probs = softmax(logits)
            confidences, preds = torch.max(probs, dim=1)

            for i in range(inputs.size(0)):
                pred_i = int(preds[i].item())
                true_i = int(targets[i].item())
                conf_i = float(confidences[i].item())

                if pred_i != true_i:
                    sample = {
                        "x": inputs[i].detach().cpu(),
                        "y": true_i,
                        "pred": pred_i,
                        "confidence": conf_i,
                    }
                    all_wrong_samples.append(sample)
                    if conf_i > threshold:
                        strict_samples.append(sample)

    relaxed_threshold = False
    used_samples = strict_samples[:target_count]

    if len(used_samples) < target_count:
        relaxed_threshold = True
        all_wrong_samples = sorted(all_wrong_samples, key=lambda s: s["confidence"], reverse=True)
        used_samples = all_wrong_samples[:target_count]

    if len(used_samples) == 0:
        raise RuntimeError("No wrong predictions found for failure-case analysis.")

    x_tensor = torch.stack([sample["x"] for sample in used_samples], dim=0)
    y_tensor = torch.tensor([sample["y"] for sample in used_samples], dtype=torch.long)
    pred_list = [sample["pred"] for sample in used_samples]
    conf_list = [sample["confidence"] for sample in used_samples]

    print(
        f"Collected {len(used_samples)} failure cases "
        f"(strict threshold > {threshold}: {len(strict_samples)}, threshold relaxed: {relaxed_threshold})"
    )

    return {
        "inputs": x_tensor,
        "targets": y_tensor,
        "predictions": pred_list,
        "confidences": conf_list,
        "threshold": threshold,
        "relaxed_threshold": relaxed_threshold,
        "strict_count": len(strict_samples),
    }


def _evaluate_gradients_on_cases(model: torch.nn.Module, case_data: dict, device: str) -> dict:
    mse_loss_fn = MSELoss()
    ce_loss_fn = CrossEntropyLoss()

    inputs = case_data["inputs"]
    targets = case_data["targets"]

    mse_grad_norms = []
    ce_grad_norms = []
    mse_losses = []
    ce_losses = []
    mse_probs = []
    ce_probs = []
    mse_confidences = []
    ce_confidences = []

    model.eval()

    for i in range(inputs.size(0)):
        x_i = inputs[i : i + 1].to(device)
        y_i = targets[i : i + 1].to(device)

        model.zero_grad()
        logits_mse = model(x_i)
        probs_mse = softmax(logits_mse)
        loss_mse = mse_loss_fn(logits_mse, y_i)
        loss_mse.backward()
        grad_mse = compute_gradient_norm(model)

        mse_grad_norms.append(float(grad_mse))
        mse_losses.append(float(loss_mse.item()))
        mse_probs.append(probs_mse.detach().cpu().squeeze(0).tolist())
        mse_confidences.append(float(torch.max(probs_mse).item()))

        model.zero_grad()
        logits_ce = model(x_i)
        probs_ce = softmax(logits_ce)
        loss_ce = ce_loss_fn(logits_ce, y_i)
        loss_ce.backward()
        grad_ce = compute_gradient_norm(model)

        ce_grad_norms.append(float(grad_ce))
        ce_losses.append(float(loss_ce.item()))
        ce_probs.append(probs_ce.detach().cpu().squeeze(0).tolist())
        ce_confidences.append(float(torch.max(probs_ce).item()))

    mse_mean_grad = float(np.mean(mse_grad_norms))
    ce_mean_grad = float(np.mean(ce_grad_norms))
    grad_ratio = float(ce_mean_grad / max(mse_mean_grad, 1e-12))

    mse_mean_loss = float(np.mean(mse_losses))
    ce_mean_loss = float(np.mean(ce_losses))

    return {
        "num_cases": int(inputs.size(0)),
        "mse": {
            "grad_norms": mse_grad_norms,
            "loss_values": mse_losses,
            "probabilities": mse_probs,
            "confidences": mse_confidences,
        },
        "ce": {
            "grad_norms": ce_grad_norms,
            "loss_values": ce_losses,
            "probabilities": ce_probs,
            "confidences": ce_confidences,
        },
        "true_labels": targets.tolist(),
        "predicted_labels": case_data["predictions"],
        "summary": {
            "mse_mean_grad_norm": mse_mean_grad,
            "ce_mean_grad_norm": ce_mean_grad,
            "grad_ratio_ce_over_mse": grad_ratio,
            "mse_mean_loss": mse_mean_loss,
            "ce_mean_loss": ce_mean_loss,
        },
    }


def main() -> dict:
    print("Experiment 5: Failure Cases (Confident Wrong Predictions)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, test_loader = _train_underfit_model(device)
    case_data = _collect_confidently_wrong(model, test_loader, device=device, threshold=0.8, target_count=50)
    analysis = _evaluate_gradients_on_cases(model, case_data, device=device)

    payload = {
        "num_failure_cases": analysis["num_cases"],
        "confidence_threshold": case_data["threshold"],
        "threshold_relaxed": case_data["relaxed_threshold"],
        "strict_case_count": case_data["strict_count"],
        "mse": analysis["mse"],
        "ce": analysis["ce"],
        "true_labels": analysis["true_labels"],
        "predicted_labels": analysis["predicted_labels"],
        "summary": analysis["summary"],
    }

    save_results(payload, "exp5_failure_cases.json")
    plot_failure_cases(payload)

    print("On confidently wrong predictions:")
    print(f"  MSE mean gradient norm: {analysis['summary']['mse_mean_grad_norm']:.4f}")
    print(f"  CE  mean gradient norm: {analysis['summary']['ce_mean_grad_norm']:.4f}")
    print(f"  Ratio (CE/MSE): {analysis['summary']['grad_ratio_ce_over_mse']:.2f}")
    print(f"  MSE mean loss: {analysis['summary']['mse_mean_loss']:.4f}")
    print(f"  CE  mean loss: {analysis['summary']['ce_mean_loss']:.4f}")

    return payload


if __name__ == "__main__":
    main()
