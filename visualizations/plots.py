from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.utils import load_results


plt.style.use("seaborn-v0_8-whitegrid")


def _as_numeric_font_size(value: Any, fallback: float) -> float:
    return float(value) if isinstance(value, (int, float)) else fallback


_base_font = _as_numeric_font_size(plt.rcParams.get("font.size", 10.0), 10.0)
_base_axes_title = _as_numeric_font_size(plt.rcParams.get("axes.titlesize", _base_font + 2.0), _base_font + 2.0)
_base_axes_label = _as_numeric_font_size(plt.rcParams.get("axes.labelsize", _base_font), _base_font)
_base_legend = _as_numeric_font_size(plt.rcParams.get("legend.fontsize", _base_font), _base_font)

plt.rcParams.update(
    {
        "font.size": _base_font + 2.0,
        "axes.titlesize": _base_axes_title + 2.0,
        "axes.labelsize": _base_axes_label + 2.0,
        "legend.fontsize": _base_legend + 2.0,
    }
)

MSE_COLOR = "#E74C3C"
CE_COLOR = "#2ECC71"
DPI = 150

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def _save_figure(fig: plt.Figure, filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / filename
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _safe_load(filename: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    return load_results(filename)


def plot_convergence(mse_history: dict, ce_history: dict) -> Path:
    epochs = mse_history.get("epochs") or list(range(1, len(mse_history["train_loss"]) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)

    axes[0, 0].plot(epochs, mse_history["train_loss"], color=MSE_COLOR, label="MSE")
    axes[0, 0].plot(epochs, ce_history["train_loss"], color=CE_COLOR, label="CE")
    axes[0, 0].set_title("Train Loss vs Epoch")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, mse_history["test_loss"], color=MSE_COLOR, label="MSE")
    axes[0, 1].plot(epochs, ce_history["test_loss"], color=CE_COLOR, label="CE")
    axes[0, 1].set_title("Test Loss vs Epoch")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, mse_history["train_accuracy"], color=MSE_COLOR, label="MSE")
    axes[1, 0].plot(epochs, ce_history["train_accuracy"], color=CE_COLOR, label="CE")
    axes[1, 0].set_title("Train Accuracy vs Epoch")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, mse_history["test_accuracy"], color=MSE_COLOR, label="MSE")
    axes[1, 1].plot(epochs, ce_history["test_accuracy"], color=CE_COLOR, label="CE")
    axes[1, 1].set_title("Test Accuracy vs Epoch")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()

    fig.suptitle("Experiment 1: Loss Convergence Comparison")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return _save_figure(fig, "exp1_convergence.png")


def plot_gradient_norms(mse_grad_data: dict, ce_grad_data: dict) -> Path:
    epochs = mse_grad_data["epochs"]

    mse_mean = np.array(mse_grad_data["mean_grad_norm"], dtype=float)
    mse_std = np.array(mse_grad_data["std_grad_norm"], dtype=float)
    ce_mean = np.array(ce_grad_data["mean_grad_norm"], dtype=float)
    ce_std = np.array(ce_grad_data["std_grad_norm"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

    axes[0].plot(epochs, mse_mean, color=MSE_COLOR, label="MSE")
    axes[0].fill_between(epochs, np.maximum(mse_mean - mse_std, 1e-12), mse_mean + mse_std, color=MSE_COLOR, alpha=0.2)
    axes[0].plot(epochs, ce_mean, color=CE_COLOR, label="CE")
    axes[0].fill_between(epochs, np.maximum(ce_mean - ce_std, 1e-12), ce_mean + ce_std, color=CE_COLOR, alpha=0.2)
    axes[0].set_title("Gradient Norm Mean +/- Std")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Gradient Norm")
    axes[0].legend()

    axes[1].plot(epochs, mse_mean, color=MSE_COLOR, label="MSE")
    axes[1].plot(epochs, ce_mean, color=CE_COLOR, label="CE")
    axes[1].set_yscale("log")
    axes[1].set_title("Gradient Norm (Log Scale)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Gradient Norm (log)")
    axes[1].legend()

    fig.suptitle("Experiment 2: Gradient Magnitude Analysis")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return _save_figure(fig, "exp2_gradient_norms.png")


def plot_lr_sensitivity(lr_results: dict) -> Path:
    lrs = sorted([float(k) for k in lr_results.keys()])

    mse_train_loss = [lr_results[str(lr)]["mse"]["train_loss"] for lr in lrs]
    ce_train_loss = [lr_results[str(lr)]["ce"]["train_loss"] for lr in lrs]
    mse_test_acc = [lr_results[str(lr)]["mse"]["acc"] for lr in lrs]
    ce_test_acc = [lr_results[str(lr)]["ce"]["acc"] for lr in lrs]

    mse_divergence = [1 if lr_results[str(lr)]["mse"]["diverged"] else 0 for lr in lrs]
    ce_divergence = [1 if lr_results[str(lr)]["ce"]["diverged"] else 0 for lr in lrs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)

    axes[0, 0].plot(lrs, mse_train_loss, marker="o", color=MSE_COLOR, label="MSE")
    axes[0, 0].plot(lrs, ce_train_loss, marker="o", color=CE_COLOR, label="CE")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_title("Final Train Loss vs Learning Rate")
    axes[0, 0].set_xlabel("Learning Rate")
    axes[0, 0].set_ylabel("Train Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(lrs, mse_test_acc, marker="o", color=MSE_COLOR, label="MSE")
    axes[0, 1].plot(lrs, ce_test_acc, marker="o", color=CE_COLOR, label="CE")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].set_title("Final Test Accuracy vs Learning Rate")
    axes[0, 1].set_xlabel("Learning Rate")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()

    divergence_rates = [np.mean(mse_divergence), np.mean(ce_divergence)]
    axes[1, 0].bar(["MSE", "CE"], divergence_rates, color=[MSE_COLOR, CE_COLOR], alpha=0.85)
    axes[1, 0].set_title("Divergence Rate")
    axes[1, 0].set_ylabel("Fraction Diverged")
    axes[1, 0].set_ylim(0, 1.0)

    heatmap = np.array(
        [
            [np.nan if lr_results[str(lr)]["mse"]["diverged"] else lr_results[str(lr)]["mse"]["acc"] for lr in lrs],
            [np.nan if lr_results[str(lr)]["ce"]["diverged"] else lr_results[str(lr)]["ce"]["acc"] for lr in lrs],
        ],
        dtype=float,
    )
    im = axes[1, 1].imshow(heatmap, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title("Heatmap: Test Accuracy")
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_yticklabels(["MSE", "CE"])
    axes[1, 1].set_xticks(range(len(lrs)))
    axes[1, 1].set_xticklabels([f"{lr:g}" for lr in lrs], rotation=30)
    axes[1, 1].set_xlabel("Learning Rate")
    for r in range(2):
        for c in range(len(lrs)):
            value = heatmap[r, c]
            label = "div" if np.isnan(value) else f"{value:.2f}"
            axes[1, 1].text(c, r, label, ha="center", va="center", color="white")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle("Experiment 3: Learning Rate Sensitivity")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return _save_figure(fig, "exp3_lr_sensitivity.png")


def plot_generalization(mse_history: dict, ce_history: dict) -> Path:
    epochs = mse_history["epochs"]
    mse_gap = np.array(mse_history["test_loss"], dtype=float) - np.array(mse_history["train_loss"], dtype=float)
    ce_gap = np.array(ce_history["test_loss"], dtype=float) - np.array(ce_history["train_loss"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

    axes[0].plot(epochs, mse_history["train_loss"], color=MSE_COLOR, linestyle="-", label="MSE Train")
    axes[0].plot(epochs, mse_history["test_loss"], color=MSE_COLOR, linestyle="--", label="MSE Test")
    axes[0].plot(epochs, ce_history["train_loss"], color=CE_COLOR, linestyle="-", label="CE Train")
    axes[0].plot(epochs, ce_history["test_loss"], color=CE_COLOR, linestyle="--", label="CE Test")
    axes[0].set_title("Train vs Test Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, mse_gap, color=MSE_COLOR, label="MSE Gap")
    axes[1].plot(epochs, ce_gap, color=CE_COLOR, label="CE Gap")
    axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle=":")
    axes[1].set_title("Generalization Gap (Test - Train Loss)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Gap")
    axes[1].legend()

    fig.suptitle("Experiment 4: Generalization Analysis (1000 training samples)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return _save_figure(fig, "exp4_generalization.png")


def plot_failure_cases(failure_data: dict) -> Path:
    mse_grad = np.array(failure_data["mse"]["grad_norms"], dtype=float)
    ce_grad = np.array(failure_data["ce"]["grad_norms"], dtype=float)
    mse_conf = np.array(failure_data["mse"]["confidences"], dtype=float)
    ce_conf = np.array(failure_data["ce"]["confidences"], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=DPI)

    axes[0].bar(["MSE", "CE"], [mse_grad.mean(), ce_grad.mean()], color=[MSE_COLOR, CE_COLOR], alpha=0.9)
    axes[0].set_title("Mean Gradient Norm")
    axes[0].set_ylabel("Gradient Norm")

    axes[1].scatter(mse_conf, mse_grad, color=MSE_COLOR, alpha=0.7, label="MSE")
    axes[1].scatter(ce_conf, ce_grad, color=CE_COLOR, alpha=0.7, label="CE")
    axes[1].set_title("Confidence vs Gradient Norm")
    axes[1].set_xlabel("Softmax Confidence")
    axes[1].set_ylabel("Gradient Norm")
    axes[1].legend()

    axes[2].boxplot([mse_grad, ce_grad], labels=["MSE", "CE"], patch_artist=True)
    axes[2].set_title("Gradient Norm Distribution")
    axes[2].set_ylabel("Gradient Norm")

    fig.suptitle("Experiment 5: Gradient Behavior on Confident Wrong Predictions")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return _save_figure(fig, "exp5_failure_cases.png")


def plot_summary_dashboard() -> Path:
    exp1_mse = _safe_load("exp1_mse_history.json")
    exp1_ce = _safe_load("exp1_ce_history.json")
    exp2_mse = _safe_load("exp2_mse_gradients.json")
    exp2_ce = _safe_load("exp2_ce_gradients.json")
    exp3 = _safe_load("exp3_lr_sensitivity.json")
    exp4 = _safe_load("exp4_generalization.json")
    exp5 = _safe_load("exp5_failure_cases.json")

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=DPI)

    # Row 1
    if exp1_mse and exp1_ce:
        epochs = exp1_mse["epochs"]
        axes[0, 0].plot(epochs, exp1_mse["test_loss"], color=MSE_COLOR, label="MSE")
        axes[0, 0].plot(epochs, exp1_ce["test_loss"], color=CE_COLOR, label="CE")
        axes[0, 0].set_title("Exp1: Test Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()

        axes[0, 1].plot(epochs, exp1_mse["test_accuracy"], color=MSE_COLOR, label="MSE")
        axes[0, 1].plot(epochs, exp1_ce["test_accuracy"], color=CE_COLOR, label="CE")
        axes[0, 1].set_title("Exp1: Test Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend()
    else:
        axes[0, 0].text(0.5, 0.5, "Missing Exp1 data", ha="center", va="center")
        axes[0, 1].text(0.5, 0.5, "Missing Exp1 data", ha="center", va="center")

    if exp2_mse and exp2_ce:
        axes[0, 2].plot(exp2_mse["epochs"], exp2_mse["mean_grad_norm"], color=MSE_COLOR, label="MSE")
        axes[0, 2].plot(exp2_ce["epochs"], exp2_ce["mean_grad_norm"], color=CE_COLOR, label="CE")
        axes[0, 2].set_title("Exp2: Mean Grad Norm")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].legend()
    else:
        axes[0, 2].text(0.5, 0.5, "Missing Exp2 data", ha="center", va="center")

    # Row 2
    if exp3:
        lrs = sorted([float(k) for k in exp3.keys()])
        mse_acc = [exp3[str(lr)]["mse"]["acc"] for lr in lrs]
        ce_acc = [exp3[str(lr)]["ce"]["acc"] for lr in lrs]
        mse_train = [exp3[str(lr)]["mse"]["train_loss"] for lr in lrs]
        ce_train = [exp3[str(lr)]["ce"]["train_loss"] for lr in lrs]
        mse_div = np.mean([1 if exp3[str(lr)]["mse"]["diverged"] else 0 for lr in lrs])
        ce_div = np.mean([1 if exp3[str(lr)]["ce"]["diverged"] else 0 for lr in lrs])

        axes[1, 0].plot(lrs, mse_acc, marker="o", color=MSE_COLOR, label="MSE")
        axes[1, 0].plot(lrs, ce_acc, marker="o", color=CE_COLOR, label="CE")
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_title("Exp3: Test Accuracy")
        axes[1, 0].set_xlabel("LR")
        axes[1, 0].legend()

        axes[1, 1].plot(lrs, mse_train, marker="o", color=MSE_COLOR, label="MSE")
        axes[1, 1].plot(lrs, ce_train, marker="o", color=CE_COLOR, label="CE")
        axes[1, 1].set_xscale("log")
        axes[1, 1].set_title("Exp3: Train Loss")
        axes[1, 1].set_xlabel("LR")
        axes[1, 1].legend()

        axes[1, 2].bar(["MSE", "CE"], [mse_div, ce_div], color=[MSE_COLOR, CE_COLOR])
        axes[1, 2].set_ylim(0, 1.0)
        axes[1, 2].set_title("Exp3: Divergence Rate")
    else:
        axes[1, 0].text(0.5, 0.5, "Missing Exp3 data", ha="center", va="center")
        axes[1, 1].text(0.5, 0.5, "Missing Exp3 data", ha="center", va="center")
        axes[1, 2].text(0.5, 0.5, "Missing Exp3 data", ha="center", va="center")

    # Row 3
    if exp4 and exp4.get("mse_history") and exp4.get("ce_history"):
        epochs = exp4["mse_history"]["epochs"]
        mse_gap = np.array(exp4["mse_history"]["test_loss"]) - np.array(exp4["mse_history"]["train_loss"])
        ce_gap = np.array(exp4["ce_history"]["test_loss"]) - np.array(exp4["ce_history"]["train_loss"])
        axes[2, 0].plot(epochs, mse_gap, color=MSE_COLOR, label="MSE")
        axes[2, 0].plot(epochs, ce_gap, color=CE_COLOR, label="CE")
        axes[2, 0].set_title("Exp4: Generalization Gap")
        axes[2, 0].set_xlabel("Epoch")
        axes[2, 0].legend()
    else:
        axes[2, 0].text(0.5, 0.5, "Missing Exp4 data", ha="center", va="center")

    if exp5:
        mse_grad = np.array(exp5["mse"]["grad_norms"], dtype=float)
        ce_grad = np.array(exp5["ce"]["grad_norms"], dtype=float)
        axes[2, 1].boxplot([mse_grad, ce_grad], labels=["MSE", "CE"], patch_artist=True)
        axes[2, 1].set_title("Exp5: Failure-Case Grads")

        summary = exp5.get("summary", {})
        lines = [
            "Key Metrics",
            f"CE/MSE grad ratio: {summary.get('grad_ratio_ce_over_mse', float('nan')):.2f}",
            f"MSE final test loss (Exp1): {exp1_mse['test_loss'][-1]:.4f}" if exp1_mse else "MSE final test loss: n/a",
            f"CE final test loss (Exp1): {exp1_ce['test_loss'][-1]:.4f}" if exp1_ce else "CE final test loss: n/a",
            f"MSE final gap (Exp4): {exp4.get('final_mse_gap', float('nan')):.4f}" if exp4 else "MSE final gap: n/a",
            f"CE final gap (Exp4): {exp4.get('final_ce_gap', float('nan')):.4f}" if exp4 else "CE final gap: n/a",
        ]
        axes[2, 2].axis("off")
        axes[2, 2].text(0.02, 0.98, "\n".join(lines), va="top", ha="left")
    else:
        axes[2, 1].text(0.5, 0.5, "Missing Exp5 data", ha="center", va="center")
        axes[2, 2].text(0.5, 0.5, "Missing Exp5 data", ha="center", va="center")

    fig.suptitle("Loss Function Analysis: Summary Dashboard")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return _save_figure(fig, "summary_dashboard.png")

