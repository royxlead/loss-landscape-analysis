from __future__ import annotations

import importlib
import time
import traceback
from pathlib import Path

from src.utils import load_results
from visualizations.plots import (
    plot_convergence,
    plot_failure_cases,
    plot_generalization,
    plot_gradient_norms,
    plot_lr_sensitivity,
    plot_summary_dashboard,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

EXPERIMENTS = [
    (1, "Convergence", "experiments.exp1_convergence"),
    (2, "Gradient Magnitude", "experiments.exp2_gradient_magnitude"),
    (3, "Learning Rate Sensitivity", "experiments.exp3_lr_sensitivity"),
    (4, "Generalization", "experiments.exp4_generalization"),
    (5, "Failure Cases", "experiments.exp5_failure_cases"),
]


def _ascii_header() -> str:
    return r"""
 _     ___  ____  ____    _____ _   _ _   _  ____ _____ ___ ___  _   _
| |   / _ \/ ___|/ ___|  |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | |
| |  | | | \___ \___ \  | |_  | | | |  \| | |     | |  | | | | |  \| |
| |__| |_| |___) |__) | |  _| | |_| | |\  | |___  | |  | | |_| | |\  |
|_____\___/|____/____/  |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|

                     LOSS FUNCTION ANALYSIS
"""


def _first_epoch_below(values: list[float], threshold: float) -> int | None:
    for idx, value in enumerate(values, start=1):
        if value < threshold:
            return idx
    return None


def _run_experiments() -> dict:
    status = {}

    for exp_num, exp_name, module_path in EXPERIMENTS:
        print(f"Running Experiment {exp_num}: {exp_name}...")
        start = time.time()

        try:
            module = importlib.import_module(module_path)
            if not hasattr(module, "main"):
                raise AttributeError(f"{module_path} has no main()")
            result = module.main()
            elapsed = time.time() - start
            print(f"[OK] Experiment {exp_num} complete. Time: {elapsed:.2f}s")
            status[f"exp{exp_num}"] = {"ok": True, "time_s": elapsed, "result": result}
        except Exception as exc:
            elapsed = time.time() - start
            print(f"[FAIL] Experiment {exp_num} failed. Time: {elapsed:.2f}s")
            print(f"  Error: {exc}")
            traceback.print_exc()
            status[f"exp{exp_num}"] = {"ok": False, "time_s": elapsed, "error": str(exc)}

    return status


def _regenerate_all_plots() -> None:
    print("Regenerating plots from saved JSON results...")

    try:
        mse_hist = load_results("exp1_mse_history.json")
        ce_hist = load_results("exp1_ce_history.json")
        plot_convergence(mse_hist, ce_hist)
        print("  Generated: exp1_convergence.png")
    except Exception as exc:
        print(f"  Skipped exp1 plot: {exc}")

    try:
        mse_grad = load_results("exp2_mse_gradients.json")
        ce_grad = load_results("exp2_ce_gradients.json")
        plot_gradient_norms(mse_grad, ce_grad)
        print("  Generated: exp2_gradient_norms.png")
    except Exception as exc:
        print(f"  Skipped exp2 plot: {exc}")

    try:
        lr_results = load_results("exp3_lr_sensitivity.json")
        plot_lr_sensitivity(lr_results)
        print("  Generated: exp3_lr_sensitivity.png")
    except Exception as exc:
        print(f"  Skipped exp3 plot: {exc}")

    try:
        gen_results = load_results("exp4_generalization.json")
        plot_generalization(gen_results["mse_history"], gen_results["ce_history"])
        print("  Generated: exp4_generalization.png")
    except Exception as exc:
        print(f"  Skipped exp4 plot: {exc}")

    try:
        failure_results = load_results("exp5_failure_cases.json")
        plot_failure_cases(failure_results)
        print("  Generated: exp5_failure_cases.png")
    except Exception as exc:
        print(f"  Skipped exp5 plot: {exc}")

    try:
        plot_summary_dashboard()
        print("  Generated: summary_dashboard.png")
    except Exception as exc:
        print(f"  Skipped summary dashboard: {exc}")


def _print_final_summary_table() -> None:
    print("\nFinal Summary of Key Findings")
    print("-" * 72)

    try:
        mse_hist = load_results("exp1_mse_history.json")
        ce_hist = load_results("exp1_ce_history.json")
        mse_epoch = _first_epoch_below(mse_hist["test_loss"], 0.5)
        ce_epoch = _first_epoch_below(ce_hist["test_loss"], 0.5)
        print(f"Exp1 | First epoch test loss < 0.5 | MSE: {mse_epoch} | CE: {ce_epoch}")
    except Exception:
        print("Exp1 | Not available")

    try:
        mse_grad = load_results("exp2_mse_gradients.json")
        ce_grad = load_results("exp2_ce_gradients.json")
        ratio5 = ce_grad["mean_grad_norm"][4] / max(mse_grad["mean_grad_norm"][4], 1e-12)
        ratio20 = ce_grad["mean_grad_norm"][19] / max(mse_grad["mean_grad_norm"][19], 1e-12)
        print(f"Exp2 | CE/MSE mean grad norm ratio | epoch5: {ratio5:.2f} | epoch20: {ratio20:.2f}")
    except Exception:
        print("Exp2 | Not available")

    try:
        exp3 = load_results("exp3_lr_sensitivity.json")
        lrs = sorted([float(k) for k in exp3.keys()])
        mse_div = sum(1 for lr in lrs if exp3[str(lr)]["mse"]["diverged"])
        ce_div = sum(1 for lr in lrs if exp3[str(lr)]["ce"]["diverged"])
        best_mse_lr = max(lrs, key=lambda x: exp3[str(x)]["mse"]["acc"])
        best_ce_lr = max(lrs, key=lambda x: exp3[str(x)]["ce"]["acc"])
        print(
            "Exp3 | Best accuracy and divergence "
            f"| MSE best LR={best_mse_lr:g}, acc={exp3[str(best_mse_lr)]['mse']['acc']:.4f}, div={mse_div}/{len(lrs)} "
            f"| CE best LR={best_ce_lr:g}, acc={exp3[str(best_ce_lr)]['ce']['acc']:.4f}, div={ce_div}/{len(lrs)}"
        )
    except Exception:
        print("Exp3 | Not available")

    try:
        exp4 = load_results("exp4_generalization.json")
        print(
            "Exp4 | Final generalization gap "
            f"| MSE: {exp4['final_mse_gap']:.4f} "
            f"| CE: {exp4['final_ce_gap']:.4f} "
            f"| Better: {exp4['better_generalization']}"
        )
    except Exception:
        print("Exp4 | Not available")

    try:
        exp5 = load_results("exp5_failure_cases.json")
        summary = exp5["summary"]
        print(
            "Exp5 | Failure-case gradient strength "
            f"| MSE grad: {summary['mse_mean_grad_norm']:.4f} "
            f"| CE grad: {summary['ce_mean_grad_norm']:.4f} "
            f"| Ratio CE/MSE: {summary['grad_ratio_ce_over_mse']:.2f}"
        )
    except Exception:
        print("Exp5 | Not available")

    print("-" * 72)


def main() -> None:
    start_total = time.time()

    print(_ascii_header())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _run_experiments()
    _regenerate_all_plots()
    _print_final_summary_table()

    total_runtime = time.time() - start_total
    print(f"Total runtime: {total_runtime:.2f}s")


if __name__ == "__main__":
    main()
