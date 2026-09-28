"""Phase 5 experiment: Remaining Useful Life (RUL) regression.

Run from the project root (uses the Phase-4/5 virtualenv):

    .venv/Scripts/python.exe scripts/05_rul_experiment.py

Experimental design — honest generalization testing
---------------------------------------------------
* TRAIN: 5 machines, seeds 101..105, 72 h @ 1 Hz, fault onset staggered
  deterministically at [36, 42, 48, 54, 60] h so the model sees a spread of
  failure times and cannot memorise one countdown curve.
* TEST: 2 held-out machines (seeds 201/202, faults at 44 h / 52 h) the
  model has never seen.
* Labels: RUL = clip(min(cap=60 h, fault_hour - centre_hour), 0), evaluated
  per 30-sample window at its centre (stride 30).

Naive baselines (the model must EARN its place)
-----------------------------------------------
* ``naive_median``: predict the global median of the TRAIN labels for every
  window. The "do-nothing statistics" reference.
* ``always_cap``: predict cap (60 h) for every window — what the capping
  section in ``rul/dataset.py`` says is the floor an honest model cannot
  beat on far-from-failure windows by much.

Outputs
-------
* ``reports/figures/phase5_rul.png`` — one panel per test machine: true RUL
  curve vs. model predictions at window centres, with the cap marked.
* ``reports/figures/phase5_training_loss.png`` — Huber loss curve.
* ``reports/phase5_metrics.csv`` — per test machine: model MAE/bias vs. both
  baselines' MAE.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maschinenwaechter.rul.dataset import build_rul_dataset  # noqa: E402
from maschinenwaechter.rul.model import predict_rul  # noqa: E402
from maschinenwaechter.rul.train import train_rul  # noqa: E402
from maschinenwaechter.simulation.machine import simulate_machine  # noqa: E402

DURATION_HOURS = 72.0
SAMPLING_HZ = 1
WINDOW = 30
STRIDE = 30
CAP = 60.0
TRAIN_SEEDS = [101, 102, 103, 104, 105]
TRAIN_FAULTS = [36.0, 42.0, 48.0, 54.0, 60.0]
TEST_SPECS = [("test_machine_201", 201, 44.0), ("test_machine_202", 202, 52.0)]
EPOCHS = 8
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_CSV = PROJECT_ROOT / "reports" / "phase5_metrics.csv"


def true_rul_curve(hours: np.ndarray, fault_hour: float, cap: float) -> np.ndarray:
    """The label-generating curve RUL(t) = clip(min(cap, fault - t), 0)."""
    return np.clip(np.minimum(cap, fault_hour - hours), 0.0, None)


def plot_rul(test_frames: dict[str, pd.DataFrame],
             test_faults: dict[str, float],
             predictions: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """One panel per test machine: true RUL curve vs. model predictions."""
    fig, axes = plt.subplots(len(test_frames), 1, figsize=(12, 4 * len(test_frames)), sharex=True)
    if len(test_frames) == 1:
        axes = [axes]
    for ax, (name, df) in zip(axes, test_frames.items()):
        fault_hour = test_faults[name]
        hours = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
        ax.plot(hours, true_rul_curve(hours, fault_hour, CAP), color="black", lw=1.5,
                label="true RUL")
        centres, preds = predictions[name]
        ax.scatter(centres, preds, s=8, color="tab:blue", alpha=0.6,
                   label="predicted RUL (window centres)")
        ax.axhline(CAP, color="gray", ls="--", lw=1, label=f"cap = {CAP:.0f} h")
        ax.axvline(fault_hour, color="red", ls=":", lw=1, label="fault onset")
        ax.set_ylabel("RUL (hours)")
        ax.set_title(f"{name} (fault at {fault_hour:.0f} h)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("hours")
    fig.suptitle("Phase 5: Remaining Useful Life prediction on held-out test machines")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase5_rul.png", dpi=150)
    plt.close(fig)


def plot_loss(history: list[float]) -> None:
    """Save the Huber training-loss curve."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(history) + 1), history, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Huber loss (delta=5)")
    ax.set_title("Phase 5: RUL regressor training loss")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase5_training_loss.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the full Phase-5 RUL experiment end to end."""
    started = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Simulating TRAIN fleet (5 machines, faults at 36/42/48/54/60 h) ...")
    train_frames = [
        simulate_machine(DURATION_HOURS, SAMPLING_HZ, fh, seed=s)
        for s, fh in zip(TRAIN_SEEDS, TRAIN_FAULTS)
    ]

    print("Simulating TEST machines (seeds 201/202, held out) ...")
    test_frames: dict[str, pd.DataFrame] = {}
    test_faults: dict[str, float] = {}
    for name, seed, fault in TEST_SPECS:
        test_frames[name] = simulate_machine(DURATION_HOURS, SAMPLING_HZ, fault, seed=seed)
        test_faults[name] = fault

    print(f"Building RUL datasets (window={WINDOW}, stride={STRIDE}, cap={CAP:.0f} h) ...")
    X_train, y_train, _ = build_rul_dataset(train_frames, TRAIN_FAULTS, WINDOW, STRIDE, CAP)
    test_data = {
        name: build_rul_dataset([df], [test_faults[name]], WINDOW, STRIDE, CAP)
        for name, df in test_frames.items()
    }
    print(f"  train windows: {len(X_train):,}")

    print(f"Training RUL regressor (epochs={EPOCHS}, Huber delta=5) ...")
    t0 = time.perf_counter()
    model, history = train_rul(X_train, y_train, epochs=EPOCHS, seed=42)
    train_seconds = time.perf_counter() - t0
    print(f"  trained {len(history)} epochs in {train_seconds:.1f}s; "
          f"loss {history[0]:.4f} -> {history[-1]:.4f}")

    naive_median = float(np.median(y_train))
    print(f"Naive baselines: global train median = {naive_median:.1f} h; always cap = {CAP:.0f} h")

    print("Evaluating on TEST machines ...")
    rows = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (X_test, y_test, centres) in test_data.items():
        preds = predict_rul(model, X_test)
        predictions[name] = (centres, preds)
        mae = float(np.mean(np.abs(preds - y_test)))
        bias = float(np.mean(preds - y_test))
        rows.append({"machine": name, "method": "rul_regressor",
                     "mae_hours": round(mae, 3), "bias_hours": round(bias, 3)})
        for method, constant in (("naive_median", naive_median), ("always_cap", CAP)):
            base_mae = float(np.mean(np.abs(constant - y_test)))
            base_bias = float(constant - np.mean(y_test))
            rows.append({"machine": name, "method": method,
                         "mae_hours": round(base_mae, 3), "bias_hours": round(base_bias, 3)})

    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot_rul(test_frames, test_faults, predictions)
    plot_loss(history)

    print("\n=== Phase 5 results (MAE in hours, lower is better) ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nFigures : {FIGURES_DIR / 'phase5_rul.png'}")
    print(f"          {FIGURES_DIR / 'phase5_training_loss.png'}")
    print(f"Metrics : {METRICS_CSV}")
    print(f"Total wall time: {time.perf_counter() - started:.1f}s "
          f"(training: {train_seconds:.1f}s)")


if __name__ == "__main__":
    main()
