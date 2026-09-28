"""Phase 4 experiment: LSTM-Autoencoder vs. rolling z-score baseline.

Run from the project root (uses the Phase-4 virtualenv):

    .venv/Scripts/python.exe scripts/04_compare_models.py

Experimental design — honest generalization testing
---------------------------------------------------
* Thresholds for BOTH methods are calibrated on the test machine's healthy
  portion only (first 35 h, fault starts at 40 h) at the 99.5th percentile
  of the healthy scores — no labels are used for calibration, which mimics
  deployment where only healthy history is available.

Outputs
-------
* ``reports/figures/phase4_comparison.png`` — vibration + fault shading +
  both detectors' flagged points (two stacked panels).
* ``reports/figures/phase4_training_loss.png`` — autoencoder loss curve.
* ``reports/phase4_metrics.csv`` — method, precision, recall, F1,
  false_alarms_per_hour, plus a printed table.
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

from maschinenwaechter.detection.baseline import RollingZScoreDetector  # noqa: E402
from maschinenwaechter.detection.train import score_series, train_autoencoder  # noqa: E402
from maschinenwaechter.evaluation.metrics import point_adjust_metrics  # noqa: E402
from maschinenwaechter.simulation.machine import simulate_machine  # noqa: E402

DURATION_HOURS = 72.0
SAMPLING_HZ = 1
FAULT_AT_HOUR = 40.0
CALIBRATION_HOURS = 35.0  # healthy part of the TEST machine used for thresholds
CALIBRATION_QUANTILE = 0.995
TRAIN_SEED = 7
TEST_SEED = 11
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_CSV = PROJECT_ROOT / "reports" / "phase4_metrics.csv"


def calibrate_threshold(scores: pd.Series, hours: np.ndarray) -> float:
    """99.5th percentile of scores inside the healthy calibration window."""
    healthy_scores = scores[hours < CALIBRATION_HOURS].to_numpy(dtype=float)
    return float(np.quantile(healthy_scores, CALIBRATION_QUANTILE))


def evaluate(method: str, y_true: np.ndarray, scores: np.ndarray, threshold: float,
             hours: np.ndarray) -> dict:
    """Point-adjusted metrics + false-alarm rate (FPs per healthy hour)."""
    m = point_adjust_metrics(y_true, scores, threshold)
    healthy_hours = float(hours[y_true == 0].max() - hours[y_true == 0].min())
    return {
        "method": method,
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "false_alarms_per_hour": round(m["fp"] / healthy_hours, 4),
        "threshold": round(threshold, 4),
        "fp": m["fp"],
    }


def plot_comparison(test_df: pd.DataFrame, baseline_score: pd.Series,
                    ae_score: pd.Series, base_flag: np.ndarray,
                    ae_flag: np.ndarray) -> None:
    """Two stacked panels: vibration + baseline flags, then AE error + flags."""
    hours = (np.arange(len(test_df)) / 3600.0 / SAMPLING_HZ)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    panels = (
        (axes[0], test_df["vibration_mm_s"].to_numpy(), baseline_score.to_numpy(),
         base_flag, "Rolling z-score baseline", "vibration (mm/s)"),
        (axes[1], ae_score.to_numpy(), ae_score.to_numpy(),
         ae_flag, "LSTM-Autoencoder reconstruction error", "reconstruction error (MSE)"),
    )
    for ax, signal, score, flag, title, ylabel in panels:
        ax.plot(hours, signal, color="0.75", lw=0.5, label="signal")
        ax.axvspan(FAULT_AT_HOUR, DURATION_HOURS, color="red", alpha=0.10,
                   label="true fault region")
        ax.scatter(hours[flag], score[flag], s=6, color="crimson",
                   label=f"flagged ({int(flag.sum())})")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
    axes[1].set_xlabel("hours")
    fig.suptitle("Phase 4: classical baseline vs. LSTM-Autoencoder (test machine, seed 11)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase4_comparison.png", dpi=150)
    plt.close(fig)


def plot_loss(history: list[float]) -> None:
    """Save the autoencoder training-loss curve."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(history) + 1), history, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE reconstruction loss (healthy train machine)")
    ax.set_title("Phase 4: LSTM-Autoencoder training loss")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase4_training_loss.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the full Phase-4 comparison end to end."""
    started = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Simulating train machine (seed 7) and test machine (seed 11) ...")
    train_df = simulate_machine(DURATION_HOURS, SAMPLING_HZ, FAULT_AT_HOUR, seed=TRAIN_SEED)
    test_df = simulate_machine(DURATION_HOURS, SAMPLING_HZ, FAULT_AT_HOUR, seed=TEST_SEED)
    hours = np.arange(len(test_df)) / 3600.0 / SAMPLING_HZ
    y_true = test_df["fault"].to_numpy()

    print("Scoring with the Phase-3 rolling z-score baseline ...")
    detector = RollingZScoreDetector(window=300, threshold=3.5)
    baseline_result = detector.fit_transform(test_df)
    baseline_score = baseline_result["anomaly_score"]
    baseline_threshold = calibrate_threshold(baseline_score, hours)

    print("Training LSTM-Autoencoder on the healthy part of the train machine ...")
    t0 = time.perf_counter()
    model, history = train_autoencoder(train_df, config={"epochs": 8})
    train_seconds = time.perf_counter() - t0
    print(f"  trained {len(history)} epochs in {train_seconds:.1f}s; "
          f"loss {history[0]:.4f} -> {history[-1]:.4f}")
    ae_score = score_series(model, test_df)
    ae_threshold = calibrate_threshold(ae_score, hours)

    print("Evaluating both detectors on the full test machine ...")
    rows = [
        evaluate("rolling_zscore", y_true, baseline_score.to_numpy(), baseline_threshold, hours),
        evaluate("lstm_autoencoder", y_true, ae_score.to_numpy(), ae_threshold, hours),
    ]
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    base_flag = baseline_score.to_numpy() >= baseline_threshold
    ae_flag = ae_score.to_numpy() >= ae_threshold
    plot_comparison(test_df, baseline_score, ae_score, base_flag, ae_flag)
    plot_loss(history)

    print("\n=== Phase 4 results (point-adjusted, threshold @ 99.5 pct of healthy) ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nFigures : {FIGURES_DIR / 'phase4_comparison.png'}")
    print(f"          {FIGURES_DIR / 'phase4_training_loss.png'}")
    print(f"Metrics : {METRICS_CSV}")
    print(f"Total wall time: {time.perf_counter() - started:.1f}s "
          f"(training: {train_seconds:.1f}s)")


if __name__ == "__main__":
    main()
