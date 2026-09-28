"""Phase 4 tests: LSTM-Autoencoder architecture, training and scoring.

These tests only run when torch is installed (``pytest.importorskip``), so
the suite still passes on machines without the Phase-4 environment. Run
with the project venv:

    .venv/Scripts/python.exe -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from maschinenwaechter.detection.autoencoder import LSTMAutoencoder  # noqa: E402
from maschinenwaechter.detection.dataset import WindowedDataset  # noqa: E402
from maschinenwaechter.detection.train import score_series, train_autoencoder  # noqa: E402
from maschinenwaechter.simulation.machine import simulate_machine  # noqa: E402

SENSORS = ["vibration_mm_s", "temperature_c", "acoustic_db"]


def synthetic_frame(n_samples: int = 120, seed: int = 0) -> pd.DataFrame:
    """A tiny deterministic healthy 'machine': sines + noise in all channels."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)
    frame = pd.DataFrame(
        {
            c: np.sin(2 * np.pi * t / 20.0 + i) + 0.1 * rng.standard_normal(n_samples)
            for i, c in enumerate(SENSORS)
        }
    )
    frame.insert(0, "timestamp", pd.date_range("2026-01-01", periods=n_samples, freq="s"))
    frame["fault"] = 0
    return frame


def test_dataset_window_mapping():
    """Window index -> start/center mapping must line up with the data."""
    data = np.arange(10, dtype=np.float32).reshape(10, 1).repeat(3, axis=1)
    ds = WindowedDataset(data, window_size=4, stride=2)
    assert len(ds) == 4  # starts at 0, 2, 4, 6
    assert ds.window_start(2) == 4
    assert ds.window_center(2) == 6
    window = ds[2][0].numpy()
    assert np.allclose(window[:, 0], [4, 5, 6, 7])


def test_autoencoder_overfits_tiny_signal():
    """Architecture sanity check: the model CAN memorise a simple signal.

    If 15 epochs cannot drive the training reconstruction error of a
    low-dimensional sine signal below 0.2, the architecture is broken —
    this is the cheapest possible proof-of-life for the LSTM-AE.
    """
    df = synthetic_frame(n_samples=120, seed=0)
    model, history = train_autoencoder(
        df,
        config={
            "window_size": 30,
            "hidden_size": 16,
            "epochs": 15,
            "batch_size": 16,
            "learning_rate": 1e-2,  # tiny dataset -> fewer steps -> higher lr
            "seed": 1,
        },
    )
    scaled = (df[SENSORS].to_numpy(np.float32) - model.scaler_mean_) / model.scaler_std_
    windows = np.lib.stride_tricks.sliding_window_view(scaled, (30, 3)).squeeze(1)
    errors, per_step = model.reconstruction_errors(windows)
    assert errors.mean() < 0.2, f"failed to overfit, mean error {errors.mean():.3f}"
    assert per_step.shape == (len(windows), 30)
    assert len(history) <= 15


def test_score_series_alignment_and_warmup():
    """Output length == input rows; warmup edges filled, no NaNs left."""
    df = simulate_machine(duration_hours=1.0, sampling_hz=1, fault_at_hour=0.7, seed=99)
    model, _ = train_autoencoder(
        df, config={"window_size": 30, "hidden_size": 8, "epochs": 2, "seed": 2}
    )
    scores = score_series(model, df)
    assert len(scores) == len(df)
    assert not scores.isna().any(), "warmup edges must be ffill/bfill-filled"
    # Edge values come from the nearest available window error, so they must
    # be inside the observed error range.
    assert scores.min() >= 0.0


def test_training_is_deterministic():
    """Same seed -> identical first-epoch loss (within float tolerance)."""
    df = synthetic_frame(n_samples=150, seed=3)
    config = {"window_size": 20, "hidden_size": 8, "epochs": 1, "seed": 42}
    _, history_a = train_autoencoder(df, config)
    _, history_b = train_autoencoder(df, config)
    assert history_a[0] == pytest.approx(history_b[0], abs=1e-6)


def test_detector_separates_healthy_from_faulty():
    """End-to-end: on a small seeded machine, faulty windows score higher.

    Uses the Mann-Whitney U statistic via numpy only (no scipy/sklearn): when
    almost every healthy score is *below* almost every faulty score, the
    healthy rank sum (and thus U_healthy) is close to zero.
    """
    df = simulate_machine(duration_hours=6.0, sampling_hz=1, fault_at_hour=3.0, seed=5)
    model, _ = train_autoencoder(
        df, config={"window_size": 30, "hidden_size": 16, "epochs": 6, "seed": 3}
    )
    scores = score_series(model, df).to_numpy()
    healthy = scores[df["fault"].to_numpy() == 0]
    faulty = scores[df["fault"].to_numpy() == 1]
    assert np.median(healthy) < np.median(faulty)

    # Mann-Whitney U on the "healthy < faulty" ordering (pure numpy).
    order = np.argsort(np.concatenate([healthy, faulty]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    u_healthy = ranks[: len(healthy)].sum() - len(healthy) * (len(healthy) + 1) / 2
    u_max = len(healthy) * len(faulty)
    assert u_healthy / u_max < 0.1, f"too much overlap (U ratio {u_healthy / u_max:.2f})"
