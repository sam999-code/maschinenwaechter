"""Phase 5 contract tests: RUL dataset construction, deterministic training,
model-beats-naive generalization, and finite predictions on healthy data.

The whole module is skipped when torch is unavailable (same discipline as
``tests/test_autoencoder.py``): the base teaching environment stays
torch-free, the .venv runs these tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from maschinenwaechter.rul.dataset import build_rul_dataset  # noqa: E402
from maschinenwaechter.rul.model import RULRegressor, predict_rul  # noqa: E402
from maschinenwaechter.rul.train import train_rul  # noqa: E402
from maschinenwaechter.simulation.machine import simulate_machine  # noqa: E402


def handmade_frame(duration_hours: float = 2.0, sampling_hz: int = 1) -> pd.DataFrame:
    """A tiny synthetic frame with exact timestamps (no simulator noise).

    Sensor values are a deterministic ramp so the test can reason about
    exact sample counts: 2 h @ 1 Hz = 7200 rows.
    """
    n = int(duration_hours * 3600 * sampling_hz)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq=pd.Timedelta(seconds=1 / sampling_hz)),
            "vibration_mm_s": np.linspace(0.0, 1.0, n),
            "temperature_c": np.linspace(50.0, 51.0, n),
            "acoustic_db": np.linspace(70.0, 71.0, n),
        }
    )


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def test_rul_dataset_shapes():
    """2 h @ 1 Hz, window 30, stride 30 -> 240 windows of (30, 3)."""
    df = handmade_frame()
    X, y, centres = build_rul_dataset([df], [1.0], window=30, stride=30, cap=60.0)

    n_windows = 1 + (7200 - 30) // 30
    assert X.shape == (n_windows, 30, 3)
    assert y.shape == (n_windows,)
    assert centres.shape == (n_windows,)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_rul_labels_respect_cap_and_zero_floor():
    """Labels never exceed the cap and never go below 0."""
    df = handmade_frame()
    _, y, _ = build_rul_dataset([df], [1.0], window=30, stride=30, cap=0.5)

    assert y.max() <= 0.5 + 1e-6   # capped plateau early in the run
    assert y.min() >= 0.0          # never negative after the fault


def test_rul_label_at_centre_matches_formula():
    """Handmade case: every label equals clip(min(cap, fault - centre), 0)."""
    df = handmade_frame()
    fault_hour = 1.0
    cap = 60.0
    window, stride = 30, 30
    X, y, centres = build_rul_dataset([df], [fault_hour], window=window, stride=stride, cap=cap)

    assert len(y) > 0
    for i in range(len(y)):
        expected = min(cap, fault_hour - centres[i])
        expected = max(0.0, expected)
        assert y[i] == pytest.approx(expected, abs=1e-4)
    # The centre of the first window sits at sample 15 -> 15/3600 h:
    assert centres[0] == pytest.approx(15 / 3600.0, abs=1e-9)
    assert y[0] == pytest.approx(fault_hour - 15 / 3600.0, abs=1e-3)


def test_rul_dataset_deterministic_ordering():
    """Same inputs twice -> bit-for-bit identical arrays."""
    df = handmade_frame()
    a = build_rul_dataset([df], [1.0])
    b = build_rul_dataset([df], [1.0])
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])


def test_rul_dataset_length_mismatch_raises():
    df = handmade_frame()
    with pytest.raises(ValueError, match="parallel"):
        build_rul_dataset([df], [1.0, 2.0])


# ---------------------------------------------------------------------------
# Deterministic training
# ---------------------------------------------------------------------------

def tiny_train_data(seed: int = 7):
    """Three small simulated machines (12 h, faults at 6/8/10 h) as arrays."""
    frames = [
        simulate_machine(12.0, 1, fh, seed=seed + i)
        for i, fh in enumerate([6.0, 8.0, 10.0])
    ]
    X, y, _ = build_rul_dataset(frames, [6.0, 8.0, 10.0], window=30, stride=120, cap=60.0)
    return X, y


def test_same_seed_same_first_epoch_loss():
    """Seeding discipline: identical seed -> identical first-epoch loss."""
    X, y = tiny_train_data()
    _, h1 = train_rul(X, y, epochs=1, hidden_size=16, seed=42)
    _, h2 = train_rul(X, y, epochs=1, hidden_size=16, seed=42)
    assert h1[0] == pytest.approx(h2[0], rel=1e-6)


# ---------------------------------------------------------------------------
# The model must beat the naive baseline to earn its place
# ---------------------------------------------------------------------------

def test_model_beats_naive_median():
    """On a 4th held-out machine the regressor's MAE < predict-the-median MAE.

    Margins are intentionally modest (strict inequality only): this is a
    contract test, not a benchmark — it must never flake.
    """
    X_train, y_train = tiny_train_data()
    model, _ = train_rul(X_train, y_train, epochs=6, hidden_size=16, seed=42)

    test_df = simulate_machine(12.0, 1, 7.0, seed=99)
    X_test, y_test, _ = build_rul_dataset([test_df], [7.0], window=30, stride=120, cap=60.0)
    preds = predict_rul(model, X_test)

    model_mae = float(np.mean(np.abs(preds - y_test)))
    naive_mae = float(np.mean(np.abs(np.median(y_train) - y_test)))
    assert model_mae < naive_mae


def test_predictions_finite_on_healthy_stretch():
    """No NaN/inf predictions on windows from a fully healthy machine."""
    X_train, y_train = tiny_train_data()
    model, _ = train_rul(X_train, y_train, epochs=2, hidden_size=16, seed=42)

    healthy = simulate_machine(12.0, 1, None, seed=555)
    X_h, _, _ = build_rul_dataset([healthy], [24.0], window=30, stride=120, cap=60.0)
    preds = predict_rul(model, X_h)
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0.0)  # predict_rul clips at 0


def test_predict_requires_scaler():
    """An untrained bare model must refuse to predict (guard against silent bugs)."""
    model = RULRegressor()
    with pytest.raises(RuntimeError, match="scaler"):
        predict_rul(model, np.zeros((2, 30, 3), dtype=np.float32))
