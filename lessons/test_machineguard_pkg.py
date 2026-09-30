"""Tests for the machineguard package.

Run from the lessons/ folder:
  ..\\.venv\\Scripts\\python.exe -m pytest test_machineguard_pkg.py -v
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from machineguard import (simulate_machine, make_windows, fit_scaler,
                          apply_scaler, train_autoencoder, score_windows,
                          calibrate_threshold, evaluate, sustained_alerts)


def test_simulator_is_deterministic():
    a = simulate_machine(seed=42, fault_hour=16.0)
    b = simulate_machine(seed=42, fault_hour=16.0)
    assert a.equals(b), "same seed must give identical machine"


def test_simulator_fault_column_and_physics():
    df = simulate_machine(seed=1, fault_hour=2.0, hours=4)
    assert set(df["fault"].unique()) == {0, 1}
    pre = df[df["fault"] == 0]["vibration"]
    post = df[df["fault"] == 1]["vibration"]
    assert post.std() > pre.std()  # wear makes vibration stronger


def test_scaler_uses_healthy_stats_only():
    df = simulate_machine(seed=3, fault_hour=2.0, hours=4)
    healthy = df[df["fault"] == 0]
    mean, std = fit_scaler(healthy)
    # healthy data scaled with its own stats -> mean ~0
    assert abs(apply_scaler(healthy, mean, std).mean()) < 1e-9


def test_end_to_end_detects_fault_on_new_machine():
    """Train on machine A, detect on machine B - the Lesson 8 protocol,
    in miniature so the test stays fast (<60s)."""
    A = simulate_machine(seed=42, fault_hour=2.0, hours=4)
    B = simulate_machine(seed=7, fault_hour=2.5, hours=4)
    mean, std = fit_scaler(A[A["fault"] == 0])
    train_w = make_windows(apply_scaler(A[A["fault"] == 0], mean, std))
    model = train_autoencoder(train_w, hidden=8, latent=4, epochs=2)
    thr = calibrate_threshold(score_windows(model, train_w), 99)

    test_w = make_windows(apply_scaler(B, mean, std))
    hours = B["hour"].to_numpy()[30::10][: len(test_w)]
    errors = score_windows(model, test_w)
    alarm = errors > thr
    metrics = evaluate(alarm, hours >= 2.5)
    assert metrics["recall"] > 0.5, f"missed the fault: {metrics}"
    assert metrics["precision"] > 0.5, f"too many false alarms: {metrics}"


def test_sustained_alerts_kills_single_points():
    alarm = np.zeros(20, dtype=bool)
    alarm[3] = True                      # single noise blip
    alarm[10:14] = True                  # 4 in a row
    alarm[17:20] = True                  # only 3 (end of array)
    out = sustained_alerts(alarm, run_length=3)
    assert not out[3]                    # blip killed
    assert out[10:14].all()              # real run kept
    assert out[17:20].all()              # run at the end counts too
    assert out.sum() == 7
