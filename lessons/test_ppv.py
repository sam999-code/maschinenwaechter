"""Tests for W2-3 PPV-first evaluation."""

from machineguard.ppv import evaluate_alerts


def test_first_trigger_only_counts_once():
    raises = [("m", 15.0), ("m", 15.2), ("m", 15.9)]   # all inside lead of 16h
    r = evaluate_alerts(raises, {"m": 16.0}, lead_h=1.0)
    assert r["TP"] == 1 and r["FP"] == 0 and r["PPV"] == 1.0


def test_alarm_outside_lead_window_is_fp():
    raises = [("m", 10.0), ("m", 15.8)]
    r = evaluate_alerts(raises, {"m": 16.0}, lead_h=1.0)
    assert r["TP"] == 1 and r["FP"] == 1 and r["PPV"] == 0.5


def test_missed_failure_counts_fn():
    raises = [("m", 3.0)]                                # noise only
    r = evaluate_alerts(raises, {"m": 16.0}, lead_h=1.0)
    assert r["TP"] == 0 and r["FN"] == 1 and r["TPR"] == 0.0


def test_lead_time_reported():
    raises = [("m", 14.5)]
    r = evaluate_alerts(raises, {"m": 16.0}, lead_h=3.0)
    assert r["mean_lead_h"] == 1.5


def test_beta_favors_precision():
    # same TP, more FP: F0.5 should drop (precision penalized harder)
    good = evaluate_alerts([("m", 15.5)], {"m": 16.0}, beta=0.5)
    bad = evaluate_alerts([("m", 15.5), ("m", 2.0)], {"m": 16.0}, beta=0.5)
    assert good["F0.5"] > bad["F0.5"]
