"""Tests for W2-4 calibration + verification."""

import numpy as np
import pytest

from machineguard.alarms import AlertRule, AlarmEvent
from machineguard.calibration import calibrate_rule, verify_alerts, measured_ppv

TEMPLATE = AlertRule(rule_id="t", asset="a", failure_mode="f", cause="c",
                     consequence="x", action="act", priority="high",
                     raise_score=3.0, clear_score=2.0)


def noisy_stream(seed):
    rng = np.random.default_rng(seed)
    hours = np.arange(0, 24, 10 / 3600)
    errors = np.abs(rng.normal(0, 1, len(hours)))
    return errors, hours


def test_calibration_adapts_to_machine_noise():
    """A noisier machine must get a HIGHER raise threshold."""
    quiet, hours = noisy_stream(1)
    loud, _ = noisy_stream(2)
    loud = loud * 4
    rq = calibrate_rule(TEMPLATE, quiet, hours)
    rl = calibrate_rule(TEMPLATE, loud, hours)
    assert rl.raise_score > rq.raise_score


def test_calibration_keeps_hysteresis_ratio():
    errors, hours = noisy_stream(3)
    r = calibrate_rule(TEMPLATE, errors, hours)
    assert r.clear_score == pytest.approx(r.raise_score * 0.8, abs=1e-3)


def test_calibration_rejects_too_little_data():
    errors = np.array([1.0, 2.0])
    hours = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        calibrate_rule(TEMPLATE, errors, hours)


def test_verify_marks_confirmed_true():
    # signal high until hour 12, then returns to normal; alarm clears at 12.5
    errors = np.concatenate([np.full(200, 5.0), np.full(200, 0.1)])
    hours = np.linspace(0, 24, len(errors))
    ev = [AlarmEvent("t", 5.0, "raise", 5.0, "high"),
          AlarmEvent("t", 12.5, "clear", 0.1, "high")]
    v = verify_alerts(ev, errors, hours, TEMPLATE,
                      dispositions={"t": "true_positive"})
    assert v[0]["verdict"] == "confirmed_true"


def test_verify_marks_suspicious_when_signal_stuck():
    errors = np.full(400, 5.0)                    # never returns to normal
    hours = np.linspace(0, 24, len(errors))
    ev = [AlarmEvent("t", 5.0, "raise", 5.0, "high")]
    v = verify_alerts(ev, errors, hours, TEMPLATE,
                      dispositions={"t": "true_positive"})
    assert v[0]["verdict"] == "suspicious"


def test_measured_ppv_none_until_verified():
    assert measured_ppv([{"verdict": "unverifiable"}]) is None
    assert measured_ppv([{"verdict": "confirmed_true"},
                         {"verdict": "confirmed_false"}]) == 0.5
