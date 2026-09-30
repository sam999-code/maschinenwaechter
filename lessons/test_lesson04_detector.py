"""Lesson 4: Unit tests for our detector — proof that it works.

Run with:  .venv\\Scripts\\python.exe -m pytest lessons/test_lesson04_detector.py -v

Why tests? In Lesson 3 we looked at a plot and said "looks good".
That's vibes. A test is a PROMISE that stays true even after we
change the code next month. German engineering culture runs on this.
"""

import numpy as np

from detector import rolling_zscore


def test_calm_signal_gives_few_alerts():
    """A healthy signal (small noise only) must give ALMOST no alerts.

    Statistics lesson: pure noise still crosses 3-sigma ~0.3% of the
    time by chance. So the correct promise is not "zero alarms" but
    "a very low alarm rate". This is exactly why production systems
    alert on *runs* of flags, not single points."""
    rng = np.random.default_rng(seed=1)
    calm = rng.normal(0.0, 0.2, size=1000)  # pure noise, no fault
    _, alert = rolling_zscore(calm, window=200, threshold=3.0)
    alarm_rate = alert.sum() / len(calm)
    assert alarm_rate < 0.01  # under 1% false alarms on healthy data


def test_spike_is_detected():
    """One huge spike in a calm signal MUST be flagged."""
    calm = np.zeros(1000)
    calm[500] = 10.0  # the knock!
    _, alert = rolling_zscore(calm, window=200, threshold=3.0)
    assert alert[500] is True or bool(alert[500]) is True
    # and the flag must be exactly at the spike, not everywhere
    assert alert.sum() < 10


def test_slow_drift_is_NOT_flagged_documents_blind_spot():
    """Documents a KNOWN limitation: a very slow drift escapes the
    rolling detector (the window absorbs it). This test is on purpose —
    it pins the blind spot so nobody forgets it. Real PdM software
    adds a separate trend alarm for exactly this reason."""
    drift = np.linspace(0.0, 1.0, 2000)  # slow, steady rise
    _, alert = rolling_zscore(drift, window=200, threshold=3.0)
    assert alert.sum() == 0  # blind spot confirmed — and now DOCUMENTED


def test_same_seed_same_result():
    """Determinism: same input -> same output. ML must be reproducible."""
    rng1 = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=42)
    a = rng1.normal(0.0, 0.2, 500)
    b = rng2.normal(0.0, 0.2, 500)
    _, alert_a = rolling_zscore(a)
    _, alert_b = rolling_zscore(b)
    assert np.array_equal(alert_a, alert_b)
