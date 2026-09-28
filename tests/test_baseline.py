"""Tests for the Phase 3 rolling z-score baseline detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maschinenwaechter.detection import RollingZScoreDetector

WINDOW = 300        # ~30 s at the 10 Hz test sampling rate
THRESHOLD = 3.5


@pytest.fixture(scope="module")
def detector():
    """One detector instance shared by the module's tests."""
    return RollingZScoreDetector(window=WINDOW, threshold=THRESHOLD)


@pytest.fixture(scope="module")
def scored_faulty(detector, faulty_df):
    """The faulty fixture, scored once and reused."""
    return detector.fit_transform(faulty_df)


def test_anomaly_score_shape_matches_input(scored_faulty, faulty_df):
    """Every input row must receive exactly one anomaly score."""
    assert len(scored_faulty["anomaly_score"]) == len(faulty_df)


def test_anomaly_score_is_nonnegative_float(scored_faulty):
    """Scores are absolute z values: floats >= 0."""
    scores = scored_faulty["anomaly_score"].to_numpy()
    assert scores.dtype.kind == "f"
    assert np.all(scores >= 0.0)


def test_detector_fires_within_30min_after_fault(scored_faulty):
    """At least one alarm must occur within 30 min after fault onset.

    This is the core behavioural contract: the baseline must notice the
    injected bearing_wear fault quickly, not eventually.
    """
    onset = scored_faulty.loc[scored_faulty["fault"] == 1, "timestamp"].min()
    window_end = onset + pd.Timedelta(minutes=30)
    early_alarms = scored_faulty.loc[
        (scored_faulty["timestamp"] >= onset) & (scored_faulty["timestamp"] <= window_end),
        "is_anomaly",
    ]
    assert early_alarms.any(), "detector stayed silent for 30 min after the fault"


def test_healthy_data_few_false_positives(detector, healthy_df):
    """Healthy data must produce < 5 % flagged points.

    Rolling statistics are estimated from noisy data, so a zero false-alarm
    rate is unrealistic — but a useful baseline must be far below 5 %.
    """
    scored = detector.fit_transform(healthy_df)
    flagged_fraction = scored["is_anomaly"].mean()
    assert flagged_fraction < 0.05


def test_is_anomaly_matches_threshold(scored_faulty):
    """The boolean flag must be exactly `score > threshold`."""
    expected = scored_faulty["anomaly_score"] > THRESHOLD
    assert (scored_faulty["is_anomaly"] == expected).all()


def test_missing_sensor_column_raises(detector, healthy_df):
    """A frame without the monitored columns must fail loudly, not silently."""
    with pytest.raises(KeyError):
        detector.fit_transform(healthy_df.drop(columns=["vibration_mm_s"]))


def test_warmup_rows_are_not_flagged(scored_faulty):
    """Before the rolling window fills, scores are 0 -> no spurious alarms."""
    assert not scored_faulty["is_anomaly"].iloc[: WINDOW - 1].any()
