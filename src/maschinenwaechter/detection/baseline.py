"""Classical rolling z-score anomaly detector (Phase 3 baseline).

Why a rolling z-score?
----------------------
A z-score measures how many standard deviations a value sits away from the
local mean:

    z(t) = |x(t) - mean_window(x)| / std_window(x)

Using *rolling* (moving-window) statistics instead of global ones is
crucial for machine sensors: machines drift with load, ambient temperature
and wear-in, so a global mean/std would quickly become stale. The rolling
window (default 300 samples ≈ 5 minutes at 10 Hz) adapts to slow, legitimate
change while still flagging abrupt departures from *recent* behaviour.

Why the max across sensors?
---------------------------
Each sensor column gets its own z-score (they have different units — mm/s,
°C, dB — so scores are unit-free and comparable). A point is anomalous if
*any* sensor deviates strongly, so the per-point anomaly score is the
maximum z-score across the monitored columns. This is an "any-sensor
raises the alarm" fusion rule: simple, interpretable, and hard for a
single faulty sensor to hide from.

This baseline deliberately uses **only numpy and pandas** — no scikit-learn,
no torch — so the learner can see exactly what later models must beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Sensor columns the detector monitors by default. `rpm` is excluded on
# purpose: load changes legitimately swing rpm, and we do not want the
# alarm keyed to production scheduling rather than machine health.
DEFAULT_SENSOR_COLUMNS = ("vibration_mm_s", "temperature_c", "acoustic_db")


class RollingZScoreDetector:
    """Flag anomalous samples via rolling z-scores across sensor columns.

    Parameters
    ----------
    window:
        Rolling window length in samples. Should be (much) longer than the
        transients you want to detect but short enough to track slow drift.
    threshold:
        Alarm threshold on the fused anomaly score.
    sensor_columns:
        Columns to monitor; defaults to vibration, temperature, acoustic.

    Attributes (after :meth:`fit_transform`)
    ----------------------------------------
    anomaly_score_ : np.ndarray
        Non-negative float per input row: max z-score across sensors.
    is_anomaly_ : np.ndarray
        Boolean per input row: ``anomaly_score_ > threshold``.
    """

    def __init__(
        self,
        window: int = 300,
        threshold: float = 3.5,
        sensor_columns: tuple[str, ...] = DEFAULT_SENSOR_COLUMNS,
    ) -> None:
        if window < 2:
            raise ValueError("window must be at least 2 (need a meaningful std).")
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        self.window = int(window)
        self.threshold = float(threshold)
        self.sensor_columns = tuple(sensor_columns)
        self.anomaly_score_: np.ndarray | None = None
        self.is_anomaly_: np.ndarray | None = None

    def _zscores(self, series: pd.Series) -> pd.Series:
        """Per-point rolling z-score of one sensor column.

        mean/std use the *trailing* window ending at the current sample
        (``min_periods=window`` so early samples, where the window is not
        yet full, do not produce misleading tiny standard deviations).
        """
        rolling = series.rolling(window=self.window, min_periods=self.window)
        mean = rolling.mean()
        std = rolling.std(ddof=0)  # population std: stable for small windows
        z = (series - mean).abs() / std.replace(0.0, np.nan)
        return z.fillna(0.0)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score every row of ``df`` and store the results.

        Parameters
        ----------
        df:
            DataFrame containing all ``sensor_columns``.

        Returns
        -------
        pandas.DataFrame
            Copy of the input with two added columns: ``anomaly_score``
            (float, max z across sensors) and ``is_anomaly`` (bool).
        """
        missing = [c for c in self.sensor_columns if c not in df.columns]
        if missing:
            raise KeyError(f"DataFrame is missing sensor columns: {missing}")

        z_matrix = np.column_stack([self._zscores(df[c]) for c in self.sensor_columns])
        self.anomaly_score_ = z_matrix.max(axis=1)
        self.is_anomaly_ = self.anomaly_score_ > self.threshold

        result = df.copy()
        result["anomaly_score"] = self.anomaly_score_
        result["is_anomaly"] = self.is_anomaly_
        return result
