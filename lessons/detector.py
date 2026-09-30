"""Our detector as a clean, reusable function (extracted from Lesson 3).

This is the moment a script becomes SOFTWARE:
instead of code that runs once, we have a function we can
trust, test, and reuse.
"""

import numpy as np
import pandas as pd


def rolling_zscore(signal, window=200, threshold=3.0):
    """Flag suspicious points in a 1-D signal.

    A point is flagged when it is more than `threshold` rolling
    standard-deviations away from the rolling mean.

    Parameters
    ----------
    signal : array-like of numbers (the sensor time series)
    window : how many past points define "normal right now"
    threshold : z-score above which we raise an alert

    Returns
    -------
    z : np.ndarray      z-score per point (NaN during warm-up window)
    alert : np.ndarray  boolean array, True = alert
    """
    s = pd.Series(np.asarray(signal, dtype=float))
    rolling_mean = s.rolling(window=window).mean()
    rolling_std = s.rolling(window=window).std().replace(0.0, np.nan).fillna(1.0)

    z = ((s - rolling_mean) / rolling_std).abs()
    # .to_numpy(copy=True): pandas hands us read-only arrays by default,
    # and we want to WRITE into them below (warm-up masking).
    alert = (z > threshold).fillna(False).to_numpy(copy=True)
    z = z.to_numpy(copy=True)
    # during the warm-up window there is no full window yet -> not alertable
    alert[:window] = False
    z[:window] = np.nan
    return z, alert
