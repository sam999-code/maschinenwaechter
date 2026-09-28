"""Evaluation metrics for anomaly detection (Phase 4).

Why point adjustment?
---------------------
Anomaly labels in time series are usually *segment*-level: a human engineer
marks "the bearing was worn from 09:00 to 11:00", but a detector will rarely
flag every single sample inside the segment (the fault signature comes and
goes — think of the impulsive shocks in our simulator). Under plain
point-wise recall, such a detector is punished unfairly.

The standard point-adjust protocol (Xu et al., Unsupervised Anomaly
Detection via VAE, WWW 2018) fixes this:

* **Recall side (adjusted):** a ground-truth anomalous segment counts as
  detected if *any* point inside it is flagged; then ALL its points count
  as true positives.
* **Precision side (point-wise):** a flagged point inside a healthy region
  is still a false positive, one by one — no free passes for flooding the
  timeline with alarms.

This makes recall generous and precision strict, so a good F1 under this
protocol is meaningful. We report it honestly alongside the raw counts.

Everything here is pure numpy — no scikit-learn needed at runtime.
"""

from __future__ import annotations

import numpy as np


def _adjusted_counts(y_true: np.ndarray, flagged: np.ndarray) -> tuple[int, int, int]:
    """Return (tp, fp, fn) under the point-adjust protocol."""
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(flagged).astype(bool)

    tp = int((flagged & y_true).sum())
    fp = int((flagged & ~y_true).sum())
    # Undetected anomalous segments: a whole segment counts as missed when
    # no point inside it was flagged. Collect segments, then their lengths.
    segments: list[tuple[int, int]] = []
    seg_start = None
    for i, true in enumerate(y_true):
        if true and seg_start is None:
            seg_start = i
        elif not true and seg_start is not None:
            segments.append((seg_start, i))
            seg_start = None
    if seg_start is not None:
        segments.append((seg_start, len(y_true)))
    fn = sum(end - start for start, end in segments if not flagged[start:end].any())
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def point_adjust_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict:
    """Precision / recall / F1 with point-adjusted segment recall.

    Parameters
    ----------
    y_true:
        Binary labels (1 = anomalous segment) per sample.
    y_score:
        Continuous anomaly scores per sample; higher = more anomalous.
    threshold:
        Scores >= threshold are flagged.

    Returns
    -------
    dict
        ``tp, fp, fn, precision, recall, f1, threshold, n_flagged``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape.")
    flagged = y_score >= threshold
    tp, fp, fn = _adjusted_counts(y_true, flagged)
    precision, recall, f1 = _prf(tp, fp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": float(threshold),
        "n_flagged": int(flagged.sum()),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    quantile_grid: np.ndarray | None = None,
) -> dict:
    """Search score quantiles for the threshold with the best point-adjusted F1.

    This is the *oracle* threshold (it peeks at the labels) — useful for
    comparing models at their individual best. For a realistic deployment
    estimate, calibrate the threshold on healthy data only (see
    ``scripts/04_compare_models.py``).

    Parameters
    ----------
    quantile_grid:
        Quantile levels to try; defaults to 199 levels from 0.50 to 0.999.
    """
    y_score = np.asarray(y_score, dtype=float)
    if quantile_grid is None:
        quantile_grid = np.linspace(0.50, 0.999, 199)
    best: dict = {"f1": -1.0}
    for q in quantile_grid:
        threshold = float(np.quantile(y_score, q))
        m = point_adjust_metrics(y_true, y_score, threshold)
        if m["f1"] > best["f1"]:
            best = {**m, "quantile": float(q)}
    return best
