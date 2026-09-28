"""Evaluation package: metrics to compare anomaly detectors fairly.

Phase 4 introduces :mod:`maschinenwaechter.evaluation.metrics` with the
point-adjust precision/recall/F1 protocol used throughout the anomaly-
detection literature, plus an oracle threshold search (pure numpy).
"""

from maschinenwaechter.evaluation.metrics import find_best_threshold, point_adjust_metrics

__all__ = ["point_adjust_metrics", "find_best_threshold"]
