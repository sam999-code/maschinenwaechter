"""Remaining Useful Life (RUL) prediction — Phase 5.

The flagship commercial feature of the product: instead of only saying
*"something is wrong"* (Phases 3–4, anomaly detection), we now answer the
question the plant manager actually asks: **"how many hours until this
machine fails?"**

Because OUR simulator knows when the fault starts (``fault_at_hour``), we can
turn the simulator into a supervised teacher: for every timestamp the label
is the remaining time until fault onset. That is exactly how RUL research
works on real run-to-failure data, where the failure time is known
retrospectively (e.g. the NASA CMAPSS turbofan benchmark).

Public API
----------
* :func:`~maschinenwaechter.rul.dataset.build_rul_dataset` — windows + RUL labels
* :class:`~maschinenwaechter.rul.model.RULRegressor` — LSTM regressor
* :func:`~maschinenwaechter.rul.model.predict_rul` — numpy-in / numpy-out scoring
* :func:`~maschinenwaechter.rul.train.train_rul` — deterministic training loop
"""

from maschinenwaechter.rul.dataset import build_rul_dataset
from maschinenwaechter.rul.model import RULRegressor, predict_rul
from maschinenwaechter.rul.train import train_rul

__all__ = ["build_rul_dataset", "RULRegressor", "predict_rul", "train_rul"]
