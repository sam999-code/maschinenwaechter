"""Supervised RUL dataset construction (Phase 5).

From simulator frames to ``(X, y, centers)`` regression data.

Labelling rule
--------------
Our simulator knows the fault-onset hour ``fault_at_hour``, so every timestamp
gets a label::

    RUL(t) = clip(fault_at_hour - t_hours, 0, cap)

i.e. hours until failure, never negative (after the fault the machine is
"dead", RUL = 0) and never above ``cap``.

RUL capping (standard practice — teach this!)
---------------------------------------------
Far-from-failure timestamps all receive the cap value instead of their true
(often huge) RUL. This is deliberate and standard in the RUL literature:

* Pre-fault degradation is only *observable* in the last stretch before
  failure; 200 h before a bearing dies, the sensors genuinely carry no
  information about how long it will live. Training the model to predict
  "187.4 h" from a healthy-looking window teaches it noise.
* Capping focuses the model's capacity on the operationally relevant range
  ("will it die in the next ~60 h?") and stabilises training.
* The cost is a floor on achievable MAE for early windows: any prediction
  in ``[0, cap]`` is at least ``cap - true_rul`` wrong there. Honest
  baselines (predict the cap for everything) expose exactly this.

Window centres and leakage
--------------------------
A window of 30 samples is labelled with the RUL at its *centre* sample and
placed at the centre's hour in ``centers``. Windows are allowed to *cross*
the fault hour — that is fine, their centre simply sits on the decreasing
part of the RUL curve.

A subtler point: a window that lies entirely BEFORE the fault is labelled
with a value that was computed using knowledge of the (future) fault time.
This is **label leakage in the pedagogical sense, but it is correct here**:
at training time the teacher (the simulator / the historical record) knows
when the failure happened, and every supervised RUL dataset in the
literature is built exactly this way. In *deployment* we run the trained
model forward-only: it sees the most recent window and predicts RUL now, so
no future information is needed or used at inference time.

Determinism
-----------
Machines are processed in list order, windows in index order, and the
randomness lives only in the simulator seeds — same inputs give bit-for-bit
identical ``X``, ``y`` and ``centers``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from maschinenwaechter.detection.autoencoder import DEFAULT_SENSOR_COLUMNS

SENSOR_COLUMNS = DEFAULT_SENSOR_COLUMNS  # vibration, temperature, acoustic (rpm excluded, as in Phase 3/4)


def _frame_hours(df: pd.DataFrame) -> np.ndarray:
    """Hour values of each row, derived from the monotonic ``timestamp`` column."""
    seconds = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    return seconds / 3600.0


def build_rul_dataset(
    machines: list[pd.DataFrame],
    fault_hours: list[float],
    window: int = 30,
    stride: int = 30,
    cap: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a supervised RUL dataset from simulated machine frames.

    Parameters
    ----------
    machines:
        List of frames from :func:`maschinenwaechter.simulation.simulate_machine`,
        each with a ``timestamp`` column and the three sensor columns
        (vibration, temperature, acoustic).
    fault_hours:
        ``fault_at_hour`` of each machine, parallel to ``machines``. The
        fault-onset hour is the supervised "time of death" the labels count
        down to.
    window:
        Window length in samples (30 samples @ 1 Hz = 30 s of context).
    stride:
        Step between consecutive windows in samples. ``stride == window``
        gives non-overlapping windows (~144 windows per machine-hour at
        1 Hz); smaller strides give more (correlated) training data.
    cap:
        RUL cap in hours — see the module docstring for why capping is
        standard practice.

    Returns
    -------
    X : np.ndarray, shape (n_windows, window, 3), float32
        Sensor windows in deterministic order (machine order, then window
        index order).
    y : np.ndarray, shape (n_windows,), float32
        RUL label of each window, computed at the window's CENTRE hour:
        ``clip(min(cap, fault_hour - centre_hour), 0, None)``.
    centres : np.ndarray, shape (n_windows,), float64
        The hour value of each window's centre sample — the time axis for
        plotting and evaluation.

    Raises
    ------
    ValueError
        If ``len(machines) != len(fault_hours)``, or a machine is shorter
        than one window.
    """
    if len(machines) != len(fault_hours):
        raise ValueError(
            f"machines and fault_hours must be parallel lists, got {len(machines)} and {len(fault_hours)}."
        )
    if window < 2:
        raise ValueError("window must be at least 2 samples.")
    if stride < 1:
        raise ValueError("stride must be >= 1.")
    if cap <= 0:
        raise ValueError("cap must be positive.")

    xs, ys, centres = [], [], []
    for df, fault_hour in zip(machines, fault_hours):
        missing = [c for c in SENSOR_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"machine frame is missing sensor columns: {missing}")
        array = df[list(SENSOR_COLUMNS)].to_numpy(dtype=np.float32)
        if len(array) < window:
            raise ValueError(
                f"machine run has {len(array)} samples; need at least window={window}."
            )
        hours = _frame_hours(df)

        n_windows = 1 + (len(array) - window) // stride
        for i in range(n_windows):
            start = i * stride
            centre_idx = start + window // 2
            centre_hour = float(hours[centre_idx])
            label = min(cap, fault_hour - centre_hour)
            label = max(0.0, label)
            xs.append(array[start : start + window])
            ys.append(label)
            centres.append(centre_hour)

    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    centres_arr = np.asarray(centres, dtype=np.float64)
    return X, y, centres_arr
