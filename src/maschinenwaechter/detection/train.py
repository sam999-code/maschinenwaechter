"""Training, scoring and persistence for the LSTM-Autoencoder (Phase 4).

Train-on-healthy principle
--------------------------
The model is trained ONLY on windows from rows where ``fault == 0``. This is
the classic semi-supervised anomaly-detection setup: healthy examples are
abundant (machines run fine most of the time) and labelled faults are rare
and expensive to collect. The model memorises what "normal" looks like;
everything that reconstructs poorly is suspicious.

Scaling
-------
The three sensors live on very different scales (vibration ~1 mm/s,
temperature ~60 °C, acoustic ~70 dB), so raw MSE would be dominated by the
largest-scale feature. We therefore standardise each feature with the
mean/std of the *healthy training data* and persist those statistics next
to the weights. Scoring applies the same transform.

Persistence is pickle-free on our side: weights go through ``torch.save``
(a zip of tensors) and hyperparameters + scaler stats go into a companion
JSON file, so model artefacts stay inspectable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from maschinenwaechter.detection.autoencoder import DEFAULT_SENSOR_COLUMNS, LSTMAutoencoder
from maschinenwaechter.detection.dataset import WindowedDataset

DEFAULT_CONFIG: dict = {
    "window_size": 30,
    "hidden_size": 32,
    "epochs": 8,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "patience": 2,          # early-stop after this many non-improving epochs
    "min_delta": 1e-4,      # relative improvement that counts as progress
    "seed": 42,
    "sensor_columns": list(DEFAULT_SENSOR_COLUMNS),
}


def set_seed(seed: int) -> None:
    """Seed torch and numpy so experiments are bit-for-bit reproducible."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def healthy_sensor_array(
    df: pd.DataFrame, sensor_columns: tuple[str, ...] = DEFAULT_SENSOR_COLUMNS
) -> np.ndarray:
    """Return the sensor matrix of rows where ``fault == 0`` as float32."""
    missing = [c for c in sensor_columns if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame is missing sensor columns: {missing}")
    healthy = df.loc[df["fault"] == 0, list(sensor_columns)]
    if len(healthy) < 2:
        raise ValueError("need at least 2 healthy rows to train.")
    return healthy.to_numpy(dtype=np.float32)


def fit_scaler(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std of the healthy training data."""
    return array.mean(axis=0), array.std(axis=0) + 1e-8


def train_autoencoder(
    df: pd.DataFrame, config: dict | None = None
) -> tuple[LSTMAutoencoder, list[float]]:
    """Train an LSTMAutoencoder on the healthy rows of ``df``.

    Parameters
    ----------
    df:
        Machine frame (needs ``fault`` plus the sensor columns).
    config:
        Overrides for any key of :data:`DEFAULT_CONFIG`.

    Returns
    -------
    model : LSTMAutoencoder
        Trained model (carries the scaler as ``model.scaler_mean_`` /
        ``model.scaler_std_`` and the resolved config as ``model.config_``).
    history : list[float]
        Mean training loss per epoch (for the loss-curve figure).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    set_seed(cfg["seed"])
    sensors = tuple(cfg["sensor_columns"])

    healthy = healthy_sensor_array(df, sensors)
    mean, std = fit_scaler(healthy)
    scaled = (healthy - mean) / std

    dataset = WindowedDataset(scaled, cfg["window_size"], stride=1)
    generator = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, generator=generator)

    model = LSTMAutoencoder(input_size=len(sensors), hidden_size=cfg["hidden_size"])
    model.scaler_mean_ = mean
    model.scaler_std_ = std
    model.config_ = cfg
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.MSELoss()

    history: list[float] = []
    best, stale = np.inf, 0
    for _ in range(cfg["epochs"]):
        model.train()
        total, n = 0.0, 0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
            n += len(batch)
        epoch_loss = total / n
        history.append(epoch_loss)
        # Early stop on plateau: no relative improvement for patience epochs.
        # (the np.isfinite guard covers the first epoch, where best == inf)
        if not np.isfinite(best) or epoch_loss < best - cfg["min_delta"] * max(abs(best), 1e-12):
            best, stale = epoch_loss, 0
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    return model, history


def score_series(
    model: LSTMAutoencoder,
    df: pd.DataFrame,
    sensor_columns: tuple[str, ...] = DEFAULT_SENSOR_COLUMNS,
    stride: int = 1,
) -> pd.Series:
    """Per-timestamp reconstruction error aligned with ``df``.

    Window errors are placed at the window's *centre* sample; the warmup
    edges have no window and stay NaN until filled by ffill/bfill.

    Returns
    -------
    pandas.Series
        One float per input row (index aligned to ``df``). Higher = more
        anomalous.
    """
    array = df[list(sensor_columns)].to_numpy(dtype=np.float32)
    scaled = ((array - model.scaler_mean_) / model.scaler_std_).astype(np.float32)
    win = model.config_["window_size"]
    # stride-1 windows are a contiguous sliding view — no Python loop needed
    windows = np.lib.stride_tricks.sliding_window_view(scaled, (win, scaled.shape[1]))
    windows = windows.squeeze(axis=1)[::stride]  # (n_windows, win, features)
    window_errors, _ = model.reconstruction_errors(windows)

    scores = np.full(len(df), np.nan)
    centers = np.arange(len(window_errors)) * stride + win // 2
    scores[centers] = window_errors
    series = pd.Series(scores, index=df.index, name="reconstruction_error")
    return series.ffill().bfill()


def save_autoencoder(model: LSTMAutoencoder, path: str | Path) -> Path:
    """Save weights (``torch.save``) plus a JSON of hyperparams + scaler."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    meta = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in model.config_.items()},
        "scaler_mean": model.scaler_mean_.tolist(),
        "scaler_std": model.scaler_std_.tolist(),
        "state_dict": path.name,
    }
    meta_path = path.with_suffix(path.suffix + ".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_autoencoder(path: str | Path) -> tuple[LSTMAutoencoder, dict]:
    """Inverse of :func:`save_autoencoder`; returns ``(model, meta)``."""
    path = Path(path)
    meta = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    model = LSTMAutoencoder(
        input_size=len(cfg["sensor_columns"]), hidden_size=cfg["hidden_size"]
    )
    model.load_state_dict(torch.load(path, weights_only=True))
    model.scaler_mean_ = np.asarray(meta["scaler_mean"], dtype=np.float32)
    model.scaler_std_ = np.asarray(meta["scaler_std"], dtype=np.float32)
    model.config_ = cfg
    model.eval()
    return model, meta
