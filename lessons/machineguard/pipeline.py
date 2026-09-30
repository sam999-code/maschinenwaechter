"""The pipeline: all the steps between raw CSV and verdict, as functions.

Each function is small, tested, and composable. This module is the
"recipe" of the whole product - the CLI is just a thin shell over it.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .model import LSTMAutoencoder

SENSORS = ["vibration", "temperature"]


# --- windows & scaling -----------------------------------------------------
def make_windows(data, window=60, stride=10):
    idx = np.arange(0, len(data) - window, stride)
    return np.stack([data[i : i + window] for i in idx])


def fit_scaler(df_healthy):
    """Scaler statistics from HEALTHY rows only (Lesson 6: no leakage)."""
    mean = df_healthy[SENSORS].mean().to_numpy()
    std = df_healthy[SENSORS].std().to_numpy()
    return mean, std


def apply_scaler(df, mean, std):
    return ((df[SENSORS] - mean) / std).to_numpy()


# --- training & scoring ----------------------------------------------------
def train_autoencoder(train_windows, hidden=16, latent=8, epochs=6,
                      lr=1e-3, batch=256, seed=42):
    torch.manual_seed(seed)
    model = LSTMAutoencoder(n_sensors=train_windows.shape[2],
                            hidden=hidden, latent=latent)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(train_windows, dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        for i in range(0, len(X), batch):
            b = X[i : i + batch]
            loss = F.mse_loss(model(b), b)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def score_windows(model, windows):
    """Reconstruction MSE per window."""
    with torch.no_grad():
        return np.concatenate([
            F.mse_loss(model(torch.tensor(windows[i : i + 1024], dtype=torch.float32)),
                       torch.tensor(windows[i : i + 1024], dtype=torch.float32),
                       reduction="none").mean(dim=(1, 2)).numpy()
            for i in range(0, len(windows), 1024)
        ])


def calibrate_threshold(healthy_errors, percentile=99):
    return float(np.percentile(healthy_errors, percentile))


# --- evaluation ------------------------------------------------------------
def evaluate(alarm, truth):
    tp = np.sum(alarm & truth); fp = np.sum(alarm & ~truth); fn = np.sum(~alarm & truth)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def sustained_alerts(alarm, run_length=5):
    """Production alarm logic: True only where `run_length` CONSECUTIVE
    windows are all flagged. Kills single-point noise alarms (Lesson 9)."""
    alarm = alarm.astype(int)
    kernel = np.ones(run_length, dtype=int)
    runs = np.convolve(alarm, kernel, mode="valid")
    sustained = runs == run_length
    out = np.zeros_like(alarm)
    for i, s in enumerate(sustained):
        if s:
            out[i : i + run_length] = True
    return out.astype(bool)


# --- bundle save/load (the model + its scaler + threshold travel together) -
def save_bundle(model, mean, std, threshold, window, stride, path_prefix):
    import json
    torch.save(model.state_dict(), f"{path_prefix}_weights.pt")
    with open(f"{path_prefix}_config.json", "w") as f:
        json.dump({"sensors": SENSORS, "window": window, "stride": stride,
                   "mean": mean.tolist(), "std": std.tolist(),
                   "threshold": threshold}, f, indent=2)


def load_bundle(path_prefix):
    import json
    with open(f"{path_prefix}_config.json") as f:
        cfg = json.load(f)
    model = LSTMAutoencoder(n_sensors=len(cfg["sensors"]))
    model.load_state_dict(torch.load(f"{path_prefix}_weights.pt",
                                     weights_only=True))
    model.eval()
    return model, cfg
