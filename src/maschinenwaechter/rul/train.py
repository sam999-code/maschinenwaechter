"""Deterministic training loop for the RUL regressor (Phase 5).

Mirrors the Phase 4 training discipline (seeded torch + numpy, seeded
DataLoader generator, early stopping on plateau) so the whole project has
ONE reproducibility story.

Loss: ``nn.HuberLoss(delta=5)`` — quadratic within ±5 h of the label,
linear beyond. See the model docstring for why Huber is the RUL standard:
the capped plateau plus the descent to 0 creates exactly the heavy-tail
residual distribution that L2 alone handles poorly.

Scaling
-------
Inputs are standardised per sensor with the mean/std of the TRAINING
windows (fitted here, stored on the model). The three sensors differ by
two orders of magnitude (see Phase 4 notes); unscaled inputs would let the
acoustic channel dominate every gradient.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from maschinenwaechter.rul.model import RULRegressor


def set_seed(seed: int) -> None:
    """Seed torch and numpy so experiments are bit-for-bit reproducible."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over all (window, timestep) positions."""
    return X.mean(axis=(0, 1)), X.std(axis=(0, 1)) + 1e-8


def train_rul(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 8,
    lr: float = 1e-3,
    batch: int = 64,
    seed: int = 42,
    hidden_size: int = 32,
) -> tuple[RULRegressor, list[float]]:
    """Train an :class:`RULRegressor` on ``(X, y)`` regression data.

    Parameters
    ----------
    X:
        Training windows, ``(n, window, 3)`` raw sensor values.
    y:
        RUL labels in hours, ``(n,)`` — typically capped (see
        ``rul/dataset.py``).
    epochs:
        Maximum training epochs.
    lr:
        Adam learning rate.
    batch:
        Mini-batch size.
    seed:
        Seeds torch, numpy and the DataLoader shuffle generator.
    hidden_size:
        LSTM hidden dimension of the regressor.

    Returns
    -------
    model : RULRegressor
        Trained model with ``scaler_mean_`` / ``scaler_std_`` and the
        resolved hyperparameters attached.
    history : list[float]
        Mean Huber loss per epoch (for the loss-curve figure).
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"X must be (n, window, features), got shape {X.shape}.")
    if len(X) != len(y):
        raise ValueError(f"X and y disagree on the number of windows: {len(X)} vs {len(y)}.")

    set_seed(seed)
    mean, std = fit_scaler(X)
    scaled = ((X - mean) / std).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(scaled), torch.from_numpy(y)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, generator=generator)

    model = RULRegressor(input_size=X.shape[-1], hidden_size=hidden_size)
    model.scaler_mean_ = mean
    model.scaler_std_ = std
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=5.0)

    history: list[float] = []
    best, stale = np.inf, 0
    patience, min_delta = 2, 1e-4
    for _ in range(epochs):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            n += len(xb)
        epoch_loss = total / n
        history.append(epoch_loss)
        # Early stop on plateau: no relative improvement for patience epochs.
        # Same guard as Phase 4: the np.isfinite check covers the first
        # epoch, where best == inf would otherwise be treated as a plateau
        # (inf - delta is still inf, so any finite loss should reset it).
        if not np.isfinite(best) or epoch_loss < best - min_delta * max(abs(best), 1e-12):
            best, stale = epoch_loss, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return model, history
