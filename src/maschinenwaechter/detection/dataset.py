"""Sliding-window dataset for reconstruction training (Phase 4).

Turns a raw sensor array ``(n_samples, n_features)`` into overlapping
windows ``(window_size, n_features)`` that the LSTM-Autoencoder learns to
reconstruct. Windows are the *training examples*; each one is both input
and target, which is why the dataset returns single-element tuples.

Stride matters pedagogically: ``stride=1`` gives maximally many (overlapping,
correlated) windows — best data efficiency for training; larger strides
speed up scoring at the cost of temporal resolution.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowedDataset(Dataset):
    """Sliding windows over a 2-D sensor array for reconstruction training.

    Parameters
    ----------
    data:
        Array ``(n_samples, n_features)``. Should already be restricted to
        the sensor columns the model sees (typically vibration, temperature,
        acoustic) and restricted to *healthy* rows for training.
    window_size:
        Length T of each window in samples.
    stride:
        Step between consecutive windows (1 = every possible window).

    Attributes
    ----------
    n_windows:
        Number of windows the dataset yields.
    """

    def __init__(self, data: np.ndarray, window_size: int, stride: int = 1) -> None:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("data must be 2-D (n_samples, n_features).")
        if window_size < 2:
            raise ValueError("window_size must be at least 2.")
        if stride < 1:
            raise ValueError("stride must be >= 1.")
        if len(data) < window_size:
            raise ValueError(
                f"need at least window_size={window_size} samples, got {len(data)}."
            )
        self.data = data
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.n_windows = 1 + (len(data) - self.window_size) // self.stride

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        """Return ``(window,)`` — input and target are the same window."""
        if not 0 <= index < self.n_windows:
            raise IndexError(f"window index {index} out of range [0, {self.n_windows}).")
        start = index * self.stride
        window = self.data[start : start + self.window_size]
        return (torch.as_tensor(window, dtype=torch.float32),)

    def window_start(self, index: int) -> int:
        """Index of the first sample of window ``index`` in the raw array."""
        self._check_index(index)
        return index * self.stride

    def window_center(self, index: int) -> int:
        """Index of the *central* sample of window ``index``.

        Aligning a window's reconstruction error with its centre sample is
        the standard trick to turn per-window scores into a per-timestamp
        anomaly series that can be compared 1:1 with the raw signal and its
        fault labels.
        """
        self._check_index(index)
        return self.window_start(index) + self.window_size // 2

    def _check_index(self, index: int) -> None:
        if not 0 <= index < self.n_windows:
            raise IndexError(f"window index {index} out of range [0, {self.n_windows}).")
