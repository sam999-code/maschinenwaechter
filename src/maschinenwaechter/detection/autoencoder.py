"""LSTM-Autoencoder for deep anomaly detection (Phase 4).

Why reconstruction detects anomalies
------------------------------------
An autoencoder is trained to *compress and reconstruct* its input. We train
it ONLY on healthy machine windows, so it learns the "normal" temporal
patterns of vibration, temperature and acoustic signals. When a window
contains a fault (impulsive bearing shocks, rising vibration, heating), the
network has never seen that pattern and reconstructs it poorly — the
reconstruction error becomes the anomaly score. High error ≈ the machine
looks unlike anything healthy the model knows.

Why an LSTM and not a plain feed-forward autoencoder?
-----------------------------------------------------
The rolling z-score baseline (Phase 3) treats every sample independently:
it compares ``x(t)`` against the local mean/std and knows nothing about the
sequence around it. Machine signals are deeply *temporal* — a healthy
vibration burst (e.g. a load change) looks locally similar to a shock, but
its shape over the surrounding seconds is completely different. The LSTM
encoder reads the window step by step and its hidden state accumulates
context, so the latent vector encodes *how the signal evolves*, not just
*how large it is*. This lets the model tolerate large-but-healthy
transients (rpm load swings, daily temperature cycle) while still flagging
fault signatures whose temporal shape is wrong.

Architecture (per window of T timesteps, F=3 features)
-------------------------------------------------------
1. Encoder LSTM (input F -> hidden 32, 1 layer) consumes the window; the
   final hidden state is the latent vector (length 32) — a 30x3 = 90-dim
   window compressed to 32 numbers, ~3x compression. Stronger compression
   forces the model to keep only the structure shared by healthy windows.
2. Decoder LSTM (input F -> hidden 32) is fed a repeated copy of the
   latent vector (reshaped to the encoder's last hidden state) and
   reconstructs the window step by step.
3. Output projection (32 -> 3) maps decoder states back to sensor values.

Scoring
-------
Per-window error = mean squared reconstruction error across all timesteps
and features. Per-timestep error (MSE across features at each step) is also
exposed so individual timestamps can be annotated.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

DEFAULT_SENSOR_COLUMNS = ("vibration_mm_s", "temperature_c", "acoustic_db")


class LSTMAutoencoder(nn.Module):
    """Reconstruct sensor windows; large errors indicate anomalies.

    Parameters
    ----------
    input_size:
        Number of sensor features per timestep (3: vibration, temperature,
        acoustic). ``rpm`` is excluded on purpose — it swings with
        production load, not machine health (same reasoning as the
        Phase 3 baseline).
    hidden_size:
        LSTM hidden dimension (= latent vector length).
    num_layers:
        Stacked LSTM layers for encoder and decoder.
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 32,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        self.encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        # Decoder consumes one feature vector per step; each step is
        # initialised from the latent vector produced by the encoder.
        self.decoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.output_proj = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of windows.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, T, input_size)``.

        Returns
        -------
        torch.Tensor
            Reconstruction of the same shape ``(batch, T, input_size)``.
        """
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError(
                f"expected input (batch, T, {self.input_size}), got {tuple(x.shape)}"
            )
        batch, window, _ = x.shape

        # Encoder: final hidden/cell states summarise the whole window.
        _, (hidden, cell) = self.encoder(x)

        # Decoder: start from the latent state, feed zeros as input — the
        # decoder must "replay" the window purely from its compressed memory.
        decoder_input = torch.zeros(
            batch, window, self.input_size, dtype=x.dtype, device=x.device
        )
        decoded, _ = self.decoder(decoder_input, (hidden.contiguous(), cell.contiguous()))
        return self.output_proj(decoded)

    # ------------------------------------------------------------------
    # Scoring helpers (numpy in / numpy out; torch used internally only)
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_scores(window_scores: np.ndarray, per_timestep: np.ndarray) -> None:
        if window_scores.ndim != 1:
            raise ValueError("window_scores must be 1-D (one value per window).")
        if per_timestep.ndim != 2:
            raise ValueError("per_timestep must be 2-D (windows x timesteps).")

    def reconstruction_errors(
        self,
        windows: np.ndarray,
        batch_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score a stack of windows.

        Parameters
        ----------
        windows:
            Array ``(n_windows, T, input_size)``.
        batch_size:
            Windows are scored in mini-batches to bound memory.

        Returns
        -------
        window_errors : np.ndarray
            MSE across all timesteps/features per window, shape
            ``(n_windows,)``. This is the primary anomaly score.
        per_timestep_errors : np.ndarray
            MSE across features per timestep, shape ``(n_windows, T)``,
            so individual timestamps can be annotated.
        """
        was_training = self.training
        self.eval()
        window_errors, per_step = [], []
        with torch.no_grad():
            for start in range(0, len(windows), batch_size):
                chunk = np.ascontiguousarray(windows[start : start + batch_size])
                batch = torch.from_numpy(chunk)
                recon = self.forward(batch)
                sq_err = (recon - batch) ** 2
                per_step.append(sq_err.mean(dim=-1).numpy())            # (B, T)
                window_errors.append(sq_err.mean(dim=(1, 2)).numpy())   # (B,)
        if was_training:
            self.train()
        window_scores = np.concatenate(window_errors)
        per_timestep = np.concatenate(per_step)
        self._validate_scores(window_scores, per_timestep)
        return window_scores, per_timestep
