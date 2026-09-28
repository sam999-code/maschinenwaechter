"""RUL regression model (Phase 5): an LSTM that reads a sensor window and
outputs ONE number — the predicted remaining hours until failure.

Regression, not classification — why?
-------------------------------------
* The product question is quantitative ("~40 h left"), and the business
  decision (schedule the repair within the next shift? next week?) is a
  threshold ON a continuous estimate. A regressor gives the plant manager
  the number; the operations team picks the threshold that matches their
  spare-parts and staffing reality.
* Classification would need fixed bins ("<12 h", "12–48 h", ...) chosen in
  advance; every bin boundary is a hidden business decision baked into the
  model. Bins also throw away information: "41 h" and "47 h" are the same
  class but different maintenance plans.
* RUL is a *countdown*: the natural geometry of the problem is a decreasing
  curve, exactly what a scalar output can represent.

Why an LSTM (same argument as Phase 4)?
----------------------------------------
Bearing wear is a *temporal* process: exponential vibration growth,
impulsive shocks ramping in, linear friction heating. A window summarised
by the last LSTM hidden state encodes how the three channels EVOLVED over
the last 30 seconds, which is what correlates with time-to-failure — far
more than any single sample's magnitude.

Output scaling note
-------------------
The network predicts raw hours and is trained on standardised INPUT features
(scaler fitted on the training windows, stored on the model as
``scaler_mean_`` / ``scaler_std_`` exactly like Phase 4). Predicting in
hours directly keeps the output interpretable; Huber loss (see
``rul/train.py``) handles the scale robustly.

Huber loss — why it is the standard for RUL
--------------------------------------------
RUL labels mix a long flat plateau (the cap) with a steep descent to 0.
A plain MSE is dominated by the worst residuals — early windows where the
model reasonably predicts "far away" but the capped label structure (and
any labelling noise on real run-to-failure data) produces large errors.
Huber loss (``nn.HuberLoss``) behaves like MSE for small residuals and like
absolute (L1) loss for large ones, so single bad windows cannot drag the
whole training trajectory around. That outlier robustness is why
Huber/quantile losses are the de-facto standard in the RUL literature.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class RULRegressor(nn.Module):
    """LSTM regressor: window (batch, T, 3) -> predicted RUL in hours (batch,).

    Parameters
    ----------
    input_size:
        Sensor features per timestep (3: vibration, temperature, acoustic).
    hidden_size:
        LSTM hidden dimension; the LAST hidden state summarises the window.
    """

    def __init__(self, input_size: int = 3, hidden_size: int = 32) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        # Fitted by train_rul, applied by predict_rul (same pattern as Phase 4).
        self.scaler_mean_: np.ndarray | None = None
        self.scaler_std_: np.ndarray | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict RUL for a batch of windows.

        Parameters
        ----------
        x:
            Tensor ``(batch, T, input_size)`` of already-standardised sensor
            windows.

        Returns
        -------
        torch.Tensor
            ``(batch,)`` predicted remaining hours until failure (>= 0 after
            the optional ``relu`` clip applied by callers that want it; the
            raw output is kept unclipped so gradients flow freely below 0
            during training).
        """
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError(
                f"expected input (batch, T, {self.input_size}), got {tuple(x.shape)}"
            )
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]                       # (batch, hidden)
        return self.head(last_hidden).squeeze(-1)      # (batch,)


def predict_rul(model: RULRegressor, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Predict RUL (hours) for a stack of windows — numpy in, numpy out.

    Parameters
    ----------
    model:
        A trained :class:`RULRegressor` carrying its input scaler
        (``scaler_mean_`` / ``scaler_std_``).
    X:
        Array ``(n_windows, T, 3)`` of RAW sensor windows; standardisation
        is applied here, exactly as during training.
    batch_size:
        Mini-batch size to bound memory.

    Returns
    -------
    np.ndarray
        ``(n_windows,)`` predicted RUL in hours, clipped at 0 (a negative
        remaining lifetime is not physically meaningful).
    """
    if model.scaler_mean_ is None or model.scaler_std_ is None:
        raise RuntimeError("model has no scaler; train it via maschinenwaechter.rul.train.train_rul first.")
    X = np.asarray(X, dtype=np.float32)
    scaled = ((X - model.scaler_mean_) / model.scaler_std_).astype(np.float32)

    was_training = model.training
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(scaled), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(scaled[start : start + batch_size]))
            preds.append(model(batch).numpy())
    if was_training:
        model.train()
    out = np.clip(np.concatenate(preds), 0.0, None)
    return out
