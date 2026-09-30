"""Lesson 6: Anomaly detection with an LSTM Autoencoder (PyTorch).

The problem we ended Lesson 5 with:
  - z-score catches sudden KNOCKS but misses slow WEAR growth
  - it watches ONE sensor at a time

The Autoencoder idea (this lesson):
  1. Show the network thousands of windows of HEALTHY machine data
  2. Its job: compress each window to a tiny vector, then rebuild it
     (like writing a 3-line summary of a page, then expanding the
     summary back to the full page)
  3. It becomes VERY good at rebuilding healthy patterns
  4. A sick window (growth, knocks, fever) rebuilds POORLY
  5. The rebuild error IS our alarm signal - for ALL sensors at once
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F  # F.mse_loss(..., reduction="none") lets us
                                  # keep per-window errors (used in scoring)

from detector import rolling_zscore  # our Lesson 3/4 classic, for comparison

PROJECT_ROOT = Path(__file__).resolve().parent.parent
torch.manual_seed(42)

# --- 1. Load the Lesson 5 data -------------------------------------------
df = pd.read_csv(PROJECT_ROOT / "data/sample/lesson05_machine.csv")
fault_hour = 16.0
healthy = df[df["fault"] == 0]          # hours 0-16: TRAIN on this only
sensors = ["vibration", "temperature"]

# --- 2. Scaling: mean/std computed from HEALTHY data only ----------------
# (using the fault period too would leak information about the fault
#  into training - a classic ML mistake called "data leakage")
mean = healthy[sensors].mean().to_numpy()
std = healthy[sensors].std().to_numpy()
all_scaled = ((df[sensors] - mean) / std).to_numpy()      # (86400, 2)
healthy_scaled = ((healthy[sensors] - mean) / std).to_numpy()

# --- 3. Make windows: each window = 60 seconds x 2 sensors ---------------
def make_windows(data, window=60, stride=10):
    """Cut a long (n_points, n_sensors) array into overlapping windows."""
    idx = np.arange(0, len(data) - window, stride)
    return np.stack([data[i : i + window] for i in idx])  # (n_windows, 60, 2)

WINDOW = 60
train_w = make_windows(healthy_scaled, WINDOW)            # healthy only!
score_w = make_windows(all_scaled, WINDOW)
score_hours = df["hour"].to_numpy()[WINDOW // 2 :: 10][: len(score_w)]
print(f"train windows (healthy): {len(train_w):,}")
print(f"score windows (all 24h): {len(score_w):,}")

# --- 4. The model: LSTM compresses 60x2 -> 8 numbers -> back to 60x2 ------
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_sensors=2, hidden=16, latent=8):
        super().__init__()
        self.encoder = nn.LSTM(n_sensors, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)     # squeeze to 8 numbers
        self.from_latent = nn.Linear(latent, hidden)   # expand back
        self.decoder = nn.LSTM(n_sensors, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_sensors)

    def forward(self, x):
        _, (h, _) = self.encoder(x)              # h = summary of the window
        z = self.to_latent(h[-1])                # the 8-number summary
        h_dec = self.from_latent(z).unsqueeze(0) # back to LSTM shape
        # decoder rebuilds the window step by step, feeding back its OWN
        # previous output as the next input (autoregressive decoding).
        # (Bug we hit live: feeding back zeros made every step identical,
        #  so the model could only learn the average -> loss stuck at 1.0)
        inp = torch.zeros(x.shape[0], 1, x.shape[2])  # start token: silence
        outs = []
        for t in range(x.shape[1]):
            step, (h_dec, _) = self.decoder(inp, (h_dec, torch.zeros_like(h_dec)))
            out_t = self.out(step)
            outs.append(out_t)
            inp = out_t  # feed the prediction back in
        return torch.cat(outs, dim=1)

model = LSTMAutoencoder()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# --- 5. Train: rebuild healthy windows as well as possible ----------------
X_train = torch.tensor(train_w, dtype=torch.float32)
for epoch in range(1, 7):
    model.train()
    total = 0.0
    for i in range(0, len(X_train), 256):          # mini-batches of 256
        batch = X_train[i : i + 256]
        loss = loss_fn(model(batch), batch)        # input == target!
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(batch)
    print(f"epoch {epoch}: rebuild error (MSE) = {total / len(X_train):.4f}")

# --- 6. Score every window: error = how badly the rebuild went ------------
model.eval()
with torch.no_grad():
    errors = np.concatenate([
        F.mse_loss(model(torch.tensor(score_w[i : i + 1024], dtype=torch.float32)),
                   torch.tensor(score_w[i : i + 1024], dtype=torch.float32),
                   reduction="none").mean(dim=(1, 2)).numpy()
        for i in range(0, len(score_w), 1024)
    ])

# alarm threshold: 99th percentile of errors on HEALTHY windows only
healthy_mask = score_hours < fault_hour
threshold = np.percentile(errors[healthy_mask], 99)
alarm = errors > threshold
print(f"\nAutoencoder alarms on healthy hours: {alarm[healthy_mask].sum()}")
print(f"Autoencoder alarms on fault hours:   {alarm[~healthy_mask].sum()}")

# --- 7. Compare with the z-score on the same data -------------------------
_, z_alert = rolling_zscore(df["vibration"].to_numpy(), window=3600, threshold=3.5)
z_fault_rate = z_alert[int(fault_hour * 3600) :].mean()
ae_fault_rate = alarm[~healthy_mask].mean()
print(f"\nFRACTION OF FAULT TIME FLAGGED:")
print(f"  z-score (Lesson 3):      {z_fault_rate:.1%}")
print(f"  autoencoder (Lesson 6):  {ae_fault_rate:.1%}")

# --- 8. Plot ---------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(df["hour"], df["vibration"], lw=0.3)
axes[0].set_ylabel("vibration")
axes[1].plot(score_hours, errors, lw=0.6, color="purple", label="rebuild error")
axes[1].axhline(threshold, color="red", ls="--", lw=0.8, label="threshold (healthy 99th pct)")
axes[1].plot(score_hours[alarm], errors[alarm], "r.", ms=2, label="ALERT")
for ax in axes:
    ax.axvspan(fault_hour, 24, color="red", alpha=0.08)
axes[1].set_xlabel("hour of day")
axes[1].legend()
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports/figures/lesson06_autoencoder.png", dpi=110)
print("\nSaved plot to reports/figures/lesson06_autoencoder.png")
