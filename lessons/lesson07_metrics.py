"""Lesson 7: Evaluating a detector honestly - precision, recall, F1.

"88.7% of fault time flagged" (Lesson 6) sounds great, but it's ONE
number that hides the full picture. Professionals answer three questions:

  PRECISION: Of all my alarms, how many were REAL?
             (a false alarm sends a technician to the factory at 3 AM)
  RECALL:    Of all the faulty time, how much did I catch?
             (a missed fault = the machine breaks down anyway)
  F1:        the balanced combination of both

We also learn the most practical skill in applied ML:
  MOVING THE THRESHOLD trades precision against recall.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
torch.manual_seed(42)

# --- 1. Load data, scale on healthy only (same as Lesson 6) --------------
df = pd.read_csv(PROJECT_ROOT / "data/sample/lesson05_machine.csv")
fault_hour = 16.0
healthy = df[df["fault"] == 0]
sensors = ["vibration", "temperature"]
mean = healthy[sensors].mean().to_numpy()
std = healthy[sensors].std().to_numpy()
all_scaled = ((df[sensors] - mean) / std).to_numpy()
healthy_scaled = ((healthy[sensors] - mean) / std).to_numpy()

def make_windows(data, window=60, stride=10):
    idx = np.arange(0, len(data) - window, stride)
    return np.stack([data[i : i + window] for i in idx])

WINDOW = 60
train_w = make_windows(healthy_scaled, WINDOW)
score_w = make_windows(all_scaled, WINDOW)
score_hours = df["hour"].to_numpy()[WINDOW // 2 :: 10][: len(score_w)]
is_fault_window = score_hours >= fault_hour  # TRUE labels per window

# --- 2. Train the same Autoencoder as Lesson 6 (condensed) ---------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_sensors=2, hidden=16, latent=8):
        super().__init__()
        self.encoder = nn.LSTM(n_sensors, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(n_sensors, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_sensors)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])
        h_dec = self.from_latent(z).unsqueeze(0)
        inp = torch.zeros(x.shape[0], 1, x.shape[2])
        outs = []
        for t in range(x.shape[1]):
            step, (h_dec, _) = self.decoder(inp, (h_dec, torch.zeros_like(h_dec)))
            out_t = self.out(step)
            outs.append(out_t)
            inp = out_t
        return torch.cat(outs, dim=1)

model = LSTMAutoencoder()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
X_train = torch.tensor(train_w, dtype=torch.float32)
for epoch in range(6):  # same recipe as Lesson 6
    model.train()
    for i in range(0, len(X_train), 256):
        batch = X_train[i : i + 256]
        loss = F.mse_loss(model(batch), batch)
        opt.zero_grad()
        loss.backward()
        opt.step()

model.eval()
with torch.no_grad():
    errors = np.concatenate([
        F.mse_loss(model(torch.tensor(score_w[i : i + 1024], dtype=torch.float32)),
                   torch.tensor(score_w[i : i + 1024], dtype=torch.float32),
                   reduction="none").mean(dim=(1, 2)).numpy()
        for i in range(0, len(score_w), 1024)
    ])

# --- 3. THE METRICS - computed by hand so you own the definitions --------
def evaluate(alarm, truth):
    """Return (precision, recall, f1) for boolean arrays alarm and truth."""
    tp = np.sum(alarm & truth)          # alarmed AND faulty      -> correct
    fp = np.sum(alarm & ~truth)         # alarmed BUT healthy     -> false alarm
    fn = np.sum(~alarm & truth)         # silent BUT faulty       -> miss
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

healthy_errors = errors[~is_fault_window]

# --- 4. The threshold sweep: one detector, many operating points ----------
print("threshold (healthy pct) | precision | recall |   F1   | false alarms")
rows = []
for pct in [95, 97, 98, 99, 99.5, 99.9]:
    thr = np.percentile(healthy_errors, pct)
    alarm = errors > thr
    p, r, f1 = evaluate(alarm, is_fault_window)
    n_false = np.sum(alarm & ~is_fault_window)
    rows.append((pct, p, r, f1))
    print(f"          {pct:4.1f}          |   {p:.3f}   | {r:.3f}  | {f1:.3f} |     {n_false}")

# --- 5. Plot the tradeoff -------------------------------------------------
pcts = [r[0] for r in rows]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(pcts, [r[1] for r in rows], "o-", label="precision")
ax.plot(pcts, [r[2] for r in rows], "s-", label="recall")
ax.plot(pcts, [r[3] for r in rows], "^--", label="F1")
ax.set_xlabel("threshold = percentile of HEALTHY errors")
ax.set_ylabel("score")
ax.set_title("Moving the threshold trades precision against recall")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports/figures/lesson07_metrics.png", dpi=110)
print("\nSaved plot to reports/figures/lesson07_metrics.png")
