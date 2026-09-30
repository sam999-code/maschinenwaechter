"""Lesson 13: Real data - NASA CMAPSS turbofan engines.

This is the industry-standard predictive-maintenance benchmark:
100 aircraft engines in the TRAIN set, each run TO FAILURE with
21 sensors + 3 settings recorded every flight cycle, RUL labels
(Remaining Useful Life = cycles until failure).

Everything we did with the simulator, we now do with reality:
  1. explore: watch real sensors drift as failure approaches
  2. feature selection: let the DATA tell us which sensors matter
  3. train the LSTM Autoencoder on HEALTHY windows (RUL >= 125,
     the standard "piecewise" assumption from the literature)
  4. prove: reconstruction error must RISE as RUL FALLS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
torch.manual_seed(42)

# --- 1. Load --------------------------------------------------------------
train = pd.read_parquet(PROJECT_ROOT / "data/raw/fd001_train.parquet")
test = pd.read_parquet(PROJECT_ROOT / "data/raw/fd001_test.parquet")
print(f"train: {train.shape} | test: {test.shape}")
print(f"engines: {train.unit_id.nunique()} train / {test.unit_id.nunique()} test")

# --- 2. Feature selection: keep sensors that actually VARY ----------------
# In FD001 (one operating condition) several sensors are constant -
# a constant column carries zero information. Real projects do this
# check FIRST, and here the data itself chooses ~14 informative sensors.
ID_COLS = ["unit_id", "cycle", "RUL", "dataset", "split"]
variances = train.drop(columns=ID_COLS).var()
SENSORS = variances[variances > 1e-6].index.tolist()
print(f"\ninformative sensors kept: {len(SENSORS)} of {len(variances)}")

# --- 3. Explore: one engine's degradation journey -------------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, col in zip(axes.ravel(), ["T24", "T30", "P30", "Nc", "NRc", "W31"]):
    for uid in [1, 3, 5]:
        g = train[train.unit_id == uid]
        ax.plot(g["RUL"], g[col], lw=0.8, label=f"engine {uid}")
    ax.invert_xaxis()  # time flows left->right (RUL decreases)
    ax.set_title(col); ax.set_xlabel("RUL (cycles)")
axes.ravel()[0].legend()
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports/figures/lesson13_exploration.png", dpi=110)
print("saved exploration plot (watch the curves drift as RUL -> 0)")

# --- 4. Scale using TRAIN-HEALTHY rows only (min-max, no leakage) ---------
HEALTHY_RUL = 125  # industry standard: beyond this, engines count as healthy
healthy_train = train[train.RUL >= HEALTHY_RUL]
lo = healthy_train[SENSORS].min()
hi = healthy_train[SENSORS].max()
rng = (hi - lo).replace(0, 1)

def scale(df):
    return ((df[SENSORS] - lo) / rng).clip(0, 1).to_numpy()

# --- 5. Windows per engine, train on healthy windows only -----------------
WINDOW = 30
def engine_windows(df):
    """(n_windows, WINDOW, n_sensors) + the RUL at each window center."""
    out_x, out_rul = [], []
    for _, g in df.groupby("unit_id"):
        g = g.sort_values("cycle")
        x = scale(g)
        for i in range(0, len(x) - WINDOW):
            out_x.append(x[i : i + WINDOW])
            out_rul.append(g["RUL"].iloc[i + WINDOW // 2])
    return np.stack(out_x), np.array(out_rul)

X_train_all, rul_train = engine_windows(train)
healthy_mask = rul_train >= HEALTHY_RUL
X_healthy = X_train_all[healthy_mask]
print(f"\nwindows: {len(X_train_all):,} total | {len(X_healthy):,} healthy (training set)")

# --- 6. The same LSTM Autoencoder as always --------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_sensors, hidden=16, latent=8):
        super().__init__()
        self.encoder = nn.LSTM(n_sensors, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(n_sensors, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_sensors)
    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h_dec = self.from_latent(self.to_latent(h[-1])).unsqueeze(0)
        inp = torch.zeros(x.shape[0], 1, x.shape[2]); outs = []
        for t in range(x.shape[1]):
            step, (h_dec, _) = self.decoder(inp, (h_dec, torch.zeros_like(h_dec)))
            out_t = self.out(step); outs.append(out_t); inp = out_t
        return torch.cat(outs, dim=1)

model = LSTMAutoencoder(n_sensors=len(SENSORS))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
X = torch.tensor(X_healthy, dtype=torch.float32)
for epoch in range(4):
    total = 0.0
    for i in range(0, len(X), 256):
        b = X[i : i + 256]
        loss = F.mse_loss(model(b), b)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(b)
    print(f"epoch {epoch}: healthy rebuild MSE = {total/len(X):.4f}")
model.eval()

def score(X):
    with torch.no_grad():
        return np.concatenate([F.mse_loss(
            model(torch.tensor(X[i:i+1024], dtype=torch.float32)),
            torch.tensor(X[i:i+1024], dtype=torch.float32),
            reduction="none").mean(dim=(1, 2)).numpy() for i in range(0, len(X), 1024)])

# --- 7. THE PROOF: error must rise as RUL falls ----------------------------
X_test, rul_test = engine_windows(test)
errors_test = score(X_test)
rho, pval = spearmanr(rul_test, errors_test)
print(f"\nSpearman correlation RUL vs rebuild error: rho = {rho:.3f} (p = {pval:.1e})")
print("(strong NEGATIVE value = error climbs as failure approaches -> detector works)")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(rul_test, errors_test, s=2, alpha=0.2)
ax.set_xlabel("true RUL (cycles)"); ax.set_ylabel("rebuild error")
ax.set_title(f"CMAPSS FD001 test: rho = {rho:.3f}")
ax.grid(True); plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports/figures/lesson13_proof.png", dpi=110)
print("saved proof plot")
