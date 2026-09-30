"""Lesson 14: Predicting Remaining Useful Life on real NASA data.

Lesson 13 asked "is this engine getting sick?" (anomaly score).
This lesson asks the expensive question: "HOW LONG until failure?"

We predict RUL (cycles remaining) directly - a regression task:
  input : window of 30 cycles x informative sensors
  output: one number, RUL at the window's center

Industry conventions we follow (so our numbers are comparable
to published papers):
  - RUL capping at 125 cycles (piecewise linear model)
  - Huber loss (robust to the extreme values near failure)
  - MAE/RMSE in cycles as the metrics (published ballpark ~15-30)
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

# --- 1. Load + feature selection + scaling (same recipe as Lesson 13) ----
train = pd.read_parquet(PROJECT_ROOT / "data/raw/fd001_train.parquet")
test = pd.read_parquet(PROJECT_ROOT / "data/raw/fd001_test.parquet")
ID_COLS = ["unit_id", "cycle", "RUL", "dataset", "split"]
SENSORS = train.drop(columns=ID_COLS).var()
SENSORS = SENSORS[SENSORS > 1e-6].index.tolist()

healthy = train[train.RUL >= 125]
lo, hi = healthy[SENSORS].min(), healthy[SENSORS].max()
rng = (hi - lo).replace(0, 1)
def scale(df): return ((df[SENSORS] - lo) / rng).clip(0, 1).to_numpy()

WINDOW, CAP = 30, 125
def engine_windows(df):
    X, y = [], []
    for _, g in df.groupby("unit_id"):
        g = g.sort_values("cycle"); x = scale(g)
        for i in range(0, len(x) - WINDOW):
            X.append(x[i : i + WINDOW])
            y.append(min(g["RUL"].iloc[i + WINDOW // 2], CAP))  # RUL capping
    return np.stack(X), np.array(y, dtype=np.float32)

X_tr, y_tr = engine_windows(train)
X_te, y_te = engine_windows(test)
print(f"train windows {X_tr.shape} | test windows {X_te.shape}")

# --- 2. The regressor: LSTM reads the window, MLP head outputs RUL -------
class RULRegressor(nn.Module):
    def __init__(self, n_sensors, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_sensors, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)  # one RUL per window

model = RULRegressor(n_sensors=len(SENSORS))
opt = torch.optim.Adam(model.parameters(), lr=3e-3)  # higher lr: RUL targets
                                                       # are 0-125 scale; the
                                                       # loss landscape needs it
loss_fn = nn.HuberLoss(delta=5.0)  # robust near-failure outliers
X = torch.tensor(X_tr, dtype=torch.float32); y = torch.tensor(y_tr)
for epoch in range(6):
    model.train(); total = 0.0
    for i in range(0, len(X), 256):
        bX, by = X[i:i+256], y[i:i+256]
        loss = loss_fn(model(bX), by)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(bX)
    print(f"epoch {epoch}: Huber loss = {total/len(X):.3f}")
model.eval()

# --- 3. Evaluate on the 100 held-out test engines --------------------------
with torch.no_grad():
    preds = np.concatenate([
        model(torch.tensor(X_te[i:i+1024], dtype=torch.float32)).numpy()
        for i in range(0, len(X_te), 1024)])

mae = np.mean(np.abs(preds - y_te))
rmse = float(np.sqrt(np.mean((preds - y_te) ** 2)))
critical = y_te < 50  # the zone where accuracy matters most
mae_crit = np.mean(np.abs(preds[critical] - y_te[critical]))
bias = float(np.mean(preds - y_te))
print(f"\nTEST RESULTS (100 unseen engines):")
print(f"  MAE  overall:        {mae:.1f} cycles")
print(f"  RMSE overall:        {rmse:.1f} cycles")
print(f"  MAE  in critical zone (RUL<50): {mae_crit:.1f} cycles")
print(f"  bias (pred - true):  {bias:+.1f} cycles  "
      f"({'pessimistic=safe' if bias < 0 else 'optimistic=DANGEROUS'})")

# --- THE BASELINE CHECK: a model must BEAT the dumbest possible rival ----
for c in [np.median(y_te), np.mean(y_te), float(CAP)]:
    print(f"  constant predictor RUL={c:.0f}: MAE = {np.mean(np.abs(y_te - c)):.1f}")
print("  (if our MAE is not clearly better, the model learned NOTHING)")

# --- 4. Plots ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(y_te, preds, s=2, alpha=0.15)
lim = CAP
axes[0].plot([0, lim], [0, lim], "r--", lw=1, label="perfect prediction")
axes[0].set_xlabel("true RUL"); axes[0].set_ylabel("predicted RUL")
axes[0].set_title(f"predicted vs true (MAE {mae:.1f} cycles)")
axes[0].legend(); axes[0].grid(True)

for uid in [11, 35, 72]:
    g = test[test.unit_id == uid].sort_values("cycle")
    x = scale(g)
    W = np.stack([x[i:i+WINDOW] for i in range(0, len(x)-WINDOW)])
    with torch.no_grad():
        p = model(torch.tensor(W, dtype=torch.float32)).numpy()
    axes[1].plot(g["cycle"].to_numpy()[WINDOW//2:][:len(p)], p,
                 lw=1, label=f"engine {uid} (pred)")
    axes[1].plot(g["cycle"], g["RUL"], "k:", lw=0.8, alpha=0.4)
axes[1].set_xlabel("cycle"); axes[1].set_ylabel("RUL (cycles)")
axes[1].set_title("engine trajectories: prediction (colored) vs truth (dotted)")
axes[1].legend(); axes[1].grid(True)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports/figures/lesson14_rul.png", dpi=110)
print("saved reports/figures/lesson14_rul.png")
