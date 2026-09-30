"""Lesson 8: Generalization - train on one machine, catch faults on ANOTHER.

Everything so far had a hidden cheat: we trained and tested on the
SAME machine. The model could partly memorize that machine's exact
noise pattern. Real factories have MANY machines, and each one is
slightly different (age, mounting, load, sensor placement).

The honest question: does a detector trained on machine A also work
on machine B? That property is called GENERALIZATION, and it is the
difference between a demo and a product.
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

# --- 1. The simulator as a FUNCTION (same physics as Lesson 5) -----------
# This time with `seed` and `fault_hour` as parameters, so we can build
# as many different machines as we want.
def simulate_machine(seed, fault_hour, hours=24):
    fs = 1
    n = hours * 3600
    t = np.arange(n) / 3600
    rng = np.random.default_rng(seed)
    f_idx = int(fault_hour * 3600)

    vibration = rng.normal(0.0, 1.0, n)
    vibration += 0.5 * np.sin(2 * np.pi * t)
    vibration[f_idx:] *= np.exp(0.15 * (t[f_idx:] - fault_hour))
    knocks = rng.integers(f_idx, n, size=40)
    for i, pos in enumerate(knocks):
        vibration[pos] += rng.uniform(3, 6) * (1 + i / len(knocks))

    temperature = 55 + 5 * np.sin(2 * np.pi * (t - 8) / 24)
    temperature += rng.normal(0.0, 0.1, n)
    temperature[f_idx:] += 2.0 * (t[f_idx:] - fault_hour)

    return pd.DataFrame({
        "hour": t, "vibration": vibration,
        "temperature": temperature, "fault": (t >= fault_hour).astype(int),
    })

# Machine A: our training/calibration machine (seed 42, fault at hour 16)
# Machine B: a DIFFERENT machine (seed 7, fault at hour 12!)
df_train = simulate_machine(seed=42, fault_hour=16.0)
df_test = simulate_machine(seed=7, fault_hour=12.0)
print(f"train machine: fault at 16h | test machine: fault at 12h (seed 7)")

# --- 2. Scale using the TRAIN machine's healthy period only --------------
sensors = ["vibration", "temperature"]
healthy_train = df_train[df_train["fault"] == 0]
mean = healthy_train[sensors].mean().to_numpy()
std = healthy_train[sensors].std().to_numpy()

def scaled(df):
    return ((df[sensors] - mean) / std).to_numpy()

def make_windows(data, window=60, stride=10):
    idx = np.arange(0, len(data) - window, stride)
    return np.stack([data[i : i + window] for i in idx])

WINDOW = 60
train_w = make_windows(scaled(healthy_train), WINDOW)
test_w = make_windows(scaled(df_test), WINDOW)
test_hours = df_test["hour"].to_numpy()[WINDOW // 2 :: 10][: len(test_w)]
test_is_fault = test_hours >= 12.0
# also score the train machine (same-machine test = the "easy" exam)
train_all_w = make_windows(scaled(df_train), WINDOW)
train_hours = df_train["hour"].to_numpy()[WINDOW // 2 :: 10][: len(train_all_w)]
train_is_fault = train_hours >= 16.0

# --- 3. Same Autoencoder, trained ONLY on machine A's healthy data -------
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
for epoch in range(6):
    model.train()
    for i in range(0, len(X_train), 256):
        batch = X_train[i : i + 256]
        loss = F.mse_loss(model(batch), batch)
        opt.zero_grad()
        loss.backward()
        opt.step()

model.eval()
def score(windows):
    with torch.no_grad():
        return np.concatenate([
            F.mse_loss(model(torch.tensor(windows[i : i + 1024], dtype=torch.float32)),
                       torch.tensor(windows[i : i + 1024], dtype=torch.float32),
                       reduction="none").mean(dim=(1, 2)).numpy()
            for i in range(0, len(windows), 1024)
        ])

errors_train = score(train_all_w)
errors_test = score(test_w)

# --- 4. THE HONEST COMPARISON --------------------------------------------
def evaluate(alarm, truth):
    tp = np.sum(alarm & truth); fp = np.sum(alarm & ~truth); fn = np.sum(~alarm & truth)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

threshold = np.percentile(errors_train[~train_is_fault], 99)  # calibrated on A-healthy

p_tr, r_tr, f_tr = evaluate(errors_train > threshold, train_is_fault)
p_te, r_te, f_te = evaluate(errors_test > threshold, test_is_fault)

print("\n                    precision  recall    F1")
print(f"  same machine (A):   {p_tr:.3f}     {r_tr:.3f}   {f_tr:.3f}")
print(f"  NEW machine (B):    {p_te:.3f}     {r_te:.3f}   {f_te:.3f}")

# --- 5. Plot both error curves -------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train_hours, errors_train, lw=0.5, label="machine A (train)", alpha=0.7)
ax.plot(test_hours, errors_test, lw=0.5, label="machine B (NEW)", color="darkorange")
ax.axhline(threshold, color="red", ls="--", lw=1, label="threshold (from A)")
ax.axvspan(16, 24, color="gray", alpha=0.1)
ax.axvspan(12, 24, color="red", alpha=0.06)
ax.set_xlabel("hour"); ax.set_ylabel("rebuild error")
ax.set_title("Does a model trained on A catch faults on B?")
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports/figures/lesson08_generalization.png", dpi=110)
print("\nSaved plot to reports/figures/lesson08_generalization.png")
