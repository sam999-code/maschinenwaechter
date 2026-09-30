"""Lesson 9a: Train once, SAVE the model, reuse it forever.

Until now the model died when the script ended. A real product must:
  1. train once (expensive: minutes)
  2. SAVE everything needed to make predictions (cheap: milliseconds)
  3. load it later, on new data, without retraining

The saved "bundle" must include MORE than neural network weights:
the scaler (mean/std), the threshold, the window size. Without them
the raw weights are useless -- like a car engine without the car.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
torch.manual_seed(42)

def simulate_machine(seed, fault_hour, hours=24):
    fs = 1
    n = hours * 3600
    t = np.arange(n) / 3600
    rng = np.random.default_rng(seed)
    f_idx = int(fault_hour * 3600)
    vibration = rng.normal(0.0, 1.0, n) + 0.5 * np.sin(2 * np.pi * t)
    vibration[f_idx:] *= np.exp(0.15 * (t[f_idx:] - fault_hour))
    knocks = rng.integers(f_idx, n, size=40)
    for i, pos in enumerate(knocks):
        vibration[pos] += rng.uniform(3, 6) * (1 + i / len(knocks))
    temperature = 55 + 5 * np.sin(2 * np.pi * (t - 8) / 24)
    temperature += rng.normal(0.0, 0.1, n)
    temperature[f_idx:] += 2.0 * (t[f_idx:] - fault_hour)
    return pd.DataFrame({"hour": t, "vibration": vibration,
                         "temperature": temperature, "fault": (t >= fault_hour).astype(int)})

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
        h_dec = self.from_latent(self.to_latent(h[-1])).unsqueeze(0)
        inp = torch.zeros(x.shape[0], 1, x.shape[2])
        outs = []
        for t in range(x.shape[1]):
            step, (h_dec, _) = self.decoder(inp, (h_dec, torch.zeros_like(h_dec)))
            out_t = self.out(step)
            outs.append(out_t)
            inp = out_t
        return torch.cat(outs, dim=1)

# --- train on machine A (healthy part only) -------------------------------
df = simulate_machine(seed=42, fault_hour=16.0)
SENSORS = ["vibration", "temperature"]
WINDOW = 60
healthy = df[df["fault"] == 0]
mean = healthy[SENSORS].mean().to_numpy()
std = healthy[SENSORS].std().to_numpy()

def make_windows(data, window=WINDOW, stride=10):
    idx = np.arange(0, len(data) - window, stride)
    return np.stack([data[i : i + window] for i in idx])

train_w = make_windows(((healthy[SENSORS] - mean) / std).to_numpy())

model = LSTMAutoencoder()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
X = torch.tensor(train_w, dtype=torch.float32)
for epoch in range(6):
    model.train()
    for i in range(0, len(X), 256):
        batch = X[i : i + 256]
        loss = F.mse_loss(model(batch), batch)
        opt.zero_grad(); loss.backward(); opt.step()

# calibrate threshold on healthy rebuild errors (99th percentile)
model.eval()
with torch.no_grad():
    healthy_errors = np.concatenate([
        F.mse_loss(model(torch.tensor(train_w[i:i+1024], dtype=torch.float32)),
                   torch.tensor(train_w[i:i+1024], dtype=torch.float32),
                   reduction="none").mean(dim=(1, 2)).numpy()
        for i in range(0, len(train_w), 1024)
    ])
threshold = float(np.percentile(healthy_errors, 99))

# --- SAVE THE BUNDLE -------------------------------------------------------
models_dir = PROJECT_ROOT / "models"
models_dir.mkdir(exist_ok=True)
torch.save(model.state_dict(), models_dir / "machineguard_v1_weights.pt")
with open(models_dir / "machineguard_v1_config.json", "w") as f:
    json.dump({"sensors": SENSORS, "window": WINDOW, "stride": 10,
               "mean": mean.tolist(), "std": std.tolist(),
               "threshold": threshold}, f, indent=2)
print(f"Saved model bundle to {models_dir}")
print(f"threshold = {threshold:.3f}  (99th pct of healthy rebuild errors)")
