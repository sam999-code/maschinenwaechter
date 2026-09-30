"""Lesson 9b: MachineGuard CLI - analyze ANY machine CSV file.

Usage:
  .venv\\Scripts\\python.exe lessons\\machineguard_cli.py --csv data/sample/some_machine.csv

This is the shape of every ML product: a trained model behind a
simple interface. The factory worker doesn't see LSTMs -- they run
one command and get an answer: "healthy" or "fault detected".
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

def load_bundle(models_dir):
    """Load weights + config and rebuild the full predictor."""
    with open(models_dir / "machineguard_v1_config.json") as f:
        cfg = json.load(f)
    model = LSTMAutoencoder(n_sensors=len(cfg["sensors"]))
    model.load_state_dict(torch.load(models_dir / "machineguard_v1_weights.pt",
                                     weights_only=True))
    model.eval()
    return model, cfg

def analyze(csv_path, model, cfg, plot_path=None):
    """Score one CSV file. Returns (per-window errors, hours, alarm flags)."""
    df = pd.read_csv(csv_path)
    missing = [c for c in cfg["sensors"] if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing columns: {missing}")
    mean = np.array(cfg["mean"]); std = np.array(cfg["std"])
    scaled = ((df[cfg["sensors"]] - mean) / std).to_numpy()
    w, stride = cfg["window"], cfg["stride"]
    idx = np.arange(0, len(scaled) - w, stride)
    windows = np.stack([scaled[i : i + w] for i in idx])
    hours = df["hour"].to_numpy()[w // 2 :: stride][: len(windows)]
    with torch.no_grad():
        errors = np.concatenate([
            F.mse_loss(model(torch.tensor(windows[i : i + 1024], dtype=torch.float32)),
                       torch.tensor(windows[i : i + 1024], dtype=torch.float32),
                       reduction="none").mean(dim=(1, 2)).numpy()
            for i in range(0, len(windows), 1024)
        ])
    alarm = errors > cfg["threshold"]

    n = len(errors)
    print(f"\nMachineGuard report for: {csv_path}")
    print(f"  windows analyzed: {n}")
    print(f"  windows flagged:  {alarm.sum()} ({alarm.mean():.1%})")
    print(f"  first alert at hour: {hours[np.argmax(alarm)] if alarm.any() else 'never'}")
    print(f"  health verdict: {'⚠ FAULT DETECTED' if alarm.mean() > 0.1 else '✓ looks healthy'}")

    if plot_path:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(hours, errors, lw=0.5, color="purple")
        ax.plot(hours[alarm], errors[alarm], "r.", ms=2)
        ax.axhline(cfg["threshold"], color="red", ls="--", lw=0.8)
        ax.set_xlabel("hour"); ax.set_ylabel("rebuild error")
        ax.set_title(f"MachineGuard: {Path(csv_path).name}")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=110)
        print(f"  plot saved: {plot_path}")
    return errors, hours, alarm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MachineGuard: detect machine faults in a sensor CSV")
    parser.add_argument("--csv", required=True, help="CSV with 'hour', 'vibration', 'temperature' columns")
    parser.add_argument("--plot", default=None, help="optional path to save the error plot")
    args = parser.parse_args()

    model, cfg = load_bundle(PROJECT_ROOT / "models")
    analyze(args.csv, model, cfg, plot_path=args.plot)
