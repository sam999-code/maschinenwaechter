"""MachineGuard CLI - now a thin shell over the package.

Run from the lessons/ folder:
  ..\\.venv\\Scripts\\python.exe -m machineguard.cli --csv ../data/sample/machine_b.csv

(Also shows how to run a package module with -m: Python finds
machineguard/ because the current folder is on sys.path.)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from .pipeline import (make_windows, apply_scaler, score_windows,
                       sustained_alerts, load_bundle)


def analyze(csv_path, models_prefix, plot_path=None, run_length=5):
    df = pd.read_csv(csv_path)
    model, cfg = load_bundle(models_prefix)
    mean = np.array(cfg["mean"]); std = np.array(cfg["std"])
    scaled = apply_scaler(df, mean, std)
    w, stride = cfg["window"], cfg["stride"]
    windows = make_windows(scaled, w, stride)
    hours = df["hour"].to_numpy()[w // 2 :: stride][: len(windows)]
    errors = score_windows(model, windows)
    flagged = errors > cfg["threshold"]
    alarm = sustained_alerts(flagged, run_length=run_length)  # Lesson 9 exercise, built in

    print(f"\nMachineGuard report for: {csv_path}")
    print(f"  windows analyzed:      {len(windows)}")
    print(f"  raw flagged windows:   {flagged.sum()} ({flagged.mean():.1%})")
    print(f"  SUSTAINED alarms (run>={run_length}): {alarm.sum()} ({alarm.mean():.1%})")
    if alarm.any():
        print(f"  first sustained alert: hour {hours[np.argmax(alarm)]:.2f}")
    print(f"  health verdict: {'⚠ FAULT DETECTED' if alarm.mean() > 0.05 else '✓ looks healthy'}")

    if plot_path:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(hours, errors, lw=0.5, color="purple")
        ax.plot(hours[alarm], errors[alarm], "r.", ms=2, label="sustained alarm")
        ax.axhline(cfg["threshold"], color="red", ls="--", lw=0.8)
        ax.set_xlabel("hour"); ax.set_ylabel("rebuild error")
        ax.set_title(f"MachineGuard: {Path(csv_path).name}")
        ax.legend(); ax.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=110)
        print(f"  plot saved: {plot_path}")
    return errors, hours, alarm


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MachineGuard fault detection")
    p.add_argument("--csv", required=True)
    p.add_argument("--models", default="../models/machineguard_v1",
                   help="bundle path prefix (default: the v1 bundle)")
    p.add_argument("--plot", default=None)
    args = p.parse_args()
    analyze(args.csv, args.models, plot_path=args.plot)
