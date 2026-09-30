"""Utility: generate a machine CSV with any seed and fault hour.

  .venv\\Scripts\\python.exe lessons\\make_machine_csv.py --seed 7 --fault-hour 12 --out data/sample/machine_b.csv

(Yes, this is the same simulator code again -- Lesson 10 will fix
this duplication properly by putting the simulator in ONE module.)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def simulate_machine(seed, fault_hour, hours=24):
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

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fault-hour", type=float, default=12.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    df = simulate_machine(args.seed, args.fault_hour)
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    df.to_csv(out, index=False)
    print(f"saved {out} (seed={args.seed}, fault at {args.fault_hour}h)")
