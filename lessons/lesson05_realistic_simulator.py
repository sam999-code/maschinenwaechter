"""Lesson 5: A realistic machine simulator (many hours, several sensors).

Reality differs from our rehearsal in Lessons 1-4 in three ways:
  1. TIME SCALE: faults develop over HOURS, not seconds
  2. DAY RHYTHM: temperature follows the day/night cycle
  3. MULTIPLE SENSORS: real machines have more than one

We simulate 24 hours at 1 measurement/second (= 86,400 points),
then run OUR DETECTOR from Lesson 3/4 on the result. Full circle.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path  # pathlib = modern, reliable path handling

from detector import rolling_zscore

# Paths relative to the PROJECT ROOT (the folder that contains lessons/),
# so the script works no matter which folder you run it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- 1. The timeline: 24 hours at 1 Hz -----------------------------------
hours = 24
fs = 1  # 1 measurement per second (keeps files small; real = 100-20,000 Hz)
n = hours * 3600 * fs
t_hours = np.arange(n) / (3600 * fs)  # time in hours: 0.0 ... 23.99

rng = np.random.default_rng(seed=42)
fault_hour = 16.0
fault_idx = int(fault_hour * 3600 * fs)

# --- 2. Vibration: healthy wave, then wear after hour 16 ------------------
# NB: at 1 Hz we cannot represent a 30 Hz real vibration wave (Nyquist!)
# so we simulate the vibration ENVELOPE - its strength over time.
vibration = rng.normal(0.0, 1.0, n)            # baseline wiggle
vibration += 0.5 * np.sin(2 * np.pi * t_hours)  # slow load variation
wear = np.exp(0.15 * (t_hours[fault_idx:] - fault_hour))  # growing damage
vibration[fault_idx:] *= wear
knock_pos = rng.integers(fault_idx, n, size=40)  # impacts, stronger over time
for i, pos in enumerate(knock_pos):
    vibration[pos] += rng.uniform(3, 6) * (1 + i / len(knock_pos))

# --- 3. Temperature: day/night cycle + heating after the fault -------------
temperature = 55 + 5 * np.sin(2 * np.pi * (t_hours - 8) / 24)  # day cycle
temperature += rng.normal(0.0, 0.1, n)                 # sensor noise
temperature[fault_idx:] += 2.0 * (t_hours[fault_idx:] - fault_hour)  # heating

# --- 4. Package as a DataFrame = the table every ML project uses ----------
df = pd.DataFrame({
    "hour": t_hours,
    "vibration": vibration,
    "temperature": temperature,
    "fault": (t_hours >= fault_hour).astype(int),  # 0 = healthy, 1 = fault
})
df.to_csv(PROJECT_ROOT / "data/sample/lesson05_machine.csv", index=False)
print(f"Simulated {len(df):,} measurements "
      f"({df['fault'].sum():,} in fault state)")

# --- 5. Our detector, applied to 24 hours of "factory data" ---------------
z, alert = rolling_zscore(df["vibration"].to_numpy(), window=3600, threshold=3.5)
alarms_healthy = alert[:fault_idx].sum()
alarms_fault = alert[fault_idx:].sum()
print(f"Detector alarms on HEALTHY hours: {alarms_healthy:,}")
print(f"Detector alarms on FAULT hours:   {alarms_fault:,}")

# --- 6. Overview plot ------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
axes[0].plot(t_hours, vibration, lw=0.3)
axes[0].set_ylabel("vibration")
axes[1].plot(t_hours, temperature, lw=0.5, color="darkorange")
axes[1].set_ylabel("temperature (°C)")
axes[2].plot(t_hours, z, lw=0.3, color="purple")
axes[2].plot(t_hours[alert], z[alert], "r.", ms=1, label="ALERT")
axes[2].axhline(3.5, color="red", ls="--", lw=0.8)
for ax in axes:
    ax.axvspan(fault_hour, hours, color="red", alpha=0.08)
axes[2].set_xlabel("hour of day")
axes[2].legend()
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports/figures/lesson05_simulator.png", dpi=110)
print("Saved plot to reports/figures/lesson05_simulator.png")
