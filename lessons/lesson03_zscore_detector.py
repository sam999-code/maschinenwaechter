"""Lesson 3: Detecting the fault AUTOMATICALLY with a rolling z-score.

In Lesson 2 your EYES saw the difference. Now we teach the computer
to see it. The idea (our first real ML algorithm):

  "A point is suspicious if it is far away from what is normal
   RIGHT NOW, measured in units of how much the signal usually wiggles."

That distance measure is called the z-score:

      z(t) = | signal(t) - rolling_mean(t) | / rolling_std(t)

  - rolling_mean(t): average of the signal over the last W points
  - rolling_std(t):  how much the signal usually wiggles in that window
  - if z(t) > 3  ->  "this point is 3 wiggles away from normal -> ALERT"

This is exactly the "RollingZScoreDetector" used in real industry!
"""

import numpy as np
import pandas as pd  # pandas gives us the easy .rolling() tool
import matplotlib.pyplot as plt
from pathlib import Path  # Lesson 5 taught us: never trust relative paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- 0. Build the same SICK signal as Lesson 2 ---------------------------
sampling_rate = 100
duration_s = 6
n_points = sampling_rate * duration_s
time = np.arange(n_points) / sampling_rate
rng = np.random.default_rng(seed=42)

healthy = 2.0 * np.sin(2 * np.pi * 5 * time) + rng.normal(0.0, 0.2, n_points)

sick = healthy.copy()
fault_start_s = 3.0
fault_idx = int(fault_start_s * sampling_rate)
growth = np.exp(0.5 * (time[fault_idx:] - fault_start_s))
sick[fault_idx:] = sick[fault_idx:] * growth
for pos in rng.integers(fault_idx, n_points, size=12):
    sick[pos] += rng.uniform(4.0, 8.0)

# --- 1. The rolling z-score detector (from scratch, ~10 lines) ------------
window = 200  # look at the last 200 points (= 2 seconds) at each moment
threshold = 3.0

s = pd.Series(sick)  # wrap the array so we can use .rolling()
rolling_mean = s.rolling(window=window).mean()   # moving average
rolling_std = s.rolling(window=window).std()     # moving wiggle size

# guard: at the very start there is no full window yet -> std = 0.
# Dividing by zero gives NaN, so we fill those with 1 (harmless, we
# also ignore the first `window` points later anyway).
rolling_std = rolling_std.replace(0.0, np.nan).fillna(1.0)

z = (s - rolling_mean).abs() / rolling_std  # THE z-score, one number per point
alert = (z > threshold).fillna(False)       # True/False alarm per point

# --- 2. Did it work? Count alarms before vs. after the fault --------------
alarms_before = alert[:fault_idx].sum()
alarms_after = alert[fault_idx:].sum()
print(f"Alarm points BEFORE the fault: {alarms_before}  (want: ~0)")
print(f"Alarm points AFTER  the fault: {alarms_after}  (want: many)")
print(f"First alarm at second: {time[np.argmax(alert.values)]:.2f}")

# --- 3. Draw: signal with alarms marked + the z-score curve --------------
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(time, sick, color="gray", lw=0.8, label="signal")
axes[0].plot(time[alert], sick[alert], "r.", label="ALERT")
axes[0].axvspan(fault_start_s, duration_s, color="red", alpha=0.1)
axes[0].set_ylabel("vibration (mm/s)")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(time, z, color="blue", lw=0.8)
axes[1].axhline(threshold, color="red", ls="--", label="threshold z=3")
axes[1].axvspan(fault_start_s, duration_s, color="red", alpha=0.1)
axes[1].set_xlabel("time (seconds)")
axes[1].set_ylabel("z-score")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports/figures/lesson03_zscore_detector.png", dpi=120)
print("Saved plot to reports/figures/lesson03_zscore_detector.png")
