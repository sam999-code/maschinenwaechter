"""Lesson 1: What does a machine sensor look like?

A vibration sensor on a healthy motor produces a repeating wave.
Here we CREATE such a signal ourselves, so later we know what
"normal" looks like -- and can spot "abnormal".
"""

import numpy as np  # numpy = fast math with lists of numbers
import matplotlib.pyplot as plt  # matplotlib = drawing charts

# --- 1. Create the timeline ---------------------------------------------
# A sensor measures, for example, 100 times per second for 2 seconds.
# That gives us 200 measurement points in a row. This row of numbers
# over time is called a "time series" -- our main data type in this project.
sampling_rate = 100  # measurements per second (Hz)
duration_s = 2  # seconds
n_points = sampling_rate * duration_s  # total points: 200

time = np.arange(n_points) / sampling_rate  # [0.00, 0.01, 0.02, ... 1.99]

# --- 2. Build a "healthy machine" signal --------------------------------
# A healthy motor vibrates at a steady rhythm (a sine wave)
# plus a tiny bit of random noise (real sensors are never perfectly clean).
frequency_hz = 5  # the motor vibrates 5 times per second
amplitude_mm_s = 2.0  # vibration strength: 2 millimeters per second

# np.sin() makes the repeating wave; np.random.normal adds small noise.
rng = np.random.default_rng(seed=42)  # seed=42 -> same "random" every run
noise = rng.normal(loc=0.0, scale=0.2, size=n_points)  # small random jitter

signal = amplitude_mm_s * np.sin(2 * np.pi * frequency_hz * time) + noise

# --- 3. Draw it ----------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(time, signal)
plt.title("Healthy motor: vibration signal (5 Hz, amplitude 2 mm/s)")
plt.xlabel("time (seconds)")
plt.ylabel("vibration (mm/s)")
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/figures/lesson01_first_signal.png", dpi=120)
print("Saved plot to reports/figures/lesson01_first_signal.png")
print(f"First 5 signal values: {signal[:5].round(3)}")
