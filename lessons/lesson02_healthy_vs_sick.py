"""Lesson 2: Healthy machine vs. sick machine -- seeing the difference.

We build TWO signals:
  1. healthy: steady wave + small noise (same as Lesson 1)
  2. sick: starts healthy, then a "damaged bearing" appears:
     - the vibration gets LOUDER over time (amplitude grows)
     - sharp random IMPACTS appear (the damaged surface hits
       other parts -- like a knocking sound)

Then we plot them one above the other and print simple numbers,
so your EYES learn the difference before we teach the computer.
"""

import numpy as np
import matplotlib.pyplot as plt

sampling_rate = 100  # 100 measurements per second
duration_s = 6  # watch the machine for 6 seconds
n_points = sampling_rate * duration_s
time = np.arange(n_points) / sampling_rate  # timeline: 0.00 ... 5.99 s

rng = np.random.default_rng(seed=42)

# --- 1. The HEALTHY signal (our reference "normal") ----------------------
frequency_hz = 5
healthy = 2.0 * np.sin(2 * np.pi * frequency_hz * time)
healthy = healthy + rng.normal(0.0, 0.2, n_points)  # small sensor noise

# --- 2. The SICK signal (healthy first, then it degrades) ----------------
# After second 3, two things happen, like a real damaged bearing:
sick = healthy.copy()  # start identical to healthy
fault_start_s = 3.0
fault_idx = int(fault_start_s * sampling_rate)  # array position of second 3

# (a) amplitude GROWS after the fault: multiply the wave by a factor
#     that increases from 1.0 to ~4.0 (exponential growth, like real wear)
growth = np.exp(0.5 * (time[fault_idx:] - fault_start_s))
sick[fault_idx:] = sick[fault_idx:] * growth

# (b) sharp IMPACTS: random "knocks". Each knock = a sudden spike.
#     We pick random moments after the fault and add a spike there.
n_knocks = 12
knock_positions = rng.integers(fault_idx, n_points, size=n_knocks)
for pos in knock_positions:
    sick[pos] += rng.uniform(4.0, 8.0)  # a sharp jump of 4-8 mm/s

# --- 3. Simple numbers: average vibration strength before vs after -------
strength_before = np.mean(np.abs(sick[:fault_idx]))   # avg |value| before
strength_after = np.mean(np.abs(sick[fault_idx:]))    # avg |value| after
print(f"Average vibration strength BEFORE second 3: {strength_before:.2f} mm/s")
print(f"Average vibration strength AFTER  second 3: {strength_after:.2f} mm/s")
print(f"That's {strength_after / strength_before:.1f}x stronger!")

# --- 4. Draw both signals, one above the other ---------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(time, healthy, color="green")
axes[0].set_title("HEALTHY machine: steady wave, small noise")
axes[0].set_ylabel("vibration (mm/s)")
axes[0].grid(True)

axes[1].plot(time, sick, color="red")
axes[1].axvspan(fault_start_s, duration_s, color="red", alpha=0.1)  # shade fault zone
axes[1].set_title("SICK machine: wave grows + sharp knocks (red zone)")
axes[1].set_xlabel("time (seconds)")
axes[1].set_ylabel("vibration (mm/s)")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("reports/figures/lesson02_healthy_vs_sick.png", dpi=120)
print("Saved plot to reports/figures/lesson02_healthy_vs_sick.png")
