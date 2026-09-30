"""The machine simulator - ONE canonical copy (was copy-pasted 3x).

Physics model (simplified but honest):
  - vibration: noise + slow load rhythm; after the fault: exponential
    wear growth + random impact knocks
  - temperature: day/night cycle; after the fault: linear heating
"""

import numpy as np
import pandas as pd


def simulate_machine(seed, fault_hour, hours=24):
    """Simulate one machine. Returns a DataFrame with columns:
    hour, vibration, temperature, fault (0/1)."""
    n = hours * 3600
    t = np.arange(n) / 3600
    rng = np.random.default_rng(seed)
    f_idx = int(fault_hour * 3600)
    fault_active = f_idx < n  # fault_hour beyond the window -> stays healthy

    vibration = rng.normal(0.0, 1.0, n) + 0.5 * np.sin(2 * np.pi * t)
    temperature = 55 + 5 * np.sin(2 * np.pi * (t - 8) / 24)
    temperature += rng.normal(0.0, 0.1, n)
    if fault_active:
        vibration[f_idx:] *= np.exp(0.15 * (t[f_idx:] - fault_hour))
        knocks = rng.integers(f_idx, n, size=40)
        for i, pos in enumerate(knocks):
            vibration[pos] += rng.uniform(3, 6) * (1 + i / len(knocks))
        temperature[f_idx:] += 2.0 * (t[f_idx:] - fault_hour)

    return pd.DataFrame({
        "hour": t, "vibration": vibration, "temperature": temperature,
        "fault": (t >= fault_hour).astype(int),
    })
