"""Generate the sample fleet used by the README, tests and notebook.

Run from the project root:

    python scripts/01_generate_data.py

Creates ``data/sample/machine_XX.csv`` and ``reports/figures/machine_XX.png``
for three machines (72 h each at 1 Hz, fault injected at ~40 h).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/ layout importable without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

from maschinenwaechter.cli import plot_machine  # noqa: E402
from maschinenwaechter.detection import RollingZScoreDetector  # noqa: E402
from maschinenwaechter.simulation import generate_dataset  # noqa: E402

N_MACHINES = 3
DURATION_HOURS = 72.0
SAMPLING_HZ = 1
FAULT_AT_HOUR = 40.0
SEED = 42

DATA_DIR = PROJECT_ROOT / "data" / "sample"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def main() -> None:
    """Generate CSVs and overview PNGs for the sample fleet."""
    machines = generate_dataset(
        n_machines=N_MACHINES,
        duration_hours=DURATION_HOURS,
        sampling_hz=SAMPLING_HZ,
        fault_at_hour=FAULT_AT_HOUR,
        seed=SEED,
    )
    detector = RollingZScoreDetector()  # defaults: window=300, threshold=3.5

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for name, df in machines.items():
        csv_path = DATA_DIR / f"{name}.csv"
        df.to_csv(csv_path, index=False)

        scored = detector.fit_transform(df)
        fig_path = FIGURES_DIR / f"{name}.png"
        plot_machine(scored, fig_path, title=f"{name} — {DURATION_HOURS:.0f} h, fault at {FAULT_AT_HOUR:.0f} h")
        print(f"{name}: {len(df)} rows -> {csv_path.name}, {int(scored['is_anomaly'].sum())} anomalies -> {fig_path.name}")


if __name__ == "__main__":
    main()
