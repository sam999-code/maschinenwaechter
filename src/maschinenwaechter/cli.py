"""Command-line interface for maschinenwaechter.

Usage
-----
    python -m maschinenwaechter.cli simulate \
        --out data/sample/machine_01.csv \
        --plot reports/figures/machine_01.png

The CLI is intentionally thin: all real logic lives in the ``simulation``
and ``detection`` packages so it can be imported (and tested) directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: safe on servers and in CI

import matplotlib.pyplot as plt  # noqa: E402

from .detection import RollingZScoreDetector  # noqa: E402
from .simulation import simulate_machine  # noqa: E402

# Columns plotted in the overview figure, in panel order.
PLOT_COLUMNS = ("vibration_mm_s", "temperature_c", "acoustic_db")
PLOT_LABELS = {
    "vibration_mm_s": "Vibration (mm/s)",
    "temperature_c": "Temperature (°C)",
    "acoustic_db": "Acoustic (dB)",
}


def plot_machine(df, out_path: Path, title: str = "Machine sensor overview") -> None:
    """Plot vibration, temperature and acoustic with the fault region shaded.

    Parameters
    ----------
    df:
        Frame produced by :func:`simulate_machine` (needs a ``fault`` column
        to shade; without it no shading is drawn).
    out_path:
        Where to save the PNG (parent directories are created).
    title:
        Figure title.
    """
    fig, axes = plt.subplots(len(PLOT_COLUMNS), 1, figsize=(12, 8), sharex=True)

    for ax, column in zip(axes, PLOT_COLUMNS):
        ax.plot(df["timestamp"], df[column], linewidth=0.5, color="#1f77b4")
        ax.set_ylabel(PLOT_LABELS[column], fontsize=9)
        ax.grid(alpha=0.3)

        # Shade the faulty region red once, spanning all panels.
        if "fault" in df.columns and df["fault"].max() > 0:
            fault_start = df.loc[df["fault"] == 1, "timestamp"].min()
            ax.axvspan(fault_start, df["timestamp"].max(), color="red", alpha=0.10)
            axes[0].annotate(
                "fault injected", xy=(fault_start, axes[0].get_ylim()[1]),
                fontsize=9, color="red", va="top",
            )

    axes[-1].set_xlabel("Time")
    fig.suptitle(title)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _cmd_simulate(args: argparse.Namespace) -> int:
    """Run one simulation and (optionally) write CSV + PNG artifacts."""
    df = simulate_machine(
        duration_hours=args.hours,
        sampling_hz=args.hz,
        fault_at_hour=args.fault_at,
        fault_type=args.fault_type,
        seed=args.seed,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} rows -> {out_path}")

    if args.plot:
        detector = RollingZScoreDetector(window=args.window, threshold=args.threshold)
        scored = detector.fit_transform(df)
        plot_path = Path(args.plot)
        plot_machine(scored, plot_path, title=f"Machine overview ({args.hours:.0f} h)")
        n_anomalies = int(scored["is_anomaly"].sum())
        print(f"Wrote plot -> {plot_path} ({n_anomalies} anomalous samples flagged)")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with the ``simulate`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="maschinenwaechter",
        description="Predictive-maintenance learning toolkit (simulation + baselines).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="Simulate one machine and optionally save CSV/PNG.")
    sim.add_argument("--hours", type=float, default=72.0, help="Simulation duration in hours.")
    sim.add_argument("--hz", type=int, default=1, help="Sampling rate in Hz.")
    sim.add_argument("--fault-at", type=float, default=None, help="Hour of fault injection (omit = healthy).")
    sim.add_argument("--fault-type", type=str, default="bearing_wear", help="Fault model name.")
    sim.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    sim.add_argument("--out", type=str, default=None, help="CSV output path (optional).")
    sim.add_argument("--plot", type=str, default=None, help="PNG output path (optional).")
    sim.add_argument("--window", type=int, default=300, help="Rolling z-score window (samples).")
    sim.add_argument("--threshold", type=float, default=3.5, help="Anomaly score threshold.")
    sim.set_defaults(func=_cmd_simulate)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse args and dispatch to the chosen subcommand."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
