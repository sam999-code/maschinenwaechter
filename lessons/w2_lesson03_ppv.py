"""W2-3 demo: PPV-first evaluation - raw windows vs managed alarms.

Run from lessons/:  ..\.venv\Scripts\python.exe w2_lesson03_ppv.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from machineguard.pipeline import (make_windows, apply_scaler, score_windows,
                                   sustained_alerts, load_bundle)
from machineguard.alarms import (AlertRule, AlertRegistry, ManagedAlerts,
                                 annunciate)
from machineguard.ppv import evaluate_alerts, scorecard_line

PROJECT_ROOT = Path(__file__).resolve().parent.parent

rules = [
    AlertRule(rule_id="bearing_wear", asset="{asset}",
              failure_mode="bearing wear", cause="knocks",
              consequence="seizure", action="inspect within 48 h",
              priority="high", raise_score=3.0, clear_score=2.0,
              on_delay=5, off_delay=10),
]
registry = AlertRegistry(rules)

MACHINES = [("machine_a", "lesson05_machine.csv", 16.0),
            ("machine_b", "machine_b.csv", 12.0)]

raw_raises, managed_raises = [], []
model, cfg = load_bundle(PROJECT_ROOT / "models/machineguard_v1")

for asset, csv_name, fault_h in MACHINES:
    df = pd.read_csv(PROJECT_ROOT / "data/sample" / csv_name)
    windows = make_windows(apply_scaler(df, np.array(cfg["mean"]), np.array(cfg["std"])),
                           cfg["window"], cfg["stride"])
    hours = df["hour"].to_numpy()[cfg["window"] // 2 :: cfg["stride"]][: len(windows)]
    errors = score_windows(model, windows)

    # raw pipeline: each sustained-run START counts as one "alert"
    raw = sustained_alerts(errors > cfg["threshold"], run_length=5)
    starts = np.where(raw & ~np.roll(raw, 1))[0]
    raw_raises += [(asset, hours[i]) for i in starts if i > 0]

    # managed pipeline: raise events only
    state = ManagedAlerts(rules[0])
    events = []
    for h, s in zip(hours, errors):
        events += annunciate("bearing_wear", registry, h, s, state)
    managed_raises += [(asset, e.hour) for e in events if e.kind == "raise"]

failures = {a: f for a, _, f in MACHINES}

print("PPV-FIRST EVALUATION (2 machines, failures at 16h / 12h)")
print("=" * 60)
print("A) raw sustained-windows alerts:")
print("  ", scorecard_line(evaluate_alerts(raw_raises, failures)))
print("\nB) W2 managed alarms (registry + hysteresis + delays):")
print("  ", scorecard_line(evaluate_alerts(managed_raises, failures)))
print("\nSURPRISE - the managed layer scored WORSE (PPV 0% vs 8.3%)!")
print("Investigate, don't assume: both managed alarms fired OUTSIDE the")
print("1-hour lead window (one ~10 h early, from thresholds calibrated")
print("on a DIFFERENT machine; one just after the grace period).")
print("\nThree lessons:")
print("  1) EVALUATION BEATS ASSUMPTIONS - we predicted the managed layer")
print("     would win; the numbers said otherwise. Science, not vibes.")
print("  2) An alarm 10 h early is still a FALSE POSITIVE for trust:")
print("     operators act, find nothing, stop trusting (PPV logic).")
print("  3) The fix is W2-4: per-machine calibration + the verification")
print("     loop - and then re-run THIS evaluation to prove it worked.")
