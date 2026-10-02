r"""W2-4 demo: calibrate per machine -> verify -> re-run the W2-3 evaluation.

Run from lessons/:  ..\.venv\Scripts\python.exe w2_lesson04_calibration.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from machineguard.pipeline import (make_windows, apply_scaler, score_windows,
                                   load_bundle)
from machineguard.alarms import (AlertRule, AlertRegistry, ManagedAlerts,
                                 annunciate)
from machineguard.calibration import calibrate_rule, verify_alerts, measured_ppv
from machineguard.ppv import evaluate_alerts, scorecard_line

PROJECT_ROOT = Path(__file__).resolve().parent.parent

template = AlertRule(rule_id="bearing_wear", asset="{asset}",
                     failure_mode="bearing wear", cause="knocks",
                     consequence="seizure", action="inspect within 48 h",
                     priority="high", raise_score=3.0, clear_score=2.0,
                     on_delay=5, off_delay=10)

MACHINES = [("machine_a", "lesson05_machine.csv", 16.0),
            ("machine_b", "machine_b.csv", 12.0)]
model, cfg = load_bundle(PROJECT_ROOT / "models/machineguard_v1")

uncalibrated_raises, calibrated_raises = [], []
streams = {}
for asset, csv_name, fault_h in MACHINES:
    df = pd.read_csv(PROJECT_ROOT / "data/sample" / csv_name)
    windows = make_windows(apply_scaler(df, np.array(cfg["mean"]), np.array(cfg["std"])),
                           cfg["window"], cfg["stride"])
    hours = df["hour"].to_numpy()[cfg["window"] // 2 :: cfg["stride"]][: len(windows)]
    errors = score_windows(model, windows)
    streams[asset] = (hours, errors, fault_h)

    # --- uncalibrated (the W2-3 pipeline) ---
    st = ManagedAlerts(template)
    ev = []
    for h, s in zip(hours, errors):
        ev += annunciate("bearing_wear", AlertRegistry([template]), h, s, st)
    uncalibrated_raises += [(asset, e.hour) for e in ev if e.kind == "raise"]

    # --- W2-4: per-machine calibration from the first 6 healthy hours ---
    rule = calibrate_rule(template, errors, hours, calibrate_hours=6.0,
                          safety_factor=1.2)
    print(f"{asset}: calibrated thresholds raise={rule.raise_score:.2f} "
          f"clear={rule.clear_score:.2f} (template was 3.0/2.0)")
    st = ManagedAlerts(rule)
    ev = []
    for h, s in zip(hours, errors):
        ev += annunciate("bearing_wear", AlertRegistry([rule]), h, s, st)
    calibrated_raises += [(asset, e.hour) for e in ev if e.kind == "raise"]

failures = {a: f for a, _, f in MACHINES}
print("\n=== W2-3 evaluation, BEFORE calibration ===")
print("  ", scorecard_line(evaluate_alerts(uncalibrated_raises, failures,
                                                     lead_h=1.5, grace_h=1.0)))
print("=== W2-4 evaluation, AFTER per-machine calibration ===")
res_cal = evaluate_alerts(calibrated_raises, failures,
                          lead_h=1.5, grace_h=1.0)
print("  ", scorecard_line(res_cal))

# --- verification loop on the calibrated alarms (SIMULATED dispositions) --
print("\nVERIFICATION LOOP (dispositions SIMULATED from known fault times:")
print("true if alarm within 6 h of the real fault - labeled simulation)")
verdicts = []
for asset, csv_name, fault_h in MACHINES:
    hours, errors, _ = streams[asset]
    rule = calibrate_rule(template, errors, hours, safety_factor=1.2)
    st = ManagedAlerts(rule)
    ev = []
    for h, s in zip(hours, errors):
        ev += annunciate("bearing_wear", AlertRegistry([rule]), h, s, st)
    asset_raises = [e for e in ev if e.kind == "raise"]
    disp = {e.rule_id: ("true_positive" if abs(e.hour - fault_h) <= 6.0
                        else "false_positive") for e in asset_raises}
    verdicts += verify_alerts(ev, errors, hours, rule, dispositions=disp)

for v in verdicts:
    print(f"  alarm@{v['hour']:6.2f}h  disposition={v['disposition']:14s}"
          f" returned_to_normal={v['returned_to_normal']}  -> {v['verdict']}")
ppv = measured_ppv(verdicts)
print(f"\nMEASURED PPV (verified alarms): {ppv if ppv is not None else 'unknown'}")
print("\nThe loop is closed: alarm -> disposition -> verdict -> measured PPV.")
print("Re-run this script after changing calibration knobs - the numbers")
print("respond, and THAT is an engineering system, not a demo.")
