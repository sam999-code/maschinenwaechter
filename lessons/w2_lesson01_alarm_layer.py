"""W2-1 demo: raw threshold vs the managed alarm layer, on machine B.

You saw the raw-threshold behavior in Lesson 9 (single-point noise
alarms). Here we run BOTH pipelines on the same anomaly scores and
count what each produces. Expect: the managed layer fires far fewer,
but *real*, alarms.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from machineguard.pipeline import (make_windows, apply_scaler, score_windows,
                                   sustained_alerts, load_bundle)
from machineguard.alarms import (AlertRule, AlertRegistry, ManagedAlerts,
                                 annunciate, write_alarm_log)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- the Alert Registry (rationalization-as-code) -------------------------
rules = [
    AlertRule(rule_id="bearing_wear_vibration", asset="machine_b",
              failure_mode="bearing wear",
              cause="damaged bearing surface creates impact knocks",
              consequence="bearing seizure; unplanned production stop",
              action="schedule inspection within 48 h; order spare bearing",
              priority="high",
              raise_score=3.0, clear_score=2.0,   # in units of the AE error
              on_delay=5, off_delay=10),
    AlertRule(rule_id="slow_drift_trend", asset="machine_b",
              failure_mode="slow degradation trend",
              cause="gradual wear absorbed by rolling statistics",
              consequence="late detection of developing fault",
              action="review at weekly KPI meeting",
              priority="low",
              raise_score=4.0, clear_score=3.0,
              on_delay=20, off_delay=30),
]
registry = AlertRegistry(rules)
print("registry priority mix (target ~80/15/5):", registry.priority_mix())

# --- score machine B (same pipeline as Lessons 9-10) -----------------------
df = pd.read_csv(PROJECT_ROOT / "data/sample/machine_b.csv")
model, cfg = load_bundle(PROJECT_ROOT / "models/machineguard_v1")
windows = make_windows(apply_scaler(df, np.array(cfg["mean"]), np.array(cfg["std"])),
                       cfg["window"], cfg["stride"])
hours = df["hour"].to_numpy()[cfg["window"] // 2 :: cfg["stride"]][: len(windows)]
errors = score_windows(model, windows)
flagged = errors > cfg["threshold"]

# --- pipeline A: the OLD way (raw threshold + run-length, Lesson 9) --------
raw_alarms = sustained_alerts(flagged, run_length=5)
print(f"\nA) raw threshold pipeline:      {int(raw_alarms.sum()):5d} alarmed windows")

# --- pipeline B: the W2 managed layer --------------------------------------
states = {r.rule_id: ManagedAlerts(r) for r in rules}
managed_events = []
for h, s in zip(hours, errors):
    for rule in rules:
        managed_events += annunciate(rule.rule_id, registry, h, s,
                                     states[rule.rule_id])
raises = [e for e in managed_events if e.kind == "raise"]
print(f"B) managed layer:               {len(raises):5d} annunciated alarms "
      f"({len([e for e in managed_events if e.kind == 'clear'])} clears)")

# unregistered rule -> quarantined (nothing comes out)
ghost = ManagedAlerts(rules[0])
quarantined = annunciate("unregistered_experiment", registry, 12.0, 9.9, ghost)
print(f"   quarantined (unregistered rule): {len(quarantined)} events sent")

log_path = PROJECT_ROOT / "reports/alarm_log.csv"
write_alarm_log(managed_events, log_path)
print(f"\nalarm log written: {log_path}")
print("\nKEY INSIGHT: the managed layer turns 'windows flagged' (a number)")
print("into 'alarms' (discrete, documented, actionable events) - the unit")
print("that ISA-18.2 KPIs (next lesson) are computed on.")
