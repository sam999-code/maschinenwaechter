"""W2-2 demo: the KPI suite on (a) our real machine B log and
(b) a synthetic "bad plant" log that exercises every KPI path.

Run from lessons/:  ..\.venv\Scripts\python.exe w2_lesson02_kpi_suite.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from machineguard.pipeline import (make_windows, apply_scaler, score_windows,
                                   load_bundle)
from machineguard.alarms import (AlertRule, AlertRegistry, ManagedAlerts,
                                 annunciate, AlarmEvent)
from machineguard.alarmkpis import compute_kpis

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- (a) the real machine B alarm stream ----------------------------------
df = pd.read_csv(PROJECT_ROOT / "data/sample/machine_b.csv")
model, cfg = load_bundle(PROJECT_ROOT / "models/machineguard_v1")
windows = make_windows(apply_scaler(df, np.array(cfg["mean"]), np.array(cfg["std"])),
                       cfg["window"], cfg["stride"])
hours = df["hour"].to_numpy()[cfg["window"] // 2 :: cfg["stride"]][: len(windows)]
errors = score_windows(model, windows)

rule = AlertRule(rule_id="bearing_wear_vibration", asset="machine_b",
                 failure_mode="bearing wear", cause="knocks",
                 consequence="seizure", action="inspect within 48 h",
                 priority="high", raise_score=3.0, clear_score=2.0,
                 on_delay=5, off_delay=10)
registry = AlertRegistry([rule])
state = ManagedAlerts(rule)
events = []
for h, s in zip(hours, errors):
    events += annunciate(rule.rule_id, registry, h, s, state)

print("(a) machine B - real stream")
print(compute_kpis(events, total_hours=24.0).render())

# --- (b) a synthetic BAD PLANT: flood + stale + inflation + bad actor -----
bad = []
for i in range(15):                                   # ALARM FLOOD at t=100
    bad.append(AlarmEvent("rule_a", 100 + i * 0.005, "raise", 5.0, "high"))
bad.append(AlarmEvent("rule_a", 101.0, "clear", 1.0, "high"))
for i in range(12):                                   # bad actor: rule_b
    bad.append(AlarmEvent("rule_b", 10 + i * 2.0, "raise", 4.0, "high"))
    bad.append(AlarmEvent("rule_b", 10 + i * 2.0 + 0.2, "clear", 1.0, "high"))
bad.append(AlarmEvent("rule_c", 5.0, "raise", 3.0, "high"))   # STALE: never
                                                               # cleared, >24h
print("\n(b) synthetic bad plant - every KPI should fire")
print(compute_kpis(bad, total_hours=168.0, end_hour=168.0).render())
print("\nRead (b) line by line: flood window detected, stale alarm listed,")
print("priority mix shows inflation (high >> 5%), bad actor named.")
print("This is the 'Monitoring & Assessment' stage - the system watching")
print("the alarm system. Next (W2-3): PPV-first evaluation of alerts.")
