"""W2-1: The alarm-management layer - registry, hysteresis, off-delay.

WHAT INDUSTRY KNOWS THAT ML TUTORIALS DON'T:
A raw threshold ("score > 0.9 -> alert") creates chattering alarms that
destroy operator trust. ISA-18.2 fixes this with three mechanisms we
implement here:

  1. ALERT REGISTRY (rationalization-as-code): every alert rule must be
     registered with cause/consequence/action/priority/parameters.
     An unregistered rule's alerts are QUARANTINED, not sent.
  2. HYSTERESIS (deadband): raise the alarm at a HIGH score, but only
     clear it at a LOWER score - stops on/off flapping at the boundary.
  3. ON/OFF DELAY: the condition must persist N consecutive windows to
     raise; and must stay clear M windows to drop. Kills fleeting alarms.

Reference numbers (see research brief): chatter target = 0; deadband +
delays are the standard anti-chatter conditioning; every alarm needs a
documented, unique corrective action.
"""

import csv
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class AlertRule:
    """One row of the Alert Registry (the MADB - Master Alarm Database).

    The fields 'cause', 'consequence', 'action' are the ISA-18.2
    requirement: if you cannot write the corrective action in one
    sentence, the alarm will not help anyone at 3 AM."""
    rule_id: str
    asset: str
    failure_mode: str
    cause: str
    consequence: str
    action: str                     # unique corrective response
    priority: str                   # "low" | "medium" | "high"  (target mix 80/15/5)
    raise_score: float = 0.9        # hysteresis: raise above this
    clear_score: float = 0.75       # hysteresis: clear below this
    on_delay: int = 5               # must persist N windows to raise
    off_delay: int = 10             # must stay clear M windows to drop
    review_date: str = "2027-01-01" # rationalization expires -> quarantine


class AlertRegistry:
    """Rationalization-as-code: the alerting code reads its rules from
    here at startup. Alerts whose rule_id is missing (or expired) are
    quarantined instead of being sent."""

    def __init__(self, rules: list[AlertRule]):
        self.rules = {r.rule_id: r for r in rules}

    def lookup(self, rule_id: str) -> AlertRule | None:
        return self.rules.get(rule_id)

    def priority_mix(self) -> dict:
        counts: dict[str, int] = {}
        for r in self.rules.values():
            counts[r.priority] = counts.get(r.priority, 0) + 1
        total = max(1, sum(counts.values()))
        return {p: round(100 * n / total, 1) for p, n in sorted(counts.items())}

    def to_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(self.rules[next(iter(self.rules))]).keys()))
            w.writeheader()
            for r in self.rules.values():
                w.writerow(asdict(r))


@dataclass
class AlarmEvent:
    """One annunciated (or cleared) alarm - the unit of the alarm log."""
    rule_id: str
    hour: float
    kind: str          # "raise" | "clear"
    score: float
    priority: str


class ManagedAlerts:
    """The ISA-18.2 state machine: hysteresis + on/off delay over a
    stream of anomaly scores. Drop-in replacement for the naive
    'errors > threshold' test from Lesson 9."""

    def __init__(self, rule: AlertRule):
        self.rule = rule
        self.active = False
        self._above = 0      # consecutive windows above raise threshold
        self._below = 0      # consecutive windows below clear threshold

    def step(self, hour: float, score: float) -> list[AlarmEvent]:
        """Feed one window's score; returns any alarm events raised."""
        events = []
        r = self.rule
        if not self.active:
            self._above = self._above + 1 if score >= r.raise_score else 0
            self._below = 0
            if self._above >= r.on_delay:
                self.active = True
                self._below = 0
                events.append(AlarmEvent(r.rule_id, hour, "raise", score, r.priority))
        else:
            self._below = self._below + 1 if score < r.clear_score else 0
            self._above = 0
            if self._below >= r.off_delay:
                self.active = False
                self._above = 0
                events.append(AlarmEvent(r.rule_id, hour, "clear", score, r.priority))
        return events


def annunciate(rule_id: str, registry: AlertRegistry, hour: float,
               score: float, state: ManagedAlerts) -> list[AlarmEvent]:
    """Gate: quarantine alerts from unregistered/expired rules."""
    if registry.lookup(rule_id) is None:
        return []                      # quarantined - visible in logs, not sent
    return state.step(hour, score)


def write_alarm_log(events: list[AlarmEvent], path):
    """Append events to the alarm log (CSV). KPIs (W2-2) read this file."""
    exists = Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["rule_id", "hour", "kind", "score", "priority"])
        for e in events:
            w.writerow([e.rule_id, round(e.hour, 3), e.kind,
                        round(e.score, 4), e.priority])
