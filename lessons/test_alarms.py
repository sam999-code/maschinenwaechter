"""Tests for the W2-1 alarm-management layer."""

from machineguard.alarms import (AlertRule, AlertRegistry, ManagedAlerts,
                                 annunciate)

RULE = AlertRule(rule_id="t", asset="a", failure_mode="f", cause="c",
                 consequence="x", action="do something specific",
                 priority="high", raise_score=0.9, clear_score=0.7,
                 on_delay=3, off_delay=2)


def test_hysteresis_stops_flapping():
    """Scores oscillating 0.89/0.91 at the boundary must NOT chatter."""
    m = ManagedAlerts(RULE)
    events = []
    for _ in range(12):                # flapping around the raise line
        events += m.step(1.0, 0.91)
        events += m.step(1.0, 0.89)
    assert len([e for e in events if e.kind == "raise"]) <= 1  # no chatter


def test_on_delay_requires_persistence():
    m = ManagedAlerts(RULE)
    assert m.step(1.0, 0.95) == []     # 1 window above: not yet
    assert m.step(1.0, 0.95) == []     # 2 windows: not yet
    ev = m.step(1.0, 0.95)             # 3rd consecutive -> raise
    assert len(ev) == 1 and ev[0].kind == "raise"


def test_off_delay_holds_alarm():
    m = ManagedAlerts(RULE)
    for _ in range(3):
        m.step(1.0, 0.95)              # raise
    assert m.step(2.0, 0.5) == []      # 1 clear window: still active
    ev = m.step(2.0, 0.5)              # 2nd clear window -> clear event
    assert len(ev) == 1 and ev[0].kind == "clear"


def test_registry_quarantines_unregistered():
    reg = AlertRegistry([RULE])
    state = ManagedAlerts(RULE)
    assert annunciate("unknown_rule", reg, 1.0, 99.0, state) == []


def test_priority_mix_report():
    reg = AlertRegistry([RULE,
                         AlertRule("l1", "a", "f", "c", "x", "act", "low"),
                         AlertRule("l2", "a", "f", "c", "x", "act", "low"),
                         AlertRule("m1", "a", "f", "c", "x", "act", "medium"),
                         AlertRule("l3", "a", "f", "c", "x", "act", "low")])
    mix = reg.priority_mix()
    # 3 low + 1 medium + 1 high out of 5 rules
    assert mix["low"] == 60.0 and mix["medium"] == 20.0 and mix["high"] == 20.0
