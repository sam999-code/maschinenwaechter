"""Tests for the W2-2 KPI suite."""

from machineguard.alarms import AlarmEvent
from machineguard.alarmkpis import compute_kpis


def ev(rid, hour, kind, pri="high"):
    return AlarmEvent(rid, hour, kind, 5.0 if kind == "raise" else 1.0, pri)


def test_flood_detection():
    storm = [ev("r", 10 + i * 0.005, "raise") for i in range(15)]
    storm.append(ev("r", 11, "clear"))
    r = compute_kpis(storm, total_hours=24.0, end_hour=24.0)
    assert r.flood_windows == 1


def test_no_flood_for_sparse_alarms():
    sparse = [ev("r", 10, "raise"), ev("r", 12, "clear"),
              ev("r", 20, "raise"), ev("r", 22, "clear")]
    r = compute_kpis(sparse, total_hours=24.0, end_hour=24.0)
    assert r.flood_windows == 0


def test_stale_alarm_detection():
    e = [ev("r", 5, "raise")]          # never cleared, observation ends +48h
    r = compute_kpis(e, total_hours=168.0, end_hour=53.0)
    assert r.stale_alarms == ["r"]
    e2 = [ev("r", 5, "raise"), ev("r", 6, "clear")]
    r2 = compute_kpis(e2, total_hours=168.0, end_hour=53.0)
    assert r2.stale_alarms == []


def test_priority_inflation_flag():
    inflated = [ev("r", i, "raise", "high") for i in range(5)]
    r = compute_kpis(inflated, total_hours=24.0, end_hour=24.0)
    assert r.priority_mix["high"] == 100.0 and r.priority_inflation
    healthy = [ev("r", i, "raise", "low") for i in range(5)]
    r2 = compute_kpis(healthy, total_hours=24.0, end_hour=24.0)
    assert not r2.priority_inflation


def test_bad_actor_reporting():
    e = [ev("bad", i, "raise", "low") for i in range(6)] + \
        [ev("ok", 1, "raise", "low"), ev("ok", 2, "clear")]
    r = compute_kpis(e, total_hours=24.0, end_hour=24.0)
    assert r.bad_actors and r.bad_actors[0][0] == "bad"
