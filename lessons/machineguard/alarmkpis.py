"""W2-2: The ISA-18.2 KPI suite - measuring the alarm system itself.

Lesson W2-1 produced ALARM EVENTS. This module answers the question
ISA-18.2's "Monitoring & Assessment" lifecycle stage asks every week:

    "Is our alarm system healthy - or is it training operators to
     ignore it?"

Every KPI below is a one-pass computation over the alarm log. Targets
are the verified numbers from the research brief (EEMUA 191 / ISA-18.2):

  - average rate   ~1 alarm / 10 min per operator  (we report per day)
  - alarm flood    >10 alarms in any 10-min window; <1% of time in flood
  - stale alarms   raised but never cleared (>24 h active) -> target ~0
  - priority mix   ~80% low / 15% medium / 5% high (High >10% = inflation)
  - bad actors     top-10 rules should cause <1-5% of all alarms
  - PPV            headline trust metric - needs dispositions (W2-4);
                   until then we report "unknown", honestly.
"""

from collections import Counter
from dataclasses import dataclass, field

FLOOD_WINDOW_H = 10 / 60        # 10 minutes, in hours
FLOOD_THRESHOLD = 10            # >10 alarms in 10 min = flood
STALE_H = 24.0                  # alarm active longer than this = stale
TARGET_MIX = {"low": 80.0, "medium": 15.0, "high": 5.0}


@dataclass
class KpiReport:
    total_raises: int
    raises_per_day: float
    flood_windows: int = 0
    pct_time_in_flood: float = 0.0
    stale_alarms: list = field(default_factory=list)
    priority_mix: dict = field(default_factory=dict)
    priority_inflation: bool = False
    bad_actors: list = field(default_factory=list)   # [(rule_id, count, share%)]
    ppv: str = "unknown (needs dispositions - W2-4)"

    def render(self) -> str:
        lines = [
            "WEEKLY ALARM KPI REPORT (ISA-18.2 Monitoring & Assessment)",
            "=" * 58,
            f"  alarms raised:        {self.total_raises}"
            f"  ({self.raises_per_day:.1f}/day)",
            f"  flood windows:        {self.flood_windows}"
            f"  ({self.pct_time_in_flood:.2f}% of time; target <1%)",
            f"  stale alarms (>24h):  {len(self.stale_alarms)}"
            f"  {self.stale_alarms if self.stale_alarms else ''} (target ~0)",
            f"  priority mix:         {self.priority_mix} (target {TARGET_MIX})",
            f"  PRIORITY INFLATION:   {'YES - fix the registry!'
                                       if self.priority_inflation else 'no'}",
            f"  top bad actors:       {self.bad_actors if self.bad_actors else 'none'}",
            f"  PPV (trust metric):   {self.ppv}",
        ]
        return "\n".join(lines)


def compute_kpis(events: list, total_hours: float,
                 end_hour: float | None = None) -> KpiReport:
    """events: AlarmEvent list (from machineguard.alarms). total_hours:
    length of the observation period (for the per-day rate)."""
    raises = [e for e in events if e.kind == "raise"]
    clears = {e.rule_id: e.hour for e in events if e.kind == "clear"}
    end = end_hour if end_hour is not None else max(
        (e.hour for e in events), default=0.0)

    report = KpiReport(
        total_raises=len(raises),
        raises_per_day=len(raises) / max(total_hours / 24.0, 1e-9),
    )

    # --- flood analysis: sliding 10-minute windows over raise times ------
    times = sorted(e.hour for e in raises)
    flood_windows = 0
    if times:
        for start in times:
            n = sum(1 for t in times if start <= t < start + FLOOD_WINDOW_H)
            if n > FLOOD_THRESHOLD:
                flood_windows += 1
        report.flood_windows = 1 if flood_windows else 0   # count episodes
        report.pct_time_in_flood = min(100.0, 100.0 * flood_windows *
                                       FLOOD_WINDOW_H / max(total_hours, 1e-9))
    # --- stale alarms: raised, never cleared, active >24 h ---------------
    report.stale_alarms = [e.rule_id for e in raises
                           if e.rule_id not in clears
                           and end - e.hour > STALE_H]
    # --- priority distribution + inflation check --------------------------
    counts = Counter(e.priority for e in raises)
    total = max(1, sum(counts.values()))
    report.priority_mix = {p: round(100 * counts.get(p, 0) / total, 1)
                           for p in ("low", "medium", "high")}
    report.priority_inflation = report.priority_mix.get("high", 0) > 10.0
    # --- bad actors: rules causing a disproportionate share ---------------
    per_rule = Counter(e.rule_id for e in raises)
    top = per_rule.most_common(10)
    report.bad_actors = [(rid, n, round(100 * n / total, 1))
                         for rid, n in top if n > 1]
    return report
