"""W2-4: Per-machine calibration + the verification loop. The finale.

CALIBRATION (the industry "learning phase"):
W2-3 proved that thresholds copied from another machine destroy PPV.
So each machine gets thresholds from ITS OWN first healthy hours:
    raise_score = percentile(healthy_errors, 99) * safety_factor
    clear_score = raise_score * 0.8          (hysteresis kept)
No fault can be expected during calibration - that is the assumption
real products document and operators confirm.

VERIFICATION LOOP (closed-loop alarm confirmation):
An alarm is only trustworthy if reality confirms it. After an alarm
clears, we check:
    (a) did the score return to the healthy band?  (machine agrees)
    (b) the technician's disposition: true / false / unverifiable
Verdicts feed back -> PPV per rule becomes MEASURED, not guessed.
(Our deep research found no standard mandates this; it is a gap -
which makes it a differentiator.)
"""

import numpy as np

from machineguard.alarms import AlertRule


def calibrate_rule(template: AlertRule, errors: np.ndarray, hours: np.ndarray,
                   calibrate_hours: float = 6.0,
                   percentile: float = 99.0, safety_factor: float = 2.0) -> AlertRule:
    """Adapt a registry rule's thresholds to THIS machine's healthy data.

    Uses only the first `calibrate_hours` of data (assumed healthy).
    Returns a NEW AlertRule - the registry template stays untouched."""
    healthy = errors[hours <= calibrate_hours]
    if len(healthy) < 100:
        raise ValueError("not enough healthy data to calibrate")
    raise_at = float(np.percentile(healthy, percentile)) * safety_factor
    clear_at = raise_at * 0.8
    return AlertRule(**{**template.__dict__,
                        "raise_score": round(raise_at, 4),
                        "clear_score": round(clear_at, 4)})


def verify_alerts(events: list, errors: np.ndarray, hours: np.ndarray,
                  calibrated_rule: AlertRule,
                  dispositions: dict[str, str] | None = None,
                  healthy_band: float | None = None) -> list[dict]:
    """Close the loop: score each RAISE event with (a) return-to-normal
    evidence and (b) the human disposition (if provided).

    Return-to-normal is measured AFTER the alarm's clear event (or at
    the end of data if still active): did the score come back into the
    healthy band and stay there? A 'true' alarm whose signal never
    returns to normal is flagged 'suspicious' for review.

    dispositions: {rule_id: "true_positive"|"false_positive"|"unverifiable"}
    In production this comes from the work-order closeout. In this demo
    we pass simulated dispositions - clearly labeled as such."""
    band = healthy_band if healthy_band is not None else calibrated_rule.clear_score
    clears = {e.rule_id: e.hour for e in events if e.kind == "clear"}
    verdicts = []
    for e in [x for x in events if x.kind == "raise"]:
        clear_h = clears.get(e.rule_id)
        from_h = clear_h if clear_h is not None else hours[-1]
        i = int(np.searchsorted(hours, from_h))
        after = errors[i:i + 30]                      # ~5 min from the clear on
        returned = bool(len(after) and np.median(after) < band)
        disp = (dispositions or {}).get(e.rule_id, "unverifiable")
        if disp == "true_positive" and returned:
            verdict = "confirmed_true"                # human + machine agree
        elif disp == "false_positive":
            verdict = "confirmed_false"               # human overruled
        elif disp == "true_positive" and not returned:
            verdict = "suspicious"                    # claims true, signal stuck
        else:
            verdict = "unverifiable"
        verdicts.append({"rule_id": e.rule_id, "hour": e.hour,
                         "disposition": disp, "returned_to_normal": returned,
                         "verdict": verdict})
    return verdicts


def measured_ppv(verdicts: list[dict]) -> float | None:
    """PPV from VERIFIED alarms only (confirmed_true / confirmed_false).
    None if nothing is verified yet - honest 'unknown'."""
    decided = [v for v in verdicts
               if v["verdict"] in ("confirmed_true", "confirmed_false")]
    if not decided:
        return None
    tp = sum(1 for v in decided if v["verdict"] == "confirmed_true")
    return round(tp / len(decided), 3)
