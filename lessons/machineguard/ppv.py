"""W2-3: PPV-first alert evaluation - "were our alarms TRUE?"

The headline trust metric for an alarm system is PPV (positive
predictive value = precision): of all alarms raised, how many were
real? Human-factors research treats PPV as THE measure of a warning
system's diagnostic value - and it collapses when alerts are frequent.

Method (from the PdM literature, per-run first-trigger evaluation):
  - a failure episode earns AT MOST one True Positive: its FIRST alarm
    within the lead window [fault - lead, fault + grace]
  - every alarm outside any lead window is a False Positive
  - TPR = caught failures / all failures
  - PPV = TP / (TP + FP)
  - F-score combines both with beta (beta < 1 favors precision)
  - lead time of the first TP = the alarm's practical value
"""


def evaluate_alerts(raises: list, failures: dict,
                    lead_h: float = 1.0, grace_h: float = 0.5,
                    beta: float = 0.5) -> dict:
    """raises: list of (asset, hour) alarm raise times.
    failures: {asset: fault_hour}. Returns the PPV-first scorecard.

    beta < 1 favors precision (trust); beta > 1 favors recall (safety).
    Choose based on the plant's cost structure - Lesson 7 returns!"""
    raises = sorted(raises, key=lambda r: r[1])
    n_fail = len(failures)
    caught: dict[str, float] = {}     # asset -> first alarm hour that caught it
    tp = fp = 0
    for asset, hour in raises:
        fault = failures.get(asset)
        if fault is not None and fault - lead_h <= hour <= fault + grace_h:
            if asset not in caught:            # first trigger only
                caught[asset] = hour
                tp += 1
            # later alarms inside the same lead window: ignored, not FP
        else:
            fp += 1
    fn = n_fail - len(caught)
    ppv = tp / (tp + fp) if tp + fp else 0.0
    tpr = tp / n_fail if n_fail else 0.0
    fpr = fp / (fp + len(caught)) if fp + len(caught) else 0.0
    b2 = beta ** 2
    fbeta = ((1 + b2) * (1 - fpr) * tpr / (b2 * (1 - fpr) + tpr)
             if tpr > 0 else 0.0)
    leads = {a: failures[a] - h for a, h in caught.items()}
    return {
        "alarms": tp + fp, "TP": tp, "FP": fp, "FN": fn,
        "PPV": round(ppv, 3), "TPR": round(tpr, 3),
        "FPR": round(fpr, 3), f"F{beta:g}": round(fbeta, 3),
        "mean_lead_h": round(sum(leads.values()) / len(leads), 2) if leads else None,
    }


def scorecard_line(result: dict) -> str:
    return (f"alarms={result['alarms']:3d}  PPV={result['PPV']:.1%}  "
            f"TPR={result['TPR']:.1%}  FPR={result['FPR']:.1%}  "
            f"mean lead={result['mean_lead_h']} h")
