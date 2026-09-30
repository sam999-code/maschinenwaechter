"""Lesson 11: MachineGuard as a web service (FastAPI).

A CLI needs a human to run it. A SERVICE can be called by ANY software:
the factory's monitoring system, a phone app, your future SaaS backend.

Architecture (the standard ML-serving pattern):

  client (any program) --HTTP--> FastAPI app --calls--> machineguard package
                                        (model loaded ONCE at startup)

Run it:
  ..\\.venv\\Scripts\\python.exe -m uvicorn machineguard.service:app --port 8000

Try it:
  curl http://localhost:8000/health
  curl -F "file=@../data/sample/machine_b.csv" http://localhost:8000/analyze
"""

import io
import numpy as np
import pandas as pd
from pathlib import Path

from fastapi import FastAPI, UploadFile, Form

from .pipeline import (make_windows, apply_scaler, score_windows,
                       calibrate_threshold, sustained_alerts, load_bundle)

# The model bundle location comes from an ENVIRONMENT VARIABLE with a
# sensible default (12-factor app pattern: configuration via env vars).
# Why: the folder layout differs between the lessons/ dev setup and the
# Docker image (/app). Hardcoding one path would break the other.
import os
_DEFAULT_MODELS = Path(__file__).resolve().parent.parent.parent / "models" / "machineguard_v1"
MODELS_PREFIX = Path(os.environ.get("MACHINEGUARD_MODELS", _DEFAULT_MODELS))

# --- the standard pattern: load the model ONCE, serve many requests ------
app = FastAPI(title="MachineGuard", version="1.0")
STATE = {"model": None, "cfg": None}

@app.on_event("startup")
def load_model():
    STATE["model"], STATE["cfg"] = load_bundle(MODELS_PREFIX)

@app.get("/health")
def health():
    """Every production service exposes /health - monitoring tools
    (and Docker!) call it to check the service is alive."""
    return {"status": "ok", "model_loaded": STATE["model"] is not None}

@app.post("/analyze")
async def analyze(file: UploadFile, calibrate_hours: float = Form(0)):
    """Upload a machine CSV -> receive a JSON verdict.

    calibrate_hours > 0 enables the Lesson 10 'learning phase':
    the threshold is recomputed from the first N hours of THIS machine.
    """
    df = pd.read_csv(io.StringIO((await file.read()).decode("utf-8")))
    missing = [c for c in ("hour", "vibration", "temperature") if c not in df.columns]
    if missing:
        return {"error": f"CSV missing columns: {missing}"}

    cfg = STATE["cfg"]; model = STATE["model"]
    mean = np.array(cfg["mean"]); std = np.array(cfg["std"])
    windows = make_windows(apply_scaler(df, mean, std), cfg["window"], cfg["stride"])
    hours = df["hour"].to_numpy()[cfg["window"] // 2 :: cfg["stride"]][: len(windows)]
    errors = score_windows(model, windows)

    threshold = cfg["threshold"]
    if calibrate_hours > 0:  # per-machine learning phase (Lesson 10)
        early = hours <= calibrate_hours
        if early.sum() > 100:
            threshold = calibrate_threshold(errors[early], 99)

    flagged = errors > threshold
    alarm = sustained_alerts(flagged, run_length=5)
    return {
        "file": file.filename,
        "windows_analyzed": int(len(windows)),
        "windows_flagged": int(flagged.sum()),
        "sustained_alarm_windows": int(alarm.sum()),
        "threshold_used": round(threshold, 4),
        "first_sustained_alert_hour": (round(float(hours[np.argmax(alarm)]), 2)
                                       if alarm.any() else None),
        "verdict": "FAULT DETECTED" if alarm.mean() > 0.05 else "healthy",
    }
