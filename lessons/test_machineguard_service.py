"""API tests for the MachineGuard service.

Run from the lessons/ folder:
  ..\\.venv\\Scripts\\python.exe -m pytest test_machineguard_service.py -v

We test the API WITHOUT starting a real server - FastAPI's TestClient
runs the app in-process. This is how APIs are tested professionally:
fast, deterministic, part of the normal test suite.
"""

import io
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from machineguard.service import app
from machineguard import simulate_machine


@pytest.fixture(scope="module")
def client():
    """TestClient as a context manager: only then does FastAPI run the
    'startup' event (where our model loads). Without this, /analyze
    would run against model=None."""
    with TestClient(app) as c:
        yield c


def csv_bytes(df):
    return io.BytesIO(df.to_csv(index=False).encode("utf-8"))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


def test_missing_columns_returns_error(client):
    bad = pd.DataFrame({"a": [1, 2, 3]})
    r = client.post("/analyze", files={"file": ("bad.csv", csv_bytes(bad))})
    assert r.status_code == 200
    assert "error" in r.json()


def test_detects_fault_with_learning_phase(client):
    """Small machine (1h), fault at 0.5h. With calibration on the first
    0.4h (healthy), the service must still catch the fault."""
    df = simulate_machine(seed=7, fault_hour=0.5, hours=1)
    r = client.post("/analyze",
                    files={"file": ("m.csv", csv_bytes(df))},
                    data={"calibrate_hours": 0.4})
    out = r.json()
    assert out["verdict"] == "FAULT DETECTED", out
    assert out["first_sustained_alert_hour"] is not None
    assert out["first_sustained_alert_hour"] < 0.9, out


def test_healthy_machine_verdict(client):
    """Fault at hour 30 never arrives -> 1 hour of pure health."""
    df = simulate_machine(seed=3, fault_hour=30.0, hours=1)
    r = client.post("/analyze", files={"file": ("ok.csv", csv_bytes(df))})
    out = r.json()
    assert out["windows_flagged"] / out["windows_analyzed"] < 0.10, out
