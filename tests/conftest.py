"""Shared pytest fixtures and import-path bootstrap for the test suite.

The project uses a ``src/`` layout but is intentionally NOT pip-installed
(see README: we avoid packaging friction in the teaching setup). Instead,
both the project root and ``src/`` are appended to ``sys.path`` here so
``import maschinenwaechter`` works from anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

# Fast parameters shared by all fixtures: small durations at 10 Hz keep the
# whole test suite under a few seconds while exercising realistic windows.
DURATION_HOURS = 2.0
SAMPLING_HZ = 10
FAULT_AT_HOUR = 1.0  # halfway through the 2 h run


@pytest.fixture(scope="session")
def healthy_df():
    """A reproducible, fully healthy 2 h machine run (seeded)."""
    from maschinenwaechter.simulation import simulate_machine

    return simulate_machine(
        duration_hours=DURATION_HOURS,
        sampling_hz=SAMPLING_HZ,
        fault_at_hour=None,
        seed=123,
    )


@pytest.fixture(scope="session")
def faulty_df():
    """A reproducible 2 h run with the fault injected at the 1 h mark."""
    from maschinenwaechter.simulation import simulate_machine

    return simulate_machine(
        duration_hours=DURATION_HOURS,
        sampling_hz=SAMPLING_HZ,
        fault_at_hour=FAULT_AT_HOUR,
        fault_type="bearing_wear",
        seed=123,
    )
