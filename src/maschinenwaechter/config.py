"""Configuration loading for maschinenwaechter.

Keeps the dependency footprint light: pyyaml is preferred, but if it is
missing we fall back to returning the YAML content as a plain dict built
from a minimal parser is NOT attempted — instead a small built-in default
dict is used so the package still imports. In practice pyyaml is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


@dataclass
class FaultConfig:
    """Fault-injection parameters (see config/settings.yaml for comments)."""

    transition_hours: float = 2.0
    bearing_wear: dict[str, float] = field(default_factory=dict)


@dataclass
class Settings:
    """Top-level settings dataclass mirroring settings.yaml."""

    sampling_hz: int = 10
    generation_hz: int = 1
    window_samples: int = 300
    zscore_threshold: float = 3.5
    fault: FaultConfig = field(default_factory=FaultConfig)
    paths: dict[str, str] = field(default_factory=dict)


def _to_settings(raw: dict[str, Any]) -> Settings:
    """Convert a raw (nested) dict into a :class:`Settings` dataclass."""
    fault_raw = raw.get("fault", {}) or {}
    return Settings(
        sampling_hz=int(raw.get("sampling_hz", 10)),
        generation_hz=int(raw.get("generation_hz", 1)),
        window_samples=int(raw.get("window_samples", 300)),
        zscore_threshold=float(raw.get("zscore_threshold", 3.5)),
        fault=FaultConfig(
            transition_hours=float(fault_raw.get("transition_hours", 2.0)),
            bearing_wear=dict(fault_raw.get("bearing_wear", {}) or {}),
        ),
        paths=dict(raw.get("paths", {}) or {}),
    )


def load_settings(path: str | Path | None = None) -> Settings:
    """Load ``config/settings.yaml`` and return a :class:`Settings` object.

    Parameters
    ----------
    path:
        Optional explicit path to a YAML settings file. Defaults to the
        repo's ``config/settings.yaml``.

    Returns
    -------
    Settings
        Dataclass with typed fields. Falls back to sensible defaults if the
        file is missing or pyyaml is not installed.
    """
    settings_path = Path(path) if path else DEFAULT_SETTINGS_PATH

    if not settings_path.exists():
        return Settings()

    try:
        import yaml  # preferred; preinstalled in the managed runtime
    except ImportError:  # pragma: no cover - defensive fallback
        return Settings()

    with open(settings_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return _to_settings(raw)
