"""Physics-inspired factory machine sensor simulator.

This is the heart of Phase 1. The generator is deliberately simple but
follows physically plausible rules so that the anomaly-detection baseline
(Phase 3) has realistic signal to work with:

Healthy behaviour
-----------------
* ``vibration_mm_s``: sum of sinusoids at the shaft rotation frequency
  (``rpm / 60`` Hz) and its harmonics, plus Gaussian measurement noise.
* ``temperature_c``: a slow daily cycle (ambient / day-shift effect) around
  a base operating temperature, plus small noise.
* ``acoustic_db``: a base noise level with small fluctuations.
* ``rpm``: nominal speed with a slow load drift and small noise.

Fault "bearing_wear" (injected at ``fault_at_hour``)
----------------------------------------------------
* Vibration amplitude grows **exponentially** with time since fault onset
  (wear progresses faster once the raceway is damaged).
* **Impulsive shocks** are added: a damaged bearing produces short
  mechanical impact bursts as rolling elements hit the defect. They ramp in
  over ``shock_ramp_hours`` and are what a rolling z-score baseline detects
  most reliably (a slow drift alone would be absorbed by the rolling
  statistics).
* Temperature drifts up **linearly** (increased friction).
* Acoustic level rises by a fixed number of dB.
* The multiplicative wear transition is smoothed over ``transition_hours``
  instead of a hard step, which mirrors real machines.

All randomness flows through a single ``numpy.random.Generator`` so results
are bit-for-bit reproducible when ``seed`` is given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Physical constants (module-level so they are easy to find while teaching)
# ---------------------------------------------------------------------------
BASE_RPM = 1800.0            # nominal shaft speed (revolutions per minute)
RPM_DRIFT_AMPLITUDE = 30.0   # slow load drift of the rpm (±)
RPM_DRIFT_PERIOD_H = 8.0     # period of the load drift
RPM_NOISE_STD = 5.0          # per-sample rpm noise

VIB_HARMONIC_AMPLITUDES = (1.0, 0.5, 0.25)  # mm/s for 1st, 2nd, 3rd harmonic
VIB_NOISE_STD = 0.20         # mm/s measurement noise

TEMP_BASE_C = 55.0           # operating temperature
TEMP_DAY_AMPLITUDE = 5.0     # daily ambient cycle amplitude (°C)
TEMP_NOISE_STD = 0.30        # °C

ACOUSTIC_BASE_DB = 70.0      # healthy machine noise floor
ACOUSTIC_NOISE_STD = 0.50    # dB

DEFAULT_TRANSITION_HOURS = 2.0
DEFAULT_BEARING_WEAR = {
    "vibration_growth_rate": 0.12,   # exp() growth of vibration amplitude per hour
    "shock_ramp_hours": 10.0 / 60.0, # shocks reach full strength ~10 min after onset
    "shock_rate_per_s": 0.1,         # ~6 impulsive shocks per minute at full wear
    "shock_amplitude": 6.0,          # mm/s peak shock amplitude (±50 % jitter)
    "temperature_rise_per_hour": 0.35,  # linear friction heating °C/h
    "acoustic_rise_db": 6.0,         # step-like rise of the acoustic level
}


def _smooth_progress(hours_since_fault: np.ndarray, transition_hours: float) -> np.ndarray:
    """Return a 0→1 smooth progression of the fault over ``transition_hours``.

    Uses a cubic smoothstep so the transition has zero slope at both ends,
    which avoids artificial jumps that a detector could trivially key on.
    """
    x = np.clip(hours_since_fault / max(transition_hours, 1e-9), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _observable_shaft_freq(shaft_freq_hz: np.ndarray, sampling_hz: float, n_harmonics: int) -> np.ndarray:
    """Clamp the shaft frequency so its harmonics survive sampling.

    A real 1800 rpm machine rotates at 30 Hz, so its harmonics sit at
    30/60/90 Hz — far above the Nyquist limit of our 1-10 Hz teaching
    signals and therefore invisible (aliased away). For the simulation to
    keep a physically *shaped* vibration signal at any sampling rate, the
    shaft frequency used in the sine model is capped so that the highest
    harmonic stays below 0.4 * sampling_hz. The ``rpm`` column itself keeps
    the true physical values.
    """
    cap_hz = 0.4 * sampling_hz / n_harmonics
    return np.minimum(shaft_freq_hz, cap_hz)


def simulate_machine(
    duration_hours: float,
    sampling_hz: int = 10,
    fault_at_hour: float | None = None,
    fault_type: str = "bearing_wear",
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate one machine's sensor stream.

    Parameters
    ----------
    duration_hours:
        Total simulated runtime in hours.
    sampling_hz:
        Samples per second. Tests use 10 Hz; bulk generation uses 1 Hz.
    fault_at_hour:
        Hour at which the fault starts. ``None`` (default) means the machine
        stays healthy for the whole run.
    fault_type:
        Which fault model to inject. Only ``"bearing_wear"`` is implemented
        in Phase 1; anything else raises ``ValueError``.
    seed:
        Seed for ``numpy.random.default_rng``. Same seed -> identical frame.

    Returns
    -------
    pandas.DataFrame
        Columns: ``timestamp`` (datetime64, monotonic), ``vibration_mm_s``,
        ``temperature_c``, ``acoustic_db``, ``rpm`` (float64), ``fault``
        (0/1 int, 1 from ``fault_at_hour`` onwards) and ``fault_type`` (str).
    """
    if fault_type != "bearing_wear":
        raise ValueError(f"Unknown fault_type {fault_type!r}; only 'bearing_wear' is implemented.")

    rng = np.random.default_rng(seed)

    n_samples = int(duration_hours * 3600 * sampling_hz)
    if n_samples < 1:
        raise ValueError("duration_hours and sampling_hz must produce at least one sample.")

    t_seconds = np.arange(n_samples, dtype=float) / sampling_hz
    t_hours = t_seconds / 3600.0

    # --- rpm: nominal speed + slow load drift + noise ----------------------
    rpm = (
        BASE_RPM
        + RPM_DRIFT_AMPLITUDE * np.sin(2.0 * np.pi * t_hours / RPM_DRIFT_PERIOD_H)
        + rng.normal(0.0, RPM_NOISE_STD, n_samples)
    )
    shaft_freq_hz = rpm / 60.0  # true physical rotation frequency (Hz)
    # Frequency used inside the vibration sine model, kept below Nyquist so
    # the harmonics remain observable at any sampling_hz (see helper above).
    shaft_freq_model = _observable_shaft_freq(
        shaft_freq_hz, sampling_hz, len(VIB_HARMONIC_AMPLITUDES)
    )

    # --- healthy vibration: shaft harmonics + noise ------------------------
    vibration = rng.normal(0.0, VIB_NOISE_STD, n_samples)
    for harmonic, amplitude in enumerate(VIB_HARMONIC_AMPLITUDES, start=1):
        vibration += amplitude * np.sin(2.0 * np.pi * harmonic * shaft_freq_model * t_seconds)

    # --- healthy temperature: daily cycle + noise --------------------------
    temperature = (
        TEMP_BASE_C
        + TEMP_DAY_AMPLITUDE * np.sin(2.0 * np.pi * t_hours / 24.0)
        + rng.normal(0.0, TEMP_NOISE_STD, n_samples)
    )

    # --- healthy acoustic level -------------------------------------------
    acoustic = ACOUSTIC_BASE_DB + rng.normal(0.0, ACOUSTIC_NOISE_STD, n_samples)

    # --- fault injection ----------------------------------------------------
    fault_flag = np.zeros(n_samples, dtype=int)
    fault_label = "none"
    if fault_at_hour is not None:
        fault_label = fault_type
        hours_since_fault = np.clip(t_hours - fault_at_hour, 0.0, None)
        progress = _smooth_progress(hours_since_fault, DEFAULT_TRANSITION_HOURS)

        wear = DEFAULT_BEARING_WEAR
        # 1) Exponential vibration-amplitude growth (wear accelerates itself).
        growth = np.exp(wear["vibration_growth_rate"] * hours_since_fault)
        vibration = vibration * (1.0 + (growth - 1.0) * progress)
        # 2) Impulsive shocks from the rolling elements hitting the defect.
        #    They ramp in over ~10 min (independent of the slower multiplicative
        #    wear ramp) so the defect is visible early — this is exactly the
        #    transient signature a rolling z-score baseline keys on.
        shock_intensity = np.clip(hours_since_fault / wear["shock_ramp_hours"], 0.0, 1.0)
        # Poisson-like shocks: per-sample probability scaled to the sampling rate.
        shock_prob = wear["shock_rate_per_s"] / sampling_hz * shock_intensity
        shock_mask = rng.random(n_samples) < shock_prob
        shock_size = (
            wear["shock_amplitude"]
            * shock_intensity
            * (0.5 + rng.random(n_samples))          # ±50 % physical jitter
            * rng.choice(np.array([-1.0, 1.0]), n_samples)  # random direction
        )
        vibration += np.where(shock_mask, shock_size, 0.0)
        # 3) Linear friction heating.
        temperature += wear["temperature_rise_per_hour"] * hours_since_fault * progress
        # 4) Louder machine.
        acoustic += wear["acoustic_rise_db"] * progress

        fault_flag = (t_hours >= fault_at_hour).astype(int)

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n_samples, freq=pd.Timedelta(seconds=1 / sampling_hz)),
            "vibration_mm_s": vibration,
            "temperature_c": temperature,
            "acoustic_db": acoustic,
            "rpm": rpm,
            "fault": fault_flag,
            "fault_type": fault_label,
        }
    )
    return frame


def generate_dataset(
    n_machines: int = 3,
    duration_hours: float = 72.0,
    sampling_hz: int = 1,
    fault_at_hour: float | None = None,
    seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate a small fleet of simulated machines.

    Parameters
    ----------
    n_machines:
        How many machines to simulate.
    duration_hours:
        Runtime per machine in hours.
    sampling_hz:
        Samples per second (1 Hz keeps CSV size and runtime small).
    fault_at_hour:
        Fault onset hour shared by all machines. ``None`` keeps the whole
        fleet healthy. Defaults to 55 % of the runtime when not given.
    seed:
        Base seed; machine *i* uses ``seed + i`` so each machine differs but
        the whole dataset stays reproducible.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys ``"machine_01"`` ... (zero-padded); values are the frames from
        :func:`simulate_machine`.
    """
    if fault_at_hour is None:
        # A working default: every machine faults a bit past halfway through.
        fault_at_hour = round(duration_hours * 0.55, 2)

    machines: dict[str, pd.DataFrame] = {}
    for i in range(1, n_machines + 1):
        name = f"machine_{i:02d}"
        machines[name] = simulate_machine(
            duration_hours=duration_hours,
            sampling_hz=sampling_hz,
            fault_at_hour=fault_at_hour,
            fault_type="bearing_wear",
            seed=None if seed is None else seed + i,
        )
    return machines
