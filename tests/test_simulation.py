"""Tests for the Phase 1 sensor simulator.

These tests are behavioural rather than implementation-bound: they assert
the properties the *physics* promises (reproducibility, schema, fault
flagging, vibration growth), so a rewrite of the generator internals is
free as long as the contract holds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maschinenwaechter.simulation import generate_dataset, simulate_machine

EXPECTED_COLUMNS = {
    "timestamp",
    "vibration_mm_s",
    "temperature_c",
    "acoustic_db",
    "rpm",
    "fault",
    "fault_type",
}


def test_schema_columns_present(faulty_df):
    """The frame must carry every column of the documented schema."""
    assert EXPECTED_COLUMNS.issubset(set(faulty_df.columns))


def test_reproducibility_same_seed_identical():
    """Same seed -> bit-for-bit identical values (numpy Generator contract)."""
    df_a = simulate_machine(duration_hours=0.5, sampling_hz=10, fault_at_hour=0.25, seed=7)
    df_b = simulate_machine(duration_hours=0.5, sampling_hz=10, fault_at_hour=0.25, seed=7)
    pd_testing = pytest.importorskip("pandas.testing")
    pd_testing.assert_frame_equal(df_a, df_b)


def test_different_seed_differs():
    """Sanity check: seeds actually change the random stream."""
    df_a = simulate_machine(duration_hours=0.5, sampling_hz=10, seed=1)
    df_b = simulate_machine(duration_hours=0.5, sampling_hz=10, seed=2)
    assert not np.allclose(df_a["vibration_mm_s"].values, df_b["vibration_mm_s"].values)


def test_fault_column_is_binary_and_timed(faulty_df):
    """``fault`` must be 0/1 ints, all 0 before onset and all 1 after."""
    assert set(faulty_df["fault"].unique()).issubset({0, 1})
    assert faulty_df["fault"].dtype.kind in "iu"

    onset_mask = faulty_df["timestamp"] >= faulty_df.loc[
        faulty_df["fault"] == 1, "timestamp"
    ].min()
    assert (faulty_df.loc[~onset_mask, "fault"] == 0).all()
    assert (faulty_df.loc[onset_mask, "fault"] == 1).all()


def test_timestamps_monotonic_increasing(faulty_df):
    """Timestamps must never repeat or go backwards."""
    deltas = faulty_df["timestamp"].diff().dropna()
    assert (deltas > pd.Timedelta(0)).all()


def test_vibration_std_grows_after_fault():
    """Post-fault vibration std must exceed the pre-fault std by 2x.

    This is the canary for generator regressions: if a refactor breaks the
    wear model (or the fault flag), the post-fault segment stops growing.
    Exponential wear needs a few hours to reach 2x, so this test uses a
    dedicated long-horizon run (6 h @ 1 Hz) rather than the fast fixtures.
    """
    df = simulate_machine(
        duration_hours=6.0,
        sampling_hz=1,
        fault_at_hour=1.0,
        fault_type="bearing_wear",
        seed=99,
    )
    pre = df.loc[df["fault"] == 0, "vibration_mm_s"]
    post = df.loc[df["fault"] == 1, "vibration_mm_s"]
    assert post.std() > 2.0 * pre.std()


def test_temperature_rises_after_fault(faulty_df):
    """Friction heating: post-fault mean temperature must be clearly higher."""
    pre = faulty_df.loc[faulty_df["fault"] == 0, "temperature_c"].mean()
    post = faulty_df.loc[faulty_df["fault"] == 1, "temperature_c"].mean()
    assert post > pre + 0.5


def test_fault_type_column_labelled(faulty_df):
    """A run with an injected fault labels its rows; a healthy run says 'none'."""
    assert (faulty_df["fault_type"] == "bearing_wear").all()


def test_healthy_run_has_no_faults(healthy_df):
    """Without ``fault_at_hour`` the machine must stay clean end to end."""
    assert (healthy_df["fault"] == 0).all()
    assert (healthy_df["fault_type"] == "none").all()


def test_generate_dataset_returns_named_machines():
    """generate_dataset must return a reproducible dict of named frames."""
    fleet_a = generate_dataset(n_machines=3, duration_hours=0.25, sampling_hz=10, seed=5)
    fleet_b = generate_dataset(n_machines=3, duration_hours=0.25, sampling_hz=10, seed=5)

    assert list(fleet_a.keys()) == ["machine_01", "machine_02", "machine_03"]
    for name in fleet_a:
        pd_testing = pytest.importorskip("pandas.testing")
        pd_testing.assert_frame_equal(fleet_a[name], fleet_b[name])
