import pytest

from benchmarks.phase_suite import calibration, run
from markovq.interop import phase_experiment


def test_public_distributions():
    r = run()
    assert r["readiness"] == "READY"
    assert sum(r["pilot_shots"] + r["final_shots"]) == 10000
    assert r["total_simulated_shots"] == 39936
    assert set(r["pauli_comparisons"]) == {"IIZ", "IZI"}
    assert len(r["position_distribution_test"]) > 0


def test_invalid_calibrations():
    from dataclasses import replace

    rows, at = calibration()
    for bad in [
        [],
        [replace(r, kind="provider_estimate") for r in rows],
        rows[-1:],
        [replace(r, value=0.2) for r in rows],
        [replace(r, sample_size=None) for r in rows],
    ]:
        with pytest.raises(ValueError):
            phase_experiment(bad, at, seed=1)
