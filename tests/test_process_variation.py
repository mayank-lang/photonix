"""Process-corner and Monte Carlo contracts contain no foundry assumptions."""
from __future__ import annotations

import numpy as np
import pytest

from photonix.pdk import MonteCarloSpec, ProcessCorner, ProcessStudy


def test_process_study_combines_named_corners_and_reproducible_samples():
    nominal = ProcessCorner("nominal", {"width_um": 0.5, "thickness_um": 0.22})
    corners = (
        ProcessCorner("narrow", {"width_um": 0.48, "thickness_um": 0.22}),
        ProcessCorner("wide", {"width_um": 0.52, "thickness_um": 0.22}),
    )
    covariance = np.array([[4e-4, 5e-5], [5e-5, 1e-4]])
    monte_carlo = MonteCarloSpec(
        ("width_um", "thickness_um"), np.array([0.5, 0.22]), covariance
    )
    study = ProcessStudy(
        nominal, corners, monte_carlo, units={"width_um": "um", "thickness_um": "um"}
    )

    cases_a = study.cases(monte_carlo_samples=3, seed=17)
    cases_b = study.cases(monte_carlo_samples=3, seed=17)
    assert [case.name for case in cases_a] == [
        "nominal", "narrow", "wide", "mc_000000", "mc_000001", "mc_000002"
    ]
    assert [case.parameters for case in cases_a] == [case.parameters for case in cases_b]
    assert cases_a[-1].metadata == {"kind": "monte_carlo", "sample_index": 2, "seed": 17}


def test_independent_monte_carlo_uses_named_nominal_and_sigma_values():
    spec = MonteCarloSpec.independent({"a": 1.0, "b": 2.0}, {"a": 0.1, "b": 0.0})
    samples = spec.sample(4, seed=3)
    assert all(sample.parameters["b"] == 2.0 for sample in samples)
    assert not np.shares_memory(spec.mean, np.asarray(samples[0].parameters))


@pytest.mark.parametrize(
    "covariance, match",
    [
        (np.array([[1.0, 0.2], [0.1, 1.0]]), "symmetric"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
    ],
)
def test_monte_carlo_rejects_invalid_covariance(covariance, match):
    with pytest.raises(ValueError, match=match):
        MonteCarloSpec(("a", "b"), np.zeros(2), covariance)


def test_process_study_rejects_parameter_mismatch():
    nominal = ProcessCorner("nominal", {"width": 0.5})
    corner = ProcessCorner("bad", {"thickness": 0.22})
    with pytest.raises(ValueError, match="match nominal"):
        ProcessStudy(nominal, (corner,), units={"width": "um"})


def test_process_study_requires_explicit_units_for_every_parameter():
    nominal = ProcessCorner("nominal", {"width": 0.5, "dimensionless_bias": 0.0})
    with pytest.raises(ValueError, match="exactly one"):
        ProcessStudy(nominal, units={"width": "um"})
    study = ProcessStudy(nominal, units={"width": "um", "dimensionless_bias": "1"})
    assert study.units["dimensionless_bias"] == "1"
    with pytest.raises(ValueError, match="non-empty"):
        ProcessStudy(nominal, units={"width": "um ", "dimensionless_bias": "1"})


def test_process_study_evaluate_and_pdk_registry_preserve_case_names():
    from photonix.pdk import Pdk

    nominal = ProcessCorner("nominal", {"width": 0.5})
    corner = ProcessCorner("wide", {"width": 0.52})
    study = ProcessStudy(nominal, (corner,), units={"width": "um"})
    results = study.evaluate(lambda case: 2.0 * case.parameters["width"])
    assert results == {"nominal": 1.0, "wide": 1.04}
    assert study.map(lambda case: case.name) == {"nominal": "nominal", "wide": "wide"}

    pdk = Pdk("demo").add_process_study("linewidth", study)
    assert pdk.get_process_study("linewidth") is study


def test_process_values_and_metadata_are_validated():
    with pytest.raises(ValueError, match="finite"):
        ProcessCorner("bad", {"width": np.nan})
    with pytest.raises(ValueError, match="JSON"):
        ProcessCorner("bad", {"width": 0.5}, {"object": object()})
