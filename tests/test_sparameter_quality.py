"""Physicality certificates and passive projection for S-parameters."""
from __future__ import annotations

import numpy as np
import pytest

from photonix.components import directional_coupler, straight
from photonix.core.quality import analyze_sparameters, project_passive
from photonix.core.sparams import is_passive


def test_lossless_reciprocal_coupler_certificate():
    report = analyze_sparameters(directional_coupler(coupling=0.37), lossless_atol=1e-12)

    assert report.ports == ("o1", "o2", "o3", "o4")
    assert report.batch_shape == ()
    assert report.sample_count == 1
    assert report.passive
    assert report.reciprocal
    assert report.lossless
    assert report.worst_passivity_violation == pytest.approx(0.0, abs=1e-15)
    assert report.worst_reciprocity_error == pytest.approx(0.0, abs=1e-15)
    assert report.worst_unitarity_error < 1e-12


def test_lossy_waveguide_is_passive_reciprocal_but_not_lossless():
    report = analyze_sparameters(straight(length=1_000.0, loss_db_cm=10.0))

    assert report.passive
    assert report.reciprocal
    assert not report.lossless
    assert float(report.passivity_margin) > 0.0
    assert float(report.minimum_dissipation_eigenvalue) > 0.0


def test_batched_diagnostics_identify_worst_sample():
    matrices = np.asarray(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.02], [0.8, 0.0]],
        ],
        dtype=complex,
    )
    report = analyze_sparameters((matrices, {"in": 0, "out": 1}))

    assert report.batch_shape == (2,)
    assert report.singular_values.shape == (2, 2)
    assert not report.passive
    assert not report.reciprocal
    assert not report.lossless
    assert report.worst_passivity_violation == pytest.approx(0.02)
    assert report.worst_reciprocity_error == pytest.approx(0.22)


def test_passive_projection_is_exact_contraction_and_minimal_for_diagonal_case():
    matrix = np.diag([1.2, 0.7, 0.1j]).astype(complex)
    projected, ports = project_passive((matrix, {"o1": 0, "o2": 1, "o3": 2}))

    assert ports == {"o1": 0, "o2": 1, "o3": 2}
    assert np.allclose(projected, np.diag([1.0, 0.7, 0.1j]))
    assert is_passive((projected, ports), atol=1e-12)
    # Only the offending singular value moves, by precisely its excess.
    assert np.linalg.norm(projected - matrix, "fro") == pytest.approx(0.2)


def test_passive_projection_handles_batches_and_custom_strict_limit():
    matrices = np.asarray([np.eye(2), 2.0 * np.eye(2)], dtype=complex)
    projected, ports = project_passive((matrices, {"o1": 0, "o2": 1}), limit=0.9)
    report = analyze_sparameters((projected, ports))

    assert projected.shape == matrices.shape
    assert np.allclose(report.maximum_singular_value, 0.9)
    assert report.passive


@pytest.mark.parametrize("value", [-1.0, 1.01, np.inf, np.nan])
def test_passive_projection_rejects_invalid_limit(value):
    with pytest.raises(ValueError, match="limit"):
        project_passive((np.eye(1), {"o1": 0}), limit=value)


def test_analysis_rejects_nonfinite_data_and_tolerances():
    with pytest.raises(ValueError, match="finite"):
        analyze_sparameters((np.asarray([[np.nan]]), {"o1": 0}))
    with pytest.raises(ValueError, match="passivity_atol"):
        analyze_sparameters((np.eye(1), {"o1": 0}), passivity_atol=-1.0)
    with pytest.raises(ValueError, match="port_map"):
        analyze_sparameters((np.eye(1), {"o1": 1}))
