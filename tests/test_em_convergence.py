"""Accuracy-estimation workflow for electromagnetic observables."""
from __future__ import annotations

import numpy as np
import pytest

from photonix.em.convergence import adaptive_convergence, estimate_convergence


def test_observed_second_order_and_richardson_limit():
    resolutions = (10, 20, 40, 80)
    exact = 2.45
    values = [exact + 3.0 / resolution**2 for resolution in resolutions]

    result = estimate_convergence(resolutions, values, rtol=3e-4)

    assert result.observed_order == pytest.approx(2.0, rel=1e-12)
    assert result.extrapolation_order == pytest.approx(2.0, rel=1e-12)
    assert result.extrapolated.item() == pytest.approx(exact, abs=1e-14)
    assert result.correction_alignment == pytest.approx(1.0)
    assert result.asymptotic
    assert result.converged
    assert result.estimated_absolute_error == pytest.approx(1.25 * 3.0 / 80**2)


def test_unequal_refinement_ratios_recover_order():
    resolutions = (12, 19, 31)
    exact = -0.25
    values = [exact - 0.8 / resolution**1.7 for resolution in resolutions]

    result = estimate_convergence(resolutions, values)

    assert result.observed_order == pytest.approx(1.7, rel=1e-10)
    assert result.extrapolated.item() == pytest.approx(exact, abs=1e-12)
    assert result.asymptotic


def test_complex_vector_observable_uses_common_asymptotic_order():
    resolutions = (8, 16, 32)
    exact = np.asarray([2.0 + 0.5j, -1.0j])
    coefficient = np.asarray([0.4 - 0.2j, 0.1 + 0.3j])
    values = np.asarray([exact + coefficient / resolution**3 for resolution in resolutions])

    result = estimate_convergence(resolutions, values, order=3.0, atol=1e-3)

    assert np.allclose(result.extrapolated, exact, atol=1e-14)
    assert result.observed_order == pytest.approx(3.0, rel=1e-12)
    assert result.correction_alignment == pytest.approx(1.0)
    assert result.finest_value.shape == (2,)


def test_non_asymptotic_oscillation_cannot_claim_convergence():
    result = estimate_convergence(
        (10, 20, 40),
        [1.01, 0.995, 1.0025],
        order=1.0,
        atol=1.0,  # a loose tolerance must not override a failed regime check
    )

    assert result.correction_alignment == pytest.approx(-1.0)
    assert not result.asymptotic
    assert not result.converged


def test_inconsistent_formal_and_observed_order_cannot_claim_convergence():
    resolutions = (10, 20, 40)
    values = [1.0 + 1.0 / resolution for resolution in resolutions]
    result = estimate_convergence(
        resolutions,
        values,
        order=2.0,
        atol=1.0,
        order_rtol=0.2,
    )

    assert result.observed_order == pytest.approx(1.0)
    assert not result.order_consistent
    assert not result.asymptotic
    assert not result.converged


def test_adaptive_convergence_stops_when_certified():
    calls: list[int] = []

    def solver(resolution: int) -> float:
        calls.append(resolution)
        return 1.75 + 0.5 / resolution**2

    result = adaptive_convergence(
        solver,
        initial_resolution=10,
        refinement=2,
        max_levels=6,
        atol=2e-4,
    )

    assert result.converged
    assert result.resolutions == tuple(calls)
    assert result.levels == 3
    assert result.finest_value == pytest.approx(solver(40))


def test_exact_agreement_is_zero_error_convergence():
    result = estimate_convergence((10, 20, 40), [3.0, 3.0, 3.0])

    assert np.isinf(result.observed_order)
    assert result.estimated_absolute_error == 0.0
    assert result.asymptotic
    assert result.converged


def test_rejects_invalid_or_ragged_studies():
    with pytest.raises(ValueError, match="three"):
        estimate_convergence((10, 20), [1.0, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        estimate_convergence((10, 10, 20), [1.0, 1.0, 1.0])
    with pytest.raises((ValueError, TypeError)):
        estimate_convergence((10, 20, 40), [np.ones(1), np.ones(2), np.ones(1)])
    with pytest.raises(ValueError, match="fewer than three"):
        adaptive_convergence(
            lambda resolution: 1.0,
            initial_resolution=10,
            refinement=2,
            max_levels=3,
            max_resolution=15,
        )
