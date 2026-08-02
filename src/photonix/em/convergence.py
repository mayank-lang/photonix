"""Grid-convergence studies with Richardson extrapolation and a GCI bound.

A field solver returning many digits is not necessarily accurate.  The
defensible accuracy check for FDE/FDFD/EME discretizations is a refinement study
performed on the observable that matters (effective index, loss, coupling, ...).
This module provides that workflow without tying it to one solver.

The estimator assumes an asymptotic error expansion ``f(h) = f(0) + C h**p``.
It infers ``p`` from the three finest grids, checks that successive corrections
are aligned, Richardson-extrapolates to ``h=0``, and applies the conventional
1.25 safety factor used for a three-grid Grid Convergence Index (GCI).  A failed
asymptotic check is reported rather than hidden behind a small last-step change.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

__all__ = ["GridConvergenceResult", "estimate_convergence", "adaptive_convergence"]


@dataclass(frozen=True)
class GridConvergenceResult:
    """Result of a grid-refinement study.

    ``values[k]`` corresponds to ``resolutions[k]``.  Only the three finest
    grids determine the reported order and error, while earlier values remain
    available for plotting and judging whether the asymptotic regime was
    reached.
    """

    resolutions: tuple[int, ...]
    values: np.ndarray
    extrapolated: np.ndarray
    observed_order: float
    extrapolation_order: float
    fine_refinement_ratio: float
    estimated_absolute_error: float
    estimated_relative_error: float
    grid_convergence_index: float
    correction_alignment: float
    order_consistent: bool
    asymptotic: bool
    converged: bool
    rtol: float
    atol: float
    safety_factor: float
    order_rtol: float

    @property
    def finest_value(self):
        """Value on the finest evaluated grid, preserving scalar/array shape."""
        value = self.values[-1]
        return value.item() if value.ndim == 0 else value

    @property
    def levels(self) -> int:
        """Number of evaluated resolution levels."""
        return len(self.resolutions)


def _positive_finite(name: str, value: float, *, allow_zero: bool = False) -> float:
    value = float(value)
    valid = np.isfinite(value) and (value >= 0.0 if allow_zero else value > 0.0)
    if not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return value


def _validate_resolutions(resolutions: Sequence[int]) -> tuple[int, ...]:
    out: list[int] = []
    for value in resolutions:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("resolutions must contain integers.")
        out.append(int(value))
    if len(out) < 3:
        raise ValueError("at least three resolution levels are required.")
    if any(value <= 0 for value in out) or any(
        b <= a for a, b in zip(out, out[1:], strict=False)
    ):
        raise ValueError("resolutions must be positive and strictly increasing.")
    return tuple(out)


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.ravel(value)))


def _observed_order(
    coarse_resolution: int,
    medium_resolution: int,
    fine_resolution: int,
    coarse_difference: float,
    fine_difference: float,
) -> float:
    """Infer a positive order for possibly unequal refinement ratios."""
    if fine_difference == 0.0:
        return float("inf")
    if coarse_difference <= fine_difference or coarse_difference == 0.0:
        return float("nan")
    difference_ratio = coarse_difference / fine_difference
    ratio_coarse = medium_resolution / coarse_resolution
    ratio_fine = fine_resolution / medium_resolution
    if np.isclose(ratio_coarse, ratio_fine, rtol=1e-10, atol=0.0):
        return float(np.log(difference_ratio) / np.log(ratio_fine))

    # With h proportional to 1/resolution, the ratio of successive errors is
    # (a**p - 1) / (1 - b**-p), where a=h1/h2 and b=h2/h3.
    log_a = np.log(ratio_coarse)
    log_b = np.log(ratio_fine)

    def residual(order: float) -> float:
        numerator = np.expm1(order * log_a)
        denominator = -np.expm1(-order * log_b)
        return float(numerator / denominator - difference_ratio)

    lower, upper = 1e-8, 20.0
    f_lower, f_upper = residual(lower), residual(upper)
    if not (np.isfinite(f_lower) and np.isfinite(f_upper)) or f_lower * f_upper > 0.0:
        return float("nan")
    return float(brentq(residual, lower, upper, xtol=1e-12, rtol=1e-12))


def estimate_convergence(
    resolutions: Sequence[int],
    values,
    *,
    order: float | None = None,
    rtol: float = 1e-3,
    atol: float = 0.0,
    safety_factor: float = 1.25,
    alignment_threshold: float = 0.9,
    order_rtol: float = 0.5,
) -> GridConvergenceResult:
    """Estimate a continuum value and conservative discretization error.

    Parameters
    ----------
    resolutions
        Increasing points-per-unit (or any quantity proportional to ``1/h``).
        At least three levels are required; unequal ratios are supported.
    values
        Scalar, complex, or fixed-shape array observable at each resolution.
    order
        Known asymptotic order used for extrapolation.  If omitted, the order is
        inferred from the norms of the last two corrections.  The inferred
        ``observed_order`` is always reported.
    rtol, atol
        A study is converged when the safety-factored absolute error is no more
        than ``atol + rtol * ||extrapolated||`` *and* the last corrections pass
        the asymptotic alignment check.
    safety_factor
        Multiplier on the Richardson error.  The default 1.25 is the standard
        three-or-more-grid GCI factor when the observed order is available.
    alignment_threshold
        Minimum real cosine similarity of the last two correction vectors.  A
        value near one is expected for a single leading ``C h**p`` error term.
    order_rtol
        When ``order`` is supplied, the observed order must agree with it to
        this relative tolerance before the study can claim convergence.

    Notes
    -----
    The result certifies only the sampled observable under the assumed error
    expansion.  Domain truncation, insufficient PML, modal-basis error, and a
    wrong physical model require independent studies.
    """
    resolutions = _validate_resolutions(resolutions)
    rtol = _positive_finite("rtol", rtol, allow_zero=True)
    atol = _positive_finite("atol", atol, allow_zero=True)
    safety_factor = _positive_finite("safety_factor", safety_factor)
    alignment_threshold = float(alignment_threshold)
    if not np.isfinite(alignment_threshold) or not -1.0 <= alignment_threshold <= 1.0:
        raise ValueError("alignment_threshold must be finite and lie in [-1, 1].")
    order_rtol = _positive_finite("order_rtol", order_rtol, allow_zero=True)
    if order is not None:
        order = _positive_finite("order", order)

    sampled = np.asarray(values)
    if sampled.dtype == object or sampled.shape[:1] != (len(resolutions),):
        raise ValueError("values must form one fixed-shape sample per resolution.")
    if not np.issubdtype(sampled.dtype, np.number):
        raise ValueError("values must be numeric.")
    sampled = np.asarray(sampled, dtype=complex if np.iscomplexobj(sampled) else float)
    if not np.all(np.isfinite(sampled)):
        raise ValueError("values must be finite.")

    coarse, medium, fine = sampled[-3:]
    delta_coarse = medium - coarse
    delta_fine = fine - medium
    norm_coarse = _norm(delta_coarse)
    norm_fine = _norm(delta_fine)

    observed_order = _observed_order(
        resolutions[-3], resolutions[-2], resolutions[-1], norm_coarse, norm_fine
    )
    used_order = float(order if order is not None else observed_order)
    fine_ratio = resolutions[-1] / resolutions[-2]

    if norm_coarse == 0.0 and norm_fine == 0.0:
        alignment = 1.0
    elif norm_coarse == 0.0 or norm_fine == 0.0:
        alignment = 1.0 if norm_fine == 0.0 else -1.0
    else:
        inner = np.vdot(np.ravel(delta_coarse), np.ravel(delta_fine))
        alignment = float(np.clip(np.real(inner) / (norm_coarse * norm_fine), -1.0, 1.0))

    order_valid = np.isfinite(used_order) and used_order > 0.0
    if norm_fine == 0.0:
        correction = np.zeros_like(fine)
        # Exact agreement on the two finest grids is a zero-error estimate even
        # when the formal observed order is infinite.
        order_valid = True
    elif order_valid:
        denominator = fine_ratio**used_order - 1.0
        correction = delta_fine / denominator
    else:
        correction = np.full_like(fine, np.nan)

    extrapolated = fine + correction
    if np.all(np.isfinite(correction)):
        estimated_absolute_error = safety_factor * _norm(correction)
        scale = _norm(extrapolated)
        if scale == 0.0:
            estimated_relative_error = 0.0 if estimated_absolute_error == 0.0 else float("inf")
        else:
            estimated_relative_error = estimated_absolute_error / scale
    else:
        estimated_absolute_error = float("inf")
        estimated_relative_error = float("inf")

    decreasing = norm_fine < norm_coarse or norm_fine == 0.0
    order_consistent = bool(
        order is None
        or norm_fine == 0.0
        or (
            np.isfinite(observed_order)
            and abs(observed_order - order) <= order_rtol * order
        )
    )
    asymptotic = bool(
        order_valid
        and order_consistent
        and decreasing
        and alignment >= alignment_threshold
    )
    tolerance = atol + rtol * _norm(extrapolated) if np.all(np.isfinite(extrapolated)) else -1.0
    converged = bool(asymptotic and estimated_absolute_error <= tolerance)

    return GridConvergenceResult(
        resolutions=resolutions,
        values=sampled,
        extrapolated=np.asarray(extrapolated),
        observed_order=float(observed_order),
        extrapolation_order=used_order,
        fine_refinement_ratio=float(fine_ratio),
        estimated_absolute_error=float(estimated_absolute_error),
        estimated_relative_error=float(estimated_relative_error),
        grid_convergence_index=float(estimated_relative_error),
        correction_alignment=alignment,
        order_consistent=order_consistent,
        asymptotic=asymptotic,
        converged=converged,
        rtol=rtol,
        atol=atol,
        safety_factor=safety_factor,
        order_rtol=order_rtol,
    )


def adaptive_convergence(
    solver: Callable[[int], object],
    *,
    initial_resolution: int,
    refinement: float = 2.0,
    max_levels: int = 5,
    max_resolution: int | None = None,
    order: float | None = None,
    rtol: float = 1e-3,
    atol: float = 0.0,
    safety_factor: float = 1.25,
    alignment_threshold: float = 0.9,
    order_rtol: float = 0.5,
) -> GridConvergenceResult:
    """Refine a solver until its observable reaches the requested accuracy.

    ``solver`` is called as ``solver(resolution)``.  For example::

        result = adaptive_convergence(
            lambda r: photonix.em.n_eff_fullvector(resolution=r),
            initial_resolution=20,
            rtol=2e-3,
        )

    At least three solves are always performed.  The returned result may have
    ``converged=False`` when ``max_levels`` or ``max_resolution`` is reached;
    callers should check that flag before using the extrapolated value.
    """
    if isinstance(initial_resolution, (bool, np.bool_)) or not isinstance(
        initial_resolution, (int, np.integer)
    ) or initial_resolution <= 0:
        raise ValueError("initial_resolution must be a positive integer.")
    refinement = _positive_finite("refinement", refinement)
    if refinement <= 1.0:
        raise ValueError("refinement must be greater than one.")
    if isinstance(max_levels, (bool, np.bool_)) or not isinstance(max_levels, (int, np.integer)) or max_levels < 3:
        raise ValueError("max_levels must be an integer of at least three.")
    if max_resolution is not None:
        if isinstance(max_resolution, (bool, np.bool_)) or not isinstance(max_resolution, (int, np.integer)):
            raise ValueError("max_resolution must be an integer when provided.")
        if max_resolution < initial_resolution:
            raise ValueError("max_resolution cannot be smaller than initial_resolution.")

    resolutions: list[int] = []
    values: list[object] = []
    resolution = int(initial_resolution)
    result: GridConvergenceResult | None = None
    for _ in range(int(max_levels)):
        if max_resolution is not None and resolution > max_resolution:
            break
        resolutions.append(resolution)
        values.append(solver(resolution))
        if len(values) >= 3:
            result = estimate_convergence(
                resolutions,
                values,
                order=order,
                rtol=rtol,
                atol=atol,
                safety_factor=safety_factor,
                alignment_threshold=alignment_threshold,
                order_rtol=order_rtol,
            )
            if result.converged:
                return result
        next_resolution = max(resolution + 1, int(np.ceil(resolution * refinement)))
        resolution = next_resolution

    if result is None:
        raise ValueError(
            "max_resolution permits fewer than three levels; increase it or "
            "lower initial_resolution/refinement."
        )
    return result
