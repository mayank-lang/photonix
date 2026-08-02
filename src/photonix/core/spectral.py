"""Accurate derivatives and delay metrics for sampled optical responses.

Wavelength sweeps are generally nonuniform in angular frequency, and optical
phase commonly wraps many times across a sweep.  Applying ``diff`` directly to
either wavelength or principal phase therefore gives a biased (and often
wrong-sign) group delay.  This module supplies the small numerical kernel used
by :class:`~photonix.core.dataset.SParameterDataset` for phase-aware spectral
analysis.

The derivative uses locally scaled polynomial weights.  Scaling every stencil
about its evaluation point avoids subtracting powers of optical frequencies of
order ``1e15 rad/s`` and supports genuinely nonuniform grids.  A five-point
stencil is fourth-order accurate on a uniform grid while retaining one-sided
high-order formulas at the sweep boundaries.
"""
from __future__ import annotations

import math

import numpy as np

from .constants import C0_UM_S

__all__ = ["differentiate_samples", "group_delay", "group_delay_dispersion"]


def _axis_index(axis: int, ndim: int) -> int:
    if not isinstance(axis, int):
        raise TypeError("axis must be an integer")
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for an array of dimension {ndim}")
    return normalized


def _validate_coordinates(coordinates) -> np.ndarray:
    x = np.asarray(coordinates, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("coordinates must be a one-dimensional array with at least two samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("coordinates must contain only finite values")
    steps = np.diff(x)
    if not (np.all(steps > 0.0) or np.all(steps < 0.0)):
        raise ValueError("coordinates must be strictly monotonic")
    return x


def _local_weights(x: np.ndarray, center: int, derivative: int, width: int) -> tuple[slice, np.ndarray]:
    """Return scaled polynomial finite-difference weights at ``x[center]``."""
    start = min(max(center - width // 2, 0), x.size - width)
    selection = slice(start, start + width)
    offsets = x[selection] - x[center]
    scale = float(np.max(np.abs(offsets)))
    if scale == 0.0:  # Strict monotonicity makes this unreachable; keep the failure explicit.
        raise ValueError("coordinate stencil has zero extent")
    z = offsets / scale

    # Sum_j w_j z_j**p = d!/dz**d z**p|_0.  Solving after local scaling is
    # substantially better conditioned than using raw optical frequencies.
    powers = np.arange(width)[:, None]
    vandermonde = z[None, :] ** powers
    rhs = np.zeros(width)
    rhs[derivative] = math.factorial(derivative)
    condition = float(np.linalg.cond(vandermonde))
    if not np.isfinite(condition) or condition > 1.0e12:
        raise ValueError(
            "coordinate spacing is too ill-conditioned for a stable local derivative; "
            "remove nearly coincident samples or reduce stencil"
        )
    weights = np.linalg.solve(vandermonde, rhs) / scale**derivative
    return selection, weights


def differentiate_samples(
    coordinates,
    values,
    *,
    derivative: int = 1,
    axis: int = 0,
    stencil: int = 5,
) -> np.ndarray:
    """Differentiate sampled data on a strictly monotonic, nonuniform grid.

    Parameters
    ----------
    coordinates
        One-dimensional sample coordinates. Increasing and decreasing grids
        are both accepted.
    values
        Real or complex samples. ``values.shape[axis]`` must match the number
        of coordinates; all other dimensions are vectorized.
    derivative
        Positive derivative order. First and second derivatives are the usual
        use cases.
    axis
        Sample axis in ``values``.
    stencil
        Maximum number of neighboring samples in each local polynomial fit.
        Five gives fourth-order first derivatives on uniform interior points.

    Returns
    -------
    numpy.ndarray
        Derivative at every original coordinate, with the same shape as
        ``values``. Boundary samples use a one-sided stencil of the same width.

    Notes
    -----
    A polynomial of degree less than the effective stencil width is
    differentiated to floating-point precision, even on an irregular grid.
    This makes the routine especially useful for angular-frequency derivatives
    derived from wavelength-uniform sweeps.
    """
    x = _validate_coordinates(coordinates)
    y = np.asarray(values)
    if y.ndim == 0:
        raise ValueError("values must have at least one dimension")
    sample_axis = _axis_index(axis, y.ndim)
    if y.shape[sample_axis] != x.size:
        raise ValueError(
            f"values axis {sample_axis} has length {y.shape[sample_axis]}, expected {x.size}"
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("values must contain only finite samples")
    if not isinstance(derivative, int) or derivative < 1:
        raise ValueError("derivative must be a positive integer")
    if not isinstance(stencil, int) or stencil < 2:
        raise ValueError("stencil must be an integer of at least 2")
    width = min(stencil, x.size)
    if width <= derivative:
        raise ValueError(
            f"derivative order {derivative} requires at least {derivative + 1} stencil samples"
        )

    moved = np.moveaxis(y, sample_axis, 0)
    dtype = np.result_type(moved.dtype, np.float64)
    result = np.empty(moved.shape, dtype=dtype)
    for index in range(x.size):
        selection, weights = _local_weights(x, index, derivative, width)
        # The requested derivative annihilates lower-order polynomials. Remove
        # the dominant constant and (for order >= 2) linear terms before the
        # weighted sum so a tiny curvature is not obtained by cancelling
        # O(phase) numbers. This matters for GDD: optical phase may span
        # thousands of radians while its second derivative is close to zero.
        local = moved[selection] - moved[index]
        if derivative >= 2:
            start, stop = selection.start, selection.stop
            slope = (moved[stop - 1] - moved[start]) / (x[stop - 1] - x[start])
            offsets = x[selection] - x[index]
            reshape = (offsets.size,) + (1,) * (moved.ndim - 1)
            local = local - offsets.reshape(reshape) * slope
        result[index] = np.tensordot(weights, local, axes=(0, 0))
    return np.moveaxis(result, 0, sample_axis)


def _unwrapped_phase(
    wavelengths,
    transfer,
    *,
    axis: int,
    magnitude_floor: float,
    phase_step_limit: float | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    wl = _validate_coordinates(wavelengths)
    if np.any(wl <= 0.0):
        raise ValueError("wavelengths must be positive")
    response = np.asarray(transfer, dtype=complex)
    if response.ndim == 0:
        raise ValueError("transfer must have at least one dimension")
    sample_axis = _axis_index(axis, response.ndim)
    if response.shape[sample_axis] != wl.size:
        raise ValueError(
            f"transfer axis {sample_axis} has length {response.shape[sample_axis]}, expected {wl.size}"
        )
    if not np.all(np.isfinite(response)):
        raise ValueError("transfer must contain only finite samples")
    if not np.isfinite(magnitude_floor) or magnitude_floor < 0.0:
        raise ValueError("magnitude_floor must be a finite nonnegative relative threshold")

    magnitude = np.abs(response)
    peak = np.max(magnitude, axis=sample_axis, keepdims=True)
    invalid = magnitude <= magnitude_floor * peak
    # A zero response has no defined phase even when the requested relative
    # floor is zero.  Keep it out of angle/unwrap rather than returning a
    # plausible-looking but physically meaningless delay.
    invalid |= magnitude == 0.0
    if np.any(invalid):
        count = int(np.count_nonzero(invalid))
        raise ValueError(
            f"group-delay phase is undefined at {count} sample(s) at or below the magnitude floor"
        )

    phase = np.unwrap(np.angle(response), axis=sample_axis)
    if phase_step_limit is not None:
        limit = float(phase_step_limit)
        if not np.isfinite(limit) or limit <= 0.0 or limit > np.pi:
            raise ValueError("phase_step_limit must be in (0, pi] or None")
        largest_step = float(np.max(np.abs(np.diff(phase, axis=sample_axis))))
        if largest_step > limit:
            raise ValueError(
                f"adjacent phase step {largest_step:.6g} rad exceeds phase_step_limit={limit:.6g}; "
                "refine the wavelength grid to avoid delay aliasing"
            )
    omega = 2.0 * np.pi * C0_UM_S / wl
    return omega, phase, sample_axis


def group_delay(
    wavelengths,
    transfer,
    *,
    axis: int = 0,
    stencil: int = 5,
    magnitude_floor: float = 1.0e-12,
    phase_step_limit: float | None = 0.9 * np.pi,
) -> np.ndarray:
    """Return group delay ``-d arg(transfer) / d omega`` in seconds.

    Photonix propagation uses ``exp(-1j * beta * length)``.  The leading minus
    sign therefore makes an ordinary forward waveguide's delay positive. Phase
    is unwrapped before a high-order derivative on the *angular-frequency*
    grid, which is generally nonuniform for wavelength-sampled data.

    ``magnitude_floor`` is relative to each response trace's peak magnitude.
    Phase at or below that floor is rejected because delay at a transmission
    zero is undefined. ``phase_step_limit`` catches samples approaching the
    phase-unwrapping Nyquist limit; set it to ``None`` only when continuity is
    independently known.
    """
    omega, phase, sample_axis = _unwrapped_phase(
        wavelengths,
        transfer,
        axis=axis,
        magnitude_floor=magnitude_floor,
        phase_step_limit=phase_step_limit,
    )
    return -differentiate_samples(omega, phase, axis=sample_axis, derivative=1, stencil=stencil)


def group_delay_dispersion(
    wavelengths,
    transfer,
    *,
    axis: int = 0,
    stencil: int = 5,
    magnitude_floor: float = 1.0e-12,
    phase_step_limit: float | None = 0.9 * np.pi,
) -> np.ndarray:
    """Return group-delay dispersion ``-d2 arg(transfer)/d omega2`` in s2.

    This is the derivative of :func:`group_delay` with respect to angular
    frequency.  It is evaluated directly as a second phase derivative, avoiding
    the extra noise and boundary error from differentiating an already
    differentiated delay trace.
    """
    omega, phase, sample_axis = _unwrapped_phase(
        wavelengths,
        transfer,
        axis=axis,
        magnitude_floor=magnitude_floor,
        phase_step_limit=phase_step_limit,
    )
    return -differentiate_samples(omega, phase, axis=sample_axis, derivative=2, stencil=stencil)
