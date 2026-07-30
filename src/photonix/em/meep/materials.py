"""Translate a photonix permittivity grid into Meep materials.

The bridge is a Meep :class:`~meep.MaterialGrid`: a continuous interpolation
between two end-point media (``low`` and ``high``) driven by a ``weights`` array
in ``[0, 1]``. Any photonix ``eps`` map -- a :class:`~photonix.em.geometry.CrossSection`
profile, an FDFD device grid, or an inverse-design density -- maps onto it by

    weight = (eps - eps_low) / (eps_high - eps_low)

which is exactly the linear-in-epsilon interpolation Meep's MaterialGrid performs.
These values are **density-grid nodes**, however, not cell-centred pixels; use
:func:`photonix.em.meep.geometry.build_pixel_block` when translating a physical
FDFD/layout raster. The weight computation
(:func:`material_grid_weights`) is pure NumPy and unit-tested without Meep; the
thin :func:`to_material_grid` wrapper only touches Meep to build the object.
"""
from __future__ import annotations

from typing import cast

import numpy as np

from ._guard import require_meep

__all__ = [
    "material_grid_weights",
    "index_grid",
    "to_material_grid",
    "medium",
    "to_medium",
    "epsilon_lookup",
]


def index_grid(eps: np.ndarray) -> np.ndarray:
    """Refractive index ``sqrt(eps)`` of a (real, lossless) permittivity grid."""
    arr = _validate_eps(eps)
    return np.sqrt(arr)


def _validate_eps(eps: np.ndarray) -> np.ndarray:
    arr = np.asarray(eps, dtype=float)
    if arr.ndim == 0 or arr.size == 0:
        raise ValueError("eps must be a non-empty array")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError("eps must contain finite, positive relative permittivities")
    return arr


def material_grid_weights(
    eps: np.ndarray,
    *,
    eps_low: float | None = None,
    eps_high: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Map a permittivity grid to MaterialGrid ``(weights, eps_low, eps_high)``.

    ``weights`` is ``(eps - eps_low) / (eps_high - eps_low)`` clipped to ``[0, 1]``.
    When the grid is uniform (or ``eps_low == eps_high``) every weight is ``0`` and
    both end-point permittivities collapse to that single value, which Meep handles
    as a plain homogeneous medium.

    Examples
    --------
    >>> import numpy as np
    >>> eps = np.array([[2.0, 12.0], [12.0, 2.0]])
    >>> w, lo, hi = material_grid_weights(eps)
    >>> (lo, hi)
    (2.0, 12.0)
    >>> w.tolist()
    [[0.0, 1.0], [1.0, 0.0]]
    """
    eps = _validate_eps(eps)
    lo = float(np.min(eps)) if eps_low is None else float(eps_low)
    hi = float(np.max(eps)) if eps_high is None else float(eps_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0 or hi <= 0:
        raise ValueError("eps_low and eps_high must be positive and finite")
    if hi < lo:
        raise ValueError("eps_high must be greater than or equal to eps_low")
    if hi == lo:
        if not np.allclose(eps, lo):
            raise ValueError("equal endpoints can only represent a uniform permittivity grid")
        return np.zeros_like(eps), lo, lo
    weights = np.clip((eps - lo) / (hi - lo), 0.0, 1.0)
    return weights, lo, hi


def epsilon_lookup(eps: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Return a pure closure ``g(px, py) -> eps`` over a sampled grid.

    Nearest-cell lookup with edge clamping. ``x`` is the column (fast) axis and
    ``y`` the row (slow) axis, matching :class:`~photonix.em.geometry.CrossSection`
    (``eps`` has shape ``(len(y), len(x))``). This is the basis of a Meep
    ``material_function`` for arbitrary, non-two-material profiles; kept Meep-free
    so it is unit-testable.
    """
    eps = _validate_eps(eps)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or eps.shape != (y.size, x.size):
        raise ValueError("eps shape must be (len(y), len(x)) for one-dimensional x and y")
    if x.size == 0 or y.size == 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must be non-empty and finite")

    def g(px: float, py: float) -> float:
        # nearest index; no assumption of uniform spacing
        ix = int(np.argmin(np.abs(x - px)))
        iy = int(np.argmin(np.abs(y - py)))
        return float(eps[iy, ix])

    return g


def medium(*, index: float | None = None, epsilon: float | None = None):
    """Build a non-dispersive :class:`meep.Medium` from an index or permittivity."""
    mp = require_meep()
    if (index is None) == (epsilon is None):
        raise ValueError("pass exactly one of `index` or `epsilon`")
    eps = float(index) ** 2 if index is not None else float(cast(float, epsilon))
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("material permittivity must be positive and finite")
    return mp.Medium(epsilon=eps)


def to_medium(material, *, wl: float):
    """Freeze a native dispersive material to a Meep ``Medium`` at one wavelength.

    Photonix :class:`~photonix.em.materials.Material` objects are wavelength-to-
    index callables, not causal susceptibility fits. This conversion is therefore
    intentionally single-frequency and non-dispersive; use an explicit Meep
    ``Medium`` with Lorentz/Drude susceptibilities for broadband dispersive FDTD.
    """
    if not np.isfinite(wl) or wl <= 0:
        raise ValueError("wl must be positive and finite")
    if hasattr(material, "index"):
        value = material.index(wl)
    elif callable(material):
        value = material(wl)
    else:
        raise TypeError("material must be a Photonix Material or wavelength-to-index callable")
    arr = np.asarray(value)
    if arr.ndim != 0 or np.iscomplexobj(arr):
        raise ValueError("material must return one real scalar index at the requested wavelength")
    return medium(index=float(arr))


def to_material_grid(
    eps: np.ndarray,
    *,
    eps_low: float | None = None,
    eps_high: float | None = None,
    do_averaging: bool = True,
):
    """Build a :class:`meep.MaterialGrid` from a photonix ``eps`` grid.

    The returned object can be dropped into a ``meep.Block`` for **FDTD**
    (:mod:`meep`'s engine fully supports ``MaterialGrid``). It must **not** be
    used as MPB / ``mpb.ModeSolver`` geometry: Meep's mode solver (``libpympb``)
    has no ``MATERIAL_GRID`` material branch and aborts on it -- the MPB path in
    :mod:`.modes` passes a raw epsilon array instead. Meep indexes the weight
    array ``[ix, iy]`` whereas photonix stores ``eps[iy, ix]``, so the weights
    are transposed here -- the one orientation subtlety, localised to this
    function. MaterialGrid weights are bilinearly interpolated density *nodes*;
    they must not be described as piecewise-constant cell-centred epsilon samples.
    """
    mp = require_meep()
    weights, lo, hi = material_grid_weights(eps, eps_low=eps_low, eps_high=eps_high)
    ny, nx = weights.shape
    grid_size = mp.Vector3(nx, ny, 0)
    return mp.MaterialGrid(
        grid_size,
        mp.Medium(epsilon=lo),
        mp.Medium(epsilon=hi),
        weights=np.ascontiguousarray(weights.T),
        do_averaging=do_averaging,
        grid_type="U_MEAN",
    )
