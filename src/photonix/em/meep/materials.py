"""Translate a photonix permittivity grid into Meep materials.

The bridge is a Meep :class:`~meep.MaterialGrid`: a continuous interpolation
between two end-point media (``low`` and ``high``) driven by a ``weights`` array
in ``[0, 1]``. Any photonix ``eps`` map -- a :class:`~photonix.em.geometry.CrossSection`
profile, an FDFD device grid, or an inverse-design density -- maps onto it by

    weight = (eps - eps_low) / (eps_high - eps_low)

which is exactly the linear-in-epsilon interpolation Meep's MaterialGrid performs,
so the round trip is lossless for a two-material structure and a faithful
piecewise-linear approximation otherwise. The weight computation
(:func:`material_grid_weights`) is pure NumPy and unit-tested without Meep; the
thin :func:`to_material_grid` wrapper only touches Meep to build the object.
"""
from __future__ import annotations

import numpy as np

from ._guard import require_meep

__all__ = [
    "material_grid_weights",
    "index_grid",
    "to_material_grid",
    "medium",
    "epsilon_lookup",
]


def index_grid(eps: np.ndarray) -> np.ndarray:
    """Refractive index ``sqrt(eps)`` of a (real, lossless) permittivity grid."""
    return np.sqrt(np.asarray(eps, dtype=float))


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
    eps = np.asarray(eps, dtype=float)
    lo = float(np.min(eps)) if eps_low is None else float(eps_low)
    hi = float(np.max(eps)) if eps_high is None else float(eps_high)
    if hi <= lo:
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
    eps = np.asarray(eps, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

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
    eps = float(index) ** 2 if index is not None else float(epsilon)
    return mp.Medium(epsilon=eps)


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
    function.
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
