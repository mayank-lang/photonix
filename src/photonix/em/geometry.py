"""Cross-section geometry with subpixel permittivity averaging.

A sharp index step that does not align with the grid causes staircasing and
spoils the O(h^2) convergence of the finite-difference mode solver. Averaging the
permittivity over each grid cell by the area fraction inside the core restores
clean convergence (the standard fix used by Meep/Lumerical), letting Richardson
extrapolation reach <0.1% on the analytic slab at CPU-friendly resolutions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["CrossSection", "rectangular_waveguide", "slab_profile", "as_real_eps"]


def as_real_eps(eps, where: str = "this solver") -> np.ndarray:
    """Validate and return ``eps`` as a float array, rejecting complex input.

    The in-house eigensolvers and FDFD assume real (lossless) permittivity; a
    bare ``np.asarray(eps, float)`` would *silently truncate* an imaginary part
    (absorption/gain) and return a lossless answer. This helper turns that
    silent truncation into a loud error. Pass ``eps.real`` explicitly if
    discarding the imaginary part is intended.
    """
    eps = np.asarray(eps)
    if np.iscomplexobj(eps):
        if float(np.max(np.abs(eps.imag))) > 0.0:
            raise ValueError(
                f"{where} supports real (lossless) permittivity only; the given "
                "eps has a nonzero imaginary part, which would be silently "
                "discarded. Use eps.real explicitly if that is what you want."
            )
        eps = eps.real
    return np.asarray(eps, dtype=float)


def _validate_eps_grid(eps, grid, *, where: str):
    """Validate a uniform 2-D permittivity grid for the finite-difference solvers.

    ``grid=None`` retains the historical unit-cell spacing.  Explicit coordinate
    arrays must be finite, strictly increasing, uniformly spaced, and match the
    corresponding permittivity-grid axis; the current finite-difference
    operators do not support nonuniform meshes.
    """
    eps = as_real_eps(eps, where=where)
    if eps.ndim != 2 or min(eps.shape, default=0) < 2:
        raise ValueError("eps must be a two-dimensional grid with at least 2 cells per axis")
    if not np.all(np.isfinite(eps)):
        raise ValueError("eps must contain only finite values")

    ny, nx = eps.shape
    x: Any
    y: Any
    if grid is None:
        x = np.arange(nx, dtype=float)
        y = np.arange(ny, dtype=float)
    else:
        if not isinstance(grid, (tuple, list)) or len(grid) != 2:
            raise ValueError("grid must be a two-item (x, y) tuple")
        x = np.asarray(grid[0], dtype=float)
        y = np.asarray(grid[1], dtype=float)

    def spacing(coord, expected, name):
        if coord.ndim != 1 or len(coord) != expected:
            raise ValueError(
                f"grid {name} must be one-dimensional with length {expected}"
            )
        if not np.all(np.isfinite(coord)):
            raise ValueError(f"grid {name} must contain only finite coordinates")
        steps = np.diff(coord)
        if np.any(steps <= 0):
            raise ValueError(f"grid {name} coordinates must be strictly increasing")
        h = float(steps[0])
        if not np.allclose(steps, h, rtol=1e-9, atol=1e-12 * max(1.0, abs(h))):
            raise ValueError(
                f"grid {name} must be uniformly spaced; nonuniform grids are not supported"
            )
        return h

    dx = spacing(x, nx, "x")
    dy = spacing(y, ny, "y")
    return eps, x, y, dx, dy


@dataclass
class CrossSection:
    eps: np.ndarray   # (ny, nx) subpixel-averaged permittivity
    x: np.ndarray
    y: np.ndarray

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def dy(self) -> float:
        return float(self.y[1] - self.y[0])


def _overlap(c: np.ndarray, h: float, lo: float, hi: float) -> np.ndarray:
    """Fraction of each cell ``[c-h/2, c+h/2]`` lying within ``[lo, hi]``."""
    a = np.maximum(c - h / 2, lo)
    b = np.minimum(c + h / 2, hi)
    return np.clip(b - a, 0.0, h) / h


def rectangular_waveguide(
    *,
    width: float = 0.5,
    thickness: float = 0.22,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    margin: float = 1.5,
    resolution: int = 40,
) -> CrossSection:
    """Rectangular core in cladding with **subpixel-averaged** permittivity.

    Examples
    --------
    >>> cs = rectangular_waveguide(width=0.5, thickness=0.22, resolution=20)
    >>> bool(cs.eps.min() < cs.eps.max())
    True
    """
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be a positive finite number of points per micrometre")
    if not np.isfinite(width) or width <= 0:
        raise ValueError("width must be positive and finite")
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("thickness must be positive and finite")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("margin must be non-negative and finite")

    wx, wy = width + 2 * margin, thickness + 2 * margin
    # ``resolution`` means exactly what its public name says: samples per um.
    # The old linspace construction included both requested domain endpoints,
    # making dx = width / (n - 1) rather than 1 / resolution.  Treat x/y as
    # cell centres instead and round the cell count up so the requested margin
    # is never shortened by grid quantisation.
    h = 1.0 / float(resolution)
    nx = max(int(np.ceil(wx * resolution)), 8)
    ny = max(int(np.ceil(wy * resolution)), 8)
    x = (np.arange(nx, dtype=float) - 0.5 * (nx - 1)) * h
    y = (np.arange(ny, dtype=float) - 0.5 * (ny - 1)) * h
    dx = dy = h
    fx = _overlap(x, dx, -width / 2, width / 2)        # (nx,)
    fy = _overlap(y, dy, -thickness / 2, thickness / 2)  # (ny,)
    frac = fy[:, None] * fx[None, :]                    # area fraction in core
    eps = frac * n_core**2 + (1.0 - frac) * n_clad**2
    return CrossSection(eps=eps, x=x, y=y)


def slab_profile(
    *,
    thickness: float = 0.22,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    margin: float = 2.0,
    resolution: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """1-D subpixel-averaged slab permittivity profile ``(eps, y)``."""
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be a positive finite number of points per micrometre")
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("thickness must be positive and finite")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("margin must be non-negative and finite")

    wy = thickness + 2 * margin
    dy = 1.0 / float(resolution)
    ny = max(int(np.ceil(wy * resolution)), 8)
    y = (np.arange(ny, dtype=float) - 0.5 * (ny - 1)) * dy
    fy = _overlap(y, dy, -thickness / 2, thickness / 2)
    eps = fy * n_core**2 + (1.0 - fy) * n_clad**2
    return eps, y
