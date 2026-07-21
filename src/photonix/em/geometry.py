"""Cross-section geometry with subpixel permittivity averaging.

A sharp index step that does not align with the grid causes staircasing and
spoils the O(h^2) convergence of the finite-difference mode solver. Averaging the
permittivity over each grid cell by the area fraction inside the core restores
clean convergence (the standard fix used by Meep/Lumerical), letting Richardson
extrapolation reach <0.1% on the analytic slab at CPU-friendly resolutions.
"""
from __future__ import annotations

from dataclasses import dataclass

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
    wx, wy = width + 2 * margin, thickness + 2 * margin
    nx = max(int(round(wx * resolution)), 8)
    ny = max(int(round(wy * resolution)), 8)
    x = np.linspace(-wx / 2, wx / 2, nx)
    y = np.linspace(-wy / 2, wy / 2, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
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
    wy = thickness + 2 * margin
    ny = max(int(round(wy * resolution)), 8)
    y = np.linspace(-wy / 2, wy / 2, ny)
    dy = y[1] - y[0]
    fy = _overlap(y, dy, -thickness / 2, thickness / 2)
    eps = fy * n_core**2 + (1.0 - fy) * n_clad**2
    return eps, y
