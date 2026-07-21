"""Build 2-D waveguide cross-sections (permittivity grids)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CrossSection", "rectangular_waveguide"]


@dataclass
class CrossSection:
    """A discretized waveguide cross-section.

    Attributes
    ----------
    eps : ndarray
        Relative permittivity (n^2) on the grid, shape ``(ny, nx)``.
    x, y : ndarray
        1-D coordinate arrays (µm).
    """

    eps: np.ndarray
    x: np.ndarray
    y: np.ndarray

    @property
    def n(self) -> np.ndarray:
        """Refractive index grid (sqrt of permittivity)."""
        return np.sqrt(self.eps)

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def dy(self) -> float:
        return float(self.y[1] - self.y[0])


def rectangular_waveguide(
    *,
    width: float = 0.5,
    thickness: float = 0.22,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    margin: float = 1.5,
    resolution: int = 40,
) -> CrossSection:
    """Rectangular core embedded in uniform cladding.

    Parameters
    ----------
    width, thickness
        Core dimensions in µm (x = width, y = thickness).
    n_core, n_clad
        Core and cladding refractive indices.
    margin
        Cladding margin added on every side (µm).
    resolution
        Grid points per µm.

    Returns
    -------
    CrossSection

    Examples
    --------
    >>> cs = rectangular_waveguide(width=0.5, thickness=0.22, resolution=30)
    >>> cs.eps.max() > cs.eps.min()
    True
    """
    wx = width + 2 * margin
    wy = thickness + 2 * margin
    nx = max(int(round(wx * resolution)), 8)
    ny = max(int(round(wy * resolution)), 8)
    x = np.linspace(-wx / 2, wx / 2, nx)
    y = np.linspace(-wy / 2, wy / 2, ny)
    eps = np.full((ny, nx), n_clad**2, dtype=float)
    X, Y = np.meshgrid(x, y)
    core = (np.abs(X) <= width / 2) & (np.abs(Y) <= thickness / 2)
    eps[core] = n_core**2
    return CrossSection(eps=eps, x=x, y=y)
