"""Geometry/coordinate bridge between a photonix grid and a Meep cell.

photonix lays out a 2-D in-plane device as ``eps[iy, ix]`` with **propagation
along x** (columns) and the **transverse axis y** (rows) -- the same convention as
:func:`photonix.em.fdfd.waveguide_sparams`, whose ``src_col`` / ``*_mon_col``
arguments index columns. Meep places its cell symmetrically about the origin. The
pure helpers here (:func:`cell_size`, :func:`col_to_x`, :func:`row_to_y`) are the
single source of truth for that mapping and are unit-tested without Meep;
:func:`build_block` assembles the actual ``meep.Block`` + cell.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .materials import to_material_grid

__all__ = [
    "DeviceGrid",
    "cell_size",
    "col_to_x",
    "row_to_y",
    "build_block",
    "build_pixel_block",
]


def cell_size(eps: np.ndarray, dx: float, dy: float) -> tuple[float, float]:
    """Physical ``(sx, sy)`` in um of a pixel grid with spacings ``dx, dy``."""
    arr = np.asarray(eps)
    if arr.ndim != 2 or 0 in arr.shape:
        raise ValueError("eps must be a non-empty 2-D array")
    dx, dy = float(dx), float(dy)
    if not np.isfinite(dx) or dx <= 0 or not np.isfinite(dy) or dy <= 0:
        raise ValueError("dx and dy must be positive and finite")
    ny, nx = arr.shape
    return nx * dx, ny * dy


def col_to_x(col: float, nx: int, dx: float) -> float:
    """Column index -> Meep x-coordinate (pixel-centre, cell centred on origin)."""
    if nx <= 0 or not np.isfinite(dx) or dx <= 0:
        raise ValueError("nx and dx must be positive")
    if not np.isfinite(col) or not (-0.5 <= col <= nx - 0.5):
        raise ValueError(f"col must address the grid, got {col!r} for nx={nx}")
    return -0.5 * nx * dx + (col + 0.5) * dx


def row_to_y(row: float, ny: int, dy: float) -> float:
    """Row index -> Meep y-coordinate (pixel-centre, cell centred on origin)."""
    if ny <= 0 or not np.isfinite(dy) or dy <= 0:
        raise ValueError("ny and dy must be positive")
    if not np.isfinite(row) or not (-0.5 <= row <= ny - 0.5):
        raise ValueError(f"row must address the grid, got {row!r} for ny={ny}")
    return -0.5 * ny * dy + (row + 0.5) * dy


@dataclass(frozen=True)
class DeviceGrid:
    """Immutable description of a photonix device grid in Meep coordinates.

    Holds only plain numbers/arrays (no Meep objects), so it can be built and
    inspected -- and tested -- without Meep installed.
    """

    eps: np.ndarray
    dx: float
    dy: float

    @classmethod
    def from_cross_section(cls, cross_section) -> DeviceGrid:
        """Create a grid from a native :class:`photonix.em.geometry.CrossSection`."""
        return cls(cross_section.eps, cross_section.dx, cross_section.dy)

    def __post_init__(self) -> None:
        eps = np.asarray(self.eps, dtype=float)
        cell_size(eps, self.dx, self.dy)  # validates shape and spacings
        if not np.all(np.isfinite(eps)) or np.any(eps <= 0):
            raise ValueError("eps must contain finite, positive relative permittivities")
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "dx", float(self.dx))
        object.__setattr__(self, "dy", float(self.dy))

    @property
    def shape(self) -> tuple[int, int]:
        ny, nx = self.eps.shape
        return ny, nx

    @property
    def size(self) -> tuple[float, float]:
        return cell_size(self.eps, self.dx, self.dy)

    @property
    def resolution(self) -> float:
        """Meep resolution (pixels per um) implied by the finer of dx, dy."""
        return 1.0 / min(float(self.dx), float(self.dy))

    def x_of_col(self, col: float) -> float:
        _ny, nx = self.shape
        return col_to_x(col, nx, self.dx)

    def y_of_row(self, row: float) -> float:
        ny, _nx = self.shape
        return row_to_y(row, ny, self.dy)


def build_block(device: DeviceGrid, **material_grid_kwargs):
    """Return a node-interpolated ``MaterialGrid`` block for inverse design.

    The values are interpreted as MaterialGrid *density nodes*, not as
    piecewise-constant cell-centred permittivities. Use :func:`build_pixel_block`
    for grids rasterised by Photonix's FDFD/layout code.
    """
    from ._guard import require_meep

    mp = require_meep()
    sx, sy = device.size
    grid = to_material_grid(device.eps, **material_grid_kwargs)
    block = mp.Block(size=mp.Vector3(sx, sy, 0), center=mp.Vector3(), material=grid)
    return mp.Vector3(sx, sy, 0), [block]


def build_pixel_block(device: DeviceGrid):
    """Return a piecewise-constant, cell-centred epsilon block.

    Meep's ``MaterialGrid`` bilinearly interpolates density values and therefore
    does not reproduce a cell-centred Photonix permittivity raster. A supported
    position-dependent material callback keeps each value constant over its
    ``dx`` by ``dy`` pixel instead, without creating one geometry object per cell.
    """
    from ._guard import require_meep

    mp = require_meep()
    sx, sy = device.size
    eps = device.eps
    ny, nx = device.shape

    def sampled_medium(point):
        ix = int(np.clip(np.floor((float(point.x) + 0.5 * sx) / device.dx), 0, nx - 1))
        iy = int(np.clip(np.floor((float(point.y) + 0.5 * sy) / device.dy), 0, ny - 1))
        return mp.Medium(epsilon=float(eps[iy, ix]))

    block = mp.Block(
        size=mp.Vector3(sx, sy, mp.inf),
        center=mp.Vector3(),
        material=sampled_medium,
    )
    return mp.Vector3(sx, sy, 0), [block]
