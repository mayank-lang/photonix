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

__all__ = ["DeviceGrid", "cell_size", "col_to_x", "row_to_y", "build_block"]


def cell_size(eps: np.ndarray, dx: float, dy: float) -> tuple[float, float]:
    """Physical ``(sx, sy)`` in um of a pixel grid with spacings ``dx, dy``."""
    ny, nx = np.asarray(eps).shape
    return nx * float(dx), ny * float(dy)


def col_to_x(col: float, nx: int, dx: float) -> float:
    """Column index -> Meep x-coordinate (pixel-centre, cell centred on origin)."""
    return -0.5 * nx * dx + (col + 0.5) * dx


def row_to_y(row: float, ny: int, dy: float) -> float:
    """Row index -> Meep y-coordinate (pixel-centre, cell centred on origin)."""
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
        ny, nx = self.shape
        return col_to_x(col, nx, self.dx)

    def y_of_row(self, row: float) -> float:
        ny, nx = self.shape
        return row_to_y(row, ny, self.dy)


def build_block(device: DeviceGrid, **material_grid_kwargs):
    """Return ``(cell_size_vec, [block])`` for a Meep ``Simulation``.

    The single ``meep.Block`` spans the whole cell and carries a
    :class:`meep.MaterialGrid` interpolation of ``device.eps``.
    """
    from ._guard import require_meep

    mp = require_meep()
    sx, sy = device.size
    grid = to_material_grid(device.eps, **material_grid_kwargs)
    block = mp.Block(size=mp.Vector3(sx, sy, 0), center=mp.Vector3(), material=grid)
    return mp.Vector3(sx, sy, 0), [block]
