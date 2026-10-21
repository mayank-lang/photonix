"""Pattern density in sliding windows, density rules, and the optical keep-out for dummy fill.

Foundries bound the pattern density of etched and polished layers in sliding
windows (chemical-mechanical polishing and etch loading; typically 20-80 %
in 100-500 µm windows), and sparse photonic layouts often fail it.

Method
------
The layer is merged, clipped to the analysis region and fractured along a
uniform tile grid (:func:`gdstk.slice` along ``x``, then ``y``); the pieces
are binned by centroid, giving the *exact* covered area of every tile. The
region polygon is fractured the same way, so windows that leave the region
are normalised by the area they actually cover. Window sums are box sums of
a summed-area table (F. C. Crow, "Summed-area tables for texture mapping",
SIGGRAPH 1984): O(1) per window. The tile is the greatest common divisor of
window and step, so every window is an exact union of tiles.

Optical keep-out
----------------
In a cladding of index ``n_clad`` the evanescent field of a guided mode
decays as ``exp(-gamma d)`` with ``gamma = k0 sqrt(neff^2 - n_clad^2)``, and
a perturbation at distance ``d`` changes the mode by an amount proportional
to the intensity there, ``exp(-2 gamma d)``. :func:`evanescent_keepout`
returns the distance at which that intensity has fallen to ``target`` times
its value at the core edge, ``d = ln(1/target) / (2 gamma)``. For a 2-D
cross-section the far field decays as ``K0(gamma r)`` (an extra
``1/sqrt(r)``), so the 1-D estimate is conservative.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from photonix.core.backend import xp
from photonix.layout.cell import Cell
from photonix.layout.drc import DrcViolation

from ._geom import gdstk, layer_tuple

__all__ = ["DensityMap", "density_map", "check_density", "evanescent_keepout"]

_SLICE_PRECISION = 1e-9


def _pair(v: float | Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(v, (int, float, np.integer, np.floating)):
        a = (float(v), float(v))
    else:
        a = (float(v[0]), float(v[1]))
    if not all(np.isfinite(x) and x > 0 for x in a):
        raise ValueError(f"{name} must be positive and finite, got {v!r}")
    return a


def _gcd(a: float, b: float) -> float:
    fa, fb = Fraction(a).limit_denominator(10**6), Fraction(b).limit_denominator(10**6)
    if abs(float(fa) - a) > 1e-9 * a or abs(float(fb) - b) > 1e-9 * b:
        raise ValueError("window and step must be rational multiples of 1e-6 um")
    num = np.gcd(fa.numerator * fb.denominator, fb.numerator * fa.denominator)
    return float(Fraction(int(num), fa.denominator * fb.denominator))


@dataclass
class DensityMap:
    """Window densities of one layer.

    Attributes
    ----------
    x, y
        Window centres (µm), shapes ``(nx,)`` and ``(ny,)``.
    density
        Covered fraction of each window, shape ``(ny, nx)``; ``nan`` for a
        window that does not touch the region.
    window, step
        Window size and step (µm, ``(x, y)``).
    layer
        The analysed layer.
    """

    x: np.ndarray
    y: np.ndarray
    density: np.ndarray
    window: tuple[float, float]
    step: tuple[float, float]
    layer: tuple[int, int] = (1, 0)

    def window_polygon(self, iy: int, ix: int) -> np.ndarray:
        """Rectangle of window ``(iy, ix)``."""
        cx, cy = float(self.x[ix]), float(self.y[iy])
        hx, hy = 0.5 * self.window[0], 0.5 * self.window[1]
        return np.array([(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)])

    @property
    def min(self) -> float:
        """Smallest window density."""
        return float(np.nanmin(self.density))

    @property
    def max(self) -> float:
        """Largest window density."""
        return float(np.nanmax(self.density))

    def violations(self, min_density: float = 0.0, max_density: float = 1.0) -> list[DrcViolation]:
        """Windows outside ``[min_density, max_density]`` (rules ``min_density``/``max_density``)."""
        out = []
        for iy, ix in zip(*np.nonzero(self.density < min_density - 1e-12), strict=True):
            out.append(DrcViolation("min_density", self.layer, float(min_density), self.window_polygon(iy, ix)))
        for iy, ix in zip(*np.nonzero(self.density > max_density + 1e-12), strict=True):
            out.append(DrcViolation("max_density", self.layer, float(max_density), self.window_polygon(iy, ix)))
        return out


def _tile_areas(polys: list[Any], x0: float, y0: float, tx: float, ty: float, nx: int, ny: int) -> np.ndarray:
    """Exact area of ``polys`` inside every tile of an ``(ny, nx)`` grid."""
    gd = gdstk()
    out = np.zeros((ny, nx))
    if not polys:
        return out
    xs = [x0 + i * tx for i in range(1, nx)]
    ys = [y0 + j * ty for j in range(1, ny)]
    columns = gd.slice(polys, xs, "x", precision=_SLICE_PRECISION) if xs else [polys]
    for col in columns:
        if not col:
            continue
        rows = gd.slice(col, ys, "y", precision=_SLICE_PRECISION) if ys else [col]
        for row in rows:
            for p in row:
                a = p.area()
                if a <= 0:
                    continue
                pts = np.asarray(p.points)
                (bx0, by0), (bx1, by1) = pts.min(axis=0), pts.max(axis=0)
                i = int(np.floor((0.5 * (bx0 + bx1) - x0) / tx))
                j = int(np.floor((0.5 * (by0 + by1) - y0) / ty))
                if 0 <= i < nx and 0 <= j < ny:
                    out[j, i] += a
    return out


def _box_sums(a: np.ndarray, kx: int, ky: int, sx: int, sy: int, nwx: int, nwy: int) -> np.ndarray:
    sat = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    sat[1:, 1:] = a.cumsum(axis=0).cumsum(axis=1)
    jx = np.arange(nwx) * sx
    jy = np.arange(nwy) * sy
    x1, y1 = jx + kx, jy + ky
    return (sat[np.ix_(y1, x1)] - sat[np.ix_(jy, x1)] - sat[np.ix_(y1, jx)] + sat[np.ix_(jy, jx)])


def density_map(
    cell: Cell,
    layer: tuple[int, int],
    *,
    window: float | Sequence[float] = (100.0, 100.0),
    step: float | Sequence[float] | None = None,
    region: np.ndarray | None = None,
) -> DensityMap:
    """Exact pattern density of ``layer`` in sliding windows.

    Parameters
    ----------
    cell
        The layout (flattened internally).
    layer
        Layer to analyse.
    window
        Window size (µm), a scalar or ``(wx, wy)``.
    step
        Window step (default: the window, i.e. non-overlapping windows).
        Window and step must be rational multiples of 1e-6 µm.
    region
        Analysis region polygon (default: the bounding box of all of
        ``cell``'s geometry). Windows start at the region's lower-left
        corner and cover it; a window reaching past the region is
        normalised by its area inside the region.

    Returns
    -------
    DensityMap

    Examples
    --------
    >>> from photonix.layout import Cell
    >>> c = Cell("stripes")                      # 1 um lines on a 4 um pitch: density 0.25
    >>> for k in range(50):
    ...     _ = c.add_polygon([(4 * k, 0), (4 * k + 1, 0), (4 * k + 1, 200), (4 * k, 200)])
    >>> dm = density_map(c, (1, 0), window=40.0, step=20.0,
    ...                  region=[(0, 0), (200, 0), (200, 200), (0, 200)])
    >>> dm.density.shape, round(dm.min, 12), round(dm.max, 12)
    ((9, 9), 0.25, 0.25)
    """
    gd = gdstk()
    lay = layer_tuple(layer)
    wx, wy = _pair(window, "window")
    sx, sy = _pair(step if step is not None else (wx, wy), "step")
    if region is None:
        (bx0, by0), (bx1, by1) = cell.bbox()
        if bx1 <= bx0 or by1 <= by0:
            raise ValueError("cell has no geometry; pass an explicit region")
        reg = np.array([(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)], float)
    else:
        reg = np.asarray(region, float).reshape(-1, 2)
    rx0, ry0 = reg.min(axis=0)
    rx1, ry1 = reg.max(axis=0)
    tx, ty = _gcd(wx, sx), _gcd(wy, sy)
    kx, ky = int(round(wx / tx)), int(round(wy / ty))
    ix, iy = int(round(sx / tx)), int(round(sy / ty))
    nwx = max(1, int(np.ceil((rx1 - rx0 - wx) / sx - 1e-9)) + 1)
    nwy = max(1, int(np.ceil((ry1 - ry0 - wy) / sy - 1e-9)) + 1)
    ntx, nty = (nwx - 1) * ix + kx, (nwy - 1) * iy + ky
    region_poly = [gd.Polygon(reg)]
    polys = [gd.Polygon(p) for p, ly in cell.get_polygons() if layer_tuple(ly) == lay and len(p) >= 3]
    clipped = gd.boolean(polys, region_poly, "and", precision=_SLICE_PRECISION) if polys else []
    covered = _tile_areas(clipped, float(rx0), float(ry0), tx, ty, ntx, nty)
    valid = _tile_areas(region_poly, float(rx0), float(ry0), tx, ty, ntx, nty)
    num = _box_sums(covered, kx, ky, ix, iy, nwx, nwy)
    den = _box_sums(valid, kx, ky, ix, iy, nwx, nwy)
    with np.errstate(invalid="ignore", divide="ignore"):
        dens = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), np.nan)
    xc = rx0 + np.arange(nwx) * sx + 0.5 * wx
    yc = ry0 + np.arange(nwy) * sy + 0.5 * wy
    return DensityMap(np.asarray(xc, float), np.asarray(yc, float), np.clip(dens, 0.0, 1.0), (wx, wy), (sx, sy), lay)


def check_density(
    cell: Cell,
    layer: tuple[int, int],
    *,
    min_density: float = 0.0,
    max_density: float = 1.0,
    window: float | Sequence[float] = (100.0, 100.0),
    step: float | Sequence[float] | None = None,
    region: np.ndarray | None = None,
) -> list[DrcViolation]:
    """Windows of ``layer`` whose density lies outside ``[min_density, max_density]``.

    Examples
    --------
    >>> from photonix.layout import components as lc
    >>> v = check_density(lc.straight(100.0, width=0.5), (1, 0), min_density=0.2, window=50.0,
    ...                   region=[(0, -50), (100, -50), (100, 50), (0, 50)])
    >>> len(v), v[0].rule
    (4, 'min_density')
    """
    if not 0.0 <= min_density <= max_density <= 1.0:
        raise ValueError("need 0 <= min_density <= max_density <= 1")
    return density_map(cell, layer, window=window, step=step, region=region).violations(min_density, max_density)


def evanescent_keepout(*, neff: Any, n_clad: Any, wl: Any = 1.55, target: float = 1e-4) -> Any:
    """Distance (µm) beyond which dummy fill does not perturb a guided mode by more than ``target``.

    ``d = ln(1/target) / (2 gamma)``, ``gamma = (2 pi / wl) sqrt(neff^2 - n_clad^2)``:
    the evanescent intensity at ``d`` from the core edge is ``target`` times
    its edge value (module notes). Differentiable in ``neff``, ``n_clad`` and
    ``wl`` with the JAX backend. ``neff`` typically comes from
    :func:`photonix.xsection.n_eff` of the PDK cross-section.

    Examples
    --------
    >>> round(float(evanescent_keepout(neff=2.44, n_clad=1.444, wl=1.55, target=1e-4)), 4)
    0.5776
    """
    if not 0.0 < float(target) < 1.0:
        raise ValueError("target must lie in (0, 1)")
    try:
        if float(neff) <= float(n_clad):
            raise ValueError(f"neff ({float(neff)}) must exceed n_clad ({float(n_clad)}) for a guided mode")
    except TypeError:  # pragma: no cover - traced values are not checked
        pass
    gamma = (2.0 * np.pi / wl) * xp.sqrt(neff * neff - n_clad * n_clad)
    return float(np.log(1.0 / float(target))) / (2.0 * gamma)
