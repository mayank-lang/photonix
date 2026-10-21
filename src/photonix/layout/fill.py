"""Dummy fill: square tiles on a lattice, outside per-layer keep-out zones, to a target density.

:func:`fill` returns a cell of tile references to place over the layout, so
the drawn design is not modified. A tile site is allowed when the whole tile
lies inside the fill region and outside every keep-out zone (each listed
layer dilated by its keep-out distance with a Euclidean disc). For optical
layers the keep-out should be at least
:func:`photonix.verification.evanescent_keepout` of the guided mode, so the
tiles do not perturb it.

Method
------
The forbidden set is the union of the round-joined dilations (the polygonal
arcs are pushed outward by ``1/cos(pi/N)`` so that no tile ever comes closer
than the requested distance) plus the outside of the region; the allowed set
``A`` is its complement. Lattice sites (lower-left tile corners on a pitch of
``tile_size + tile_space``, odd rows shifted by half a pitch when
``stagger``) are classified in bulk: a site whose tile centre lies in ``A``
eroded by the tile's circumradius certainly fits, one outside ``A`` eroded
mitre-wise by half a tile certainly does not, and only the thin band in
between is tested exactly with a boolean ``tile NOT A``.

With ``target_density`` the tiles are added window by window (non-overlapping
``window`` squares, largest deficit first, sites in a spatially spread
bit-reversed order) until each window reaches the target, a greedy
heuristic for the min-variation fill problem (A. B. Kahng et al., IEEE
Trans. CAD 18(4), 445, 1999). Windows that cannot reach it are returned as
``residual`` violations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cell import Cell
from .drc import DrcViolation
from .gds import _gdstk

__all__ = ["FillResult", "fill"]

_P = 1e-4
_ROUND = 256


@dataclass
class FillResult:
    """Output of :func:`fill`.

    Attributes
    ----------
    cell
        Cell of tile references (in the filled cell's frame): add it with
        ``top.add_ref(result.cell)``.
    sites
        Lower-left corners of the placed tiles, shape ``(n, 2)``.
    tile_size
        Tile side (µm).
    layer
        Fill layer.
    density_before, density_after
        :class:`~photonix.verification.DensityMap` of the fill layer in the
        target windows (``None`` without ``target_density``).
    residual
        Windows still below ``target_density`` after filling.
    """

    cell: Cell
    sites: np.ndarray
    tile_size: float
    layer: tuple[int, int]
    density_before: Any = None
    density_after: Any = None
    residual: list[DrcViolation] = field(default_factory=list)

    @property
    def n_tiles(self) -> int:
        """Number of placed tiles."""
        return int(len(self.sites))


def _layer(layer: Sequence[int]) -> tuple[int, int]:
    return (int(layer[0]), int(layer[1]))


def _bitrev(i: np.ndarray, bits: int = 16) -> np.ndarray:
    out = np.zeros_like(i)
    v = i.copy()
    for _ in range(bits):
        out = (out << 1) | (v & 1)
        v >>= 1
    return out


def _spread_order(ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
    """Order of lattice indices that spreads consecutive picks over the window."""
    a, b = _bitrev(ii.astype(np.int64)), _bitrev(jj.astype(np.int64))
    key = np.zeros_like(a)
    for k in range(16):
        key |= ((a >> k) & 1) << (2 * k + 1)
        key |= ((b >> k) & 1) << (2 * k)
    return np.argsort(-key, kind="stable")


def fill(
    cell: Cell,
    *,
    layer: Sequence[int],
    region: np.ndarray | Sequence[Sequence[float]] | None = None,
    tile_size: float = 2.0,
    tile_space: float = 2.0,
    keepout: Mapping[tuple[int, int], float] | None = None,
    target_density: float | None = None,
    window: float = 100.0,
    stagger: bool = False,
    name: str = "fill",
) -> FillResult:
    """Dummy-fill ``cell`` on ``layer`` with square tiles.

    Parameters
    ----------
    cell
        The layout to fill (not modified).
    layer
        Layer of the fill tiles.
    region
        Fill region polygon (default: the bounding box of ``cell``).
    tile_size, tile_space
        Tile side and the gap between neighbouring tiles (µm).
    keepout
        ``{layer: distance}``: no tile comes closer than ``distance`` to any
        shape of ``layer``. The fill layer itself keeps ``tile_space`` unless
        listed.
    target_density
        Stop filling a window once the fill layer reaches this density
        (default: fill every allowed site).
    window
        Side of the square density windows used with ``target_density``.
    stagger
        Shift odd lattice rows by half a pitch.
    name
        Name of the returned fill cell (its tile cell is ``name + "_tile"``).

    Returns
    -------
    FillResult

    Examples
    --------
    >>> from photonix.layout import Cell, components as lc
    >>> chip = Cell("chip")
    >>> _ = chip.add_ref(lc.straight(40.0))
    >>> res = fill(chip, layer=(1, 0), region=[(0, -20), (40, -20), (40, 20), (0, 20)],
    ...            keepout={(1, 0): 2.0})
    >>> res.n_tiles, len(res.cell.references)
    (80, 80)
    """
    gd = _gdstk()
    lay = _layer(layer)
    t = float(tile_size)
    s = float(tile_space)
    if not (np.isfinite(t) and t > 0 and np.isfinite(s) and s >= 0):
        raise ValueError("tile_size must be positive and tile_space non-negative")
    keep = {_layer(k): float(v) for k, v in (keepout or {}).items()}
    keep.setdefault(lay, s)
    if any(not np.isfinite(v) or v < 0 for v in keep.values()):
        raise ValueError("keep-out distances must be non-negative and finite")
    if target_density is not None and not 0.0 < target_density <= 1.0:
        raise ValueError("target_density must lie in (0, 1]")
    if region is None:
        (bx0, by0), (bx1, by1) = cell.bbox()
        reg = np.array([(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)], float)
    else:
        reg = np.asarray(region, float).reshape(-1, 2)
    (rx0, ry0), (rx1, ry1) = reg.min(axis=0), reg.max(axis=0)

    by_layer: dict[tuple[int, int], list[Any]] = {}
    for pts, ly in cell.get_polygons():
        ly = _layer(ly)
        if ly in keep and len(pts) >= 3:
            by_layer.setdefault(ly, []).append(gd.Polygon(np.asarray(pts, float)))
    forbidden: list[Any] = []
    stretch = 1.0 / np.cos(np.pi / _ROUND)
    for ly, polys in by_layer.items():
        d = keep[ly]
        merged = gd.boolean(polys, [], "or", precision=_P)
        forbidden += gd.offset(merged, d * stretch + _P, join="round", tolerance=_ROUND, precision=_P,
                               use_union=True) if d > 0 else merged
    allowed = gd.boolean([gd.Polygon(reg)], forbidden, "not", precision=_P) if forbidden else [gd.Polygon(reg)]

    pitch = t + s
    nx = int(np.floor((rx1 - rx0 - t) / pitch + 1e-9)) + 1 if rx1 - rx0 >= t else 0
    ny = int(np.floor((ry1 - ry0 - t) / pitch + 1e-9)) + 1 if ry1 - ry0 >= t else 0
    sites_l: list[tuple[float, float]] = []
    idx_l: list[tuple[int, int]] = []
    for j in range(ny):
        shift = 0.5 * pitch if (stagger and j % 2) else 0.0
        for i in range(nx):
            x = rx0 + shift + i * pitch
            if x + t <= rx1 + 1e-9:
                sites_l.append((x, ry0 + j * pitch))
                idx_l.append((i, j))
    sites = np.asarray(sites_l, float).reshape(-1, 2)
    idx = np.asarray(idx_l, int).reshape(-1, 2)
    ok = np.zeros(len(sites), bool)
    if len(sites) and allowed:
        centres = sites + 0.5 * t
        sure = gd.offset(allowed, -(0.5 * t * np.sqrt(2.0) * stretch + 2 * _P), join="round",
                         tolerance=_ROUND, precision=_P)
        maybe = gd.offset(allowed, -(0.5 * t - 2 * _P), join="miter", tolerance=2.0, precision=_P)
        in_sure = np.asarray(gd.inside(centres.tolist(), sure), bool) if sure else np.zeros(len(sites), bool)
        in_maybe = np.asarray(gd.inside(centres.tolist(), maybe), bool) if maybe else np.zeros(len(sites), bool)
        ok |= in_sure
        for k in np.nonzero(in_maybe & ~in_sure)[0]:
            x, y = sites[k]
            tile = gd.rectangle((x, y), (x + t, y + t))
            rest = gd.boolean(tile, allowed, "not", precision=_P)
            ok[k] = sum(p.area() for p in rest) <= 1e-9
    sites, idx = sites[ok], idx[ok]

    before = after = None
    residual: list[DrcViolation] = []
    tile_cell = Cell(f"{name}_tile")
    tile_cell.add_polygon([(0, 0), (t, 0), (t, t), (0, t)], lay)

    def build(selected: np.ndarray) -> Cell:
        out = Cell(name)
        for x, y in selected:
            out.add_ref(tile_cell, origin=(float(x), float(y)))
        return out

    def measured(fill_cell: Cell) -> Any:
        from photonix.verification.density import density_map

        probe = Cell(f"{cell.name}+{name}")
        probe.add_ref(cell)
        probe.add_ref(fill_cell)
        return density_map(probe, lay, window=window, region=reg)

    if target_density is not None:
        from photonix.verification.density import density_map

        before = density_map(cell, lay, window=window, region=reg)
        chosen = np.zeros(len(sites), bool)
        if len(sites):
            centres = sites + 0.5 * t
            wi = np.floor((centres[:, 0] - rx0) / window).astype(int)
            wj = np.floor((centres[:, 1] - ry0) / window).astype(int)
            current = before
            # A first greedy pass plus top-up passes: tiles straddling a window
            # edge add density to the neighbour, so re-measure and top up.
            for _pass in range(4):
                deficit = target_density - np.nan_to_num(current.density, nan=1.0)
                added = False
                for flat in np.argsort(-deficit, axis=None, kind="stable"):
                    jy, ix = np.unravel_index(flat, deficit.shape)
                    need = deficit[jy, ix]
                    if need <= 1e-12:
                        break
                    poly = before.window_polygon(int(jy), int(ix))
                    win_area = sum(p.area() for p in gd.boolean(gd.Polygon(poly), gd.Polygon(reg), "and",
                                                                precision=_P))
                    members = np.nonzero((wi == ix) & (wj == jy) & ~chosen)[0]
                    if not len(members):
                        continue
                    order = members[_spread_order(idx[members, 0], idx[members, 1])]
                    n_need = int(np.ceil(need * win_area / (t * t) - 1e-9))
                    chosen[order[:n_need]] = True
                    added = True
                if not added:
                    break
                current = measured(build(sites[chosen]))
        sites = sites[chosen]
    out = build(sites)
    if before is not None:
        after = measured(out)
        residual = [v for v in after.violations(min_density=target_density or 0.0) if v.rule == "min_density"]
    return FillResult(out, sites, t, lay, before, after, residual)
