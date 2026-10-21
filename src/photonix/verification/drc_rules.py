"""A DRC rule deck beyond width/space/area, with waivers, markers and a generated KLayout deck.

:func:`photonix.layout.run_drc_precheck` checks single-layer minimum width,
space and area. This module adds the context rules a PIC layout needs before
the foundry run -- enclosure, separation, extension, overlap/inside, acute
angles, off-grid vertices, maximum width, hole area and pattern density --
collected in a :class:`RuleDeck` that

* runs on a :class:`~photonix.layout.Cell` (:meth:`RuleDeck.run`) and returns
  the shared :class:`~photonix.layout.DrcReport` record (as a
  :class:`DeckReport` that also lists waived violations);
* serialises to JSON and is built from a PDK's rule table
  (:meth:`RuleDeck.from_pdk`);
* writes an equivalent KLayout DRC script (:meth:`RuleDeck.to_klayout_drc`),
  so KLayout can serve as an independent engine (:func:`compare_with_klayout`).

The deck is photonix's open rule table, not a foundry deck.

Method
------
All checks run on merged layers with gdstk booleans and offsets on the
1e-4 µm grid of the pre-check. Distances are measured with one of three
metrics, as in KLayout:

``"euclidean"``
    Round-joined offsets (a disc structuring element): every direction,
    corner-to-corner included.
``"square"``
    Mitred offsets (a square structuring element for Manhattan edges).
``"projection"``
    Only edges whose projections overlap interact: the offsets are
    Minkowski sums with axis-parallel segments, taken separately along
    ``x`` and ``y`` (exact for Manhattan geometry; oblique edges are
    measured along the axes).

Features drawn exactly at a rule value pass (offsets are shrunk by a
2e-4 µm band) and violation pieces thinner than 3e-4 µm (2 area/perimeter)
are discarded as numerical slivers, so a rule violated by two grid units
(2 nm) is reliably found.

* *enclosure* ``(inner, outer, e)``: ``inner NOT erode(outer, e)``; inner
  shapes outside ``outer`` are violations too.
* *separation* ``(a, b, s)``: ``grow(a, s/2) AND grow(b, s/2)``; overlapping
  ``a`` and ``b`` are violations (distance 0). With ``a == b`` (*space*),
  pairs of distinct islands are checked and notches are closed with a mitred
  closing.
* *extension* ``(a, b, x)``: ``b`` must extend at least ``x`` beyond ``a``
  wherever ``a`` lies inside ``b``, except where ``a`` itself leaves ``b``:
  ``(grow(a AND b, x) NOT b) NOT grow(a NOT b, x)``. Example: the slab
  around a rib, which may end where the rib becomes a strip.
* *not_overlap* ``a AND b``; *inside* ``a NOT b``.
* *min_angle*: interior angle at every vertex of every merged outline and
  hole (``atan2`` of adjacent edges, oriented by the polygon's signed area).
* *grid*: drawn vertices with ``|x / g - round(x / g)| > 1e-6``.
* *max_width*: the morphological opening at ``w`` -- features that survive
  an erosion by ``w/2`` are reported whole.
* *hole_area*: enclosed voids (pieces of the complement that do not touch
  the bounding frame) smaller than the value.
* *min_density*, *max_density*: :func:`photonix.verification.check_density`.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from photonix.layout.cell import Cell
from photonix.layout.drc import DrcReport, DrcViolation, check_min_area, check_min_space, check_min_width

from ._geom import boundary_loops, box_polygon, gdstk, layer_tuple, signed_area, sweep_pairs

__all__ = [
    "RULE_KINDS", "METRICS", "Rule", "RuleDeck", "DeckReport", "Waiver", "apply_waivers",
    "check_space", "check_enclosure", "check_separation", "check_extension", "check_not_overlap", "check_inside",
    "check_min_angle", "check_offgrid_vertices", "check_max_width", "check_hole_area",
    "compare_with_klayout",
]

_P = 1e-4          # boolean grid (µm), as in photonix.layout.drc
_BAND = 2.0 * _P   # features exactly at the rule value pass
_NOISE = 3.0 * _P  # thinner violation pieces are numerical slivers
_ROUND = 256.0     # points per full circle of a round join
_MITER = 2.0

#: Rule kinds and whether they take a second layer.
RULE_KINDS: dict[str, bool] = {
    "width": False, "space": False, "area": False, "enclosure": True, "separation": True, "extension": True,
    "not_overlap": True, "inside": True, "min_angle": False, "grid": False, "max_width": False,
    "hole_area": False, "min_density": False, "max_density": False,
}
#: Distance metrics (KLayout names: ``euclidian``, ``square``, ``projection``).
METRICS = ("euclidean", "square", "projection")


# --------------------------------------------------------------------------- #
# polygon algebra
# --------------------------------------------------------------------------- #
def _merged(cell: Cell, layer: tuple[int, int]) -> list:
    gd = gdstk()
    lay = layer_tuple(layer)
    polys = [gd.Polygon(pts) for pts, ly in cell.get_polygons() if layer_tuple(ly) == lay and len(pts) >= 3]
    return gd.boolean(polys, [], "or", precision=_P) if polys else []


def _bool(a: list, b: list, op: str) -> list:
    if op == "and" and (not a or not b):
        return []
    if op == "not" and not b:
        return list(a)
    if op == "not" and not a:
        return []
    if op == "or" and not b:
        return gdstk().boolean(a, [], "or", precision=_P) if a else []
    return gdstk().boolean(a, b, op, precision=_P)


def _offset(polys: list, d: float, join: str) -> list:
    if not polys:
        return []
    if d == 0:
        return list(polys)
    tol = _ROUND if join == "round" else _MITER
    return gdstk().offset(polys, d, join=join, tolerance=tol, precision=_P, use_union=True)


def _grow_axis(polys: list, d: float, axis: int) -> list:
    """Minkowski sum with the segment ``[-d, d]`` along ``axis``: the polygon,
    its two translates and every edge swept along the segment."""
    if not polys or d <= 0:
        return list(polys)
    gd = gdstk()
    v = np.array([d, 0.0]) if axis == 0 else np.array([0.0, d])
    parts = []
    for p in polys:
        pts = np.asarray(p.points, float)
        parts += [gd.Polygon(pts), gd.Polygon(pts + v), gd.Polygon(pts - v)]
        q = np.roll(pts, -1, axis=0)
        for a, b in zip(pts, q, strict=True):
            quad = np.array([a - v, b - v, b + v, a + v])
            if abs(signed_area(quad)) > 1e-18:
                parts.append(gd.Polygon(quad))
    return gd.boolean(parts, [], "or", precision=_P)


def _frame(polys: Sequence, pad: float) -> list:
    pts = np.concatenate([np.asarray(p.points) for p in polys])
    (x0, y0), (x1, y1) = pts.min(axis=0) - pad, pts.max(axis=0) + pad
    return [gdstk().rectangle((x0, y0), (x1, y1))]


def _grow(polys: list, d: float, metric: str) -> list:
    if metric == "euclidean":
        return _offset(polys, d, "round")
    if metric == "square":
        return _offset(polys, d, "miter")
    raise ValueError("projection growth is axis-specific")  # pragma: no cover - internal


def _shrink(polys: list, d: float, metric: str) -> list:
    if not polys or d <= 0:
        return list(polys)
    if metric == "euclidean":
        return _offset(polys, -d, "round")
    if metric == "square":
        return _offset(polys, -d, "miter")
    frame = _frame(polys, 2 * d + 1.0)
    comp = _bool(frame, polys, "not")
    out = list(polys)
    for axis in (0, 1):
        out = _bool(out, _grow_axis(comp, d, axis), "not")
    return out


def _interacting(a: list, b: list, h: float, metric: str) -> list:
    """Points within ``2 h`` of both ``a`` and ``b`` (under ``metric``)."""
    if metric == "projection":
        out: list = []
        for axis in (0, 1):
            out += _bool(_grow_axis(a, h, axis), _grow_axis(b, h, axis), "and")
        return _bool(out, [], "or")
    return _bool(_grow(a, h, metric), _grow(b, h, metric), "and")


def _pieces(polys: Iterable, min_thickness: float = _NOISE) -> list[np.ndarray]:
    out = []
    for p in polys:
        perim = p.perimeter()
        if perim > 0 and 2.0 * p.area() / perim > min_thickness:
            out.append(np.asarray(p.points, float))
    return out


def _viol(rule: str, layer: tuple[int, int], value: float, polys: Iterable) -> list[DrcViolation]:
    return [DrcViolation(rule, layer_tuple(layer), float(value), p) for p in _pieces(polys)]


def _check_metric(metric: str) -> str:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    return metric


def _positive(name: str, value: float) -> float:
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return float(value)


def _bbox(p: Any) -> tuple[float, float, float, float]:
    (x0, y0), (x1, y1) = p.bounding_box()
    return (float(x0), float(y0), float(x1), float(y1))


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_space(cell: Cell, layer: tuple[int, int], space: float, *, metric: str = "euclidean",
                rule: str = "space") -> list[DrcViolation]:
    """Gaps between islands (and notches) of ``layer`` narrower than ``space``.

    ``metric="euclidean"`` is :func:`photonix.layout.check_min_space`;
    ``"square"`` and ``"projection"`` change how corner-to-corner gaps count
    (see the module notes).

    Examples
    --------
    >>> from photonix.layout import Cell
    >>> c = Cell("diag")                                # corners 0.6 x 0.6 apart, 0.85 diagonal
    >>> _ = c.add_polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    >>> _ = c.add_polygon([(2.6, 2.6), (4, 2.6), (4, 4), (2.6, 4)])
    >>> [len(check_space(c, (1, 0), 1.0, metric=m)) for m in ("euclidean", "square", "projection")]
    [1, 1, 0]
    """
    space = _positive("space", space)
    lay = layer_tuple(layer)
    if _check_metric(metric) == "euclidean":
        return [DrcViolation(rule, v.layer, v.value, v.polygon) for v in check_min_space(cell, lay, space)]
    merged = _merged(cell, lay)
    h = 0.5 * space - _BAND
    out: list[DrcViolation] = []
    for island in merged:
        closed = _offset(_offset([island], h, "miter"), -h, "miter")
        out += _viol(rule, lay, space, _bool(closed, [island], "not"))
    boxes = [_bbox(p) for p in merged]
    for i, j in sweep_pairs(boxes, pad=h):
        out += _viol(rule, lay, space, _interacting([merged[i]], [merged[j]], h, metric))
    return out


def check_enclosure(cell: Cell, inner: tuple[int, int], outer: tuple[int, int], enclosure: float, *,
                    metric: str = "euclidean", rule: str = "enclosure") -> list[DrcViolation]:
    """Parts of ``inner`` closer than ``enclosure`` to the edge of ``outer`` (or outside it).

    Examples
    --------
    >>> from photonix.layout import Cell
    >>> c = Cell("via")
    >>> _ = c.add_polygon([(0, 0), (2, 0), (2, 2), (0, 2)], layer=(11, 0))           # metal
    >>> _ = c.add_polygon([(0.55, 0.55), (1.45, 0.55), (1.45, 1.45), (0.55, 1.45)], layer=(12, 0))  # via
    >>> len(check_enclosure(c, (12, 0), (11, 0), 0.5)), len(check_enclosure(c, (12, 0), (11, 0), 0.6))
    (0, 1)
    """
    enclosure = _positive("enclosure", enclosure)
    _check_metric(metric)
    inn, out = _merged(cell, inner), _merged(cell, outer)
    if not inn:
        return []
    allowed = _shrink(out, enclosure - _BAND, metric)
    return _viol(rule, inner, enclosure, _bool(inn, allowed, "not"))


def check_separation(cell: Cell, a: tuple[int, int], b: tuple[int, int], separation: float, *,
                     metric: str = "euclidean", rule: str = "separation") -> list[DrcViolation]:
    """Regions where layer ``a`` comes closer than ``separation`` to layer ``b``.

    Overlapping ``a`` and ``b`` are violations too (distance zero); use
    :func:`check_not_overlap` to report overlaps on their own.

    Examples
    --------
    >>> from photonix.layout import Cell
    >>> c = Cell("heater")
    >>> _ = c.add_polygon([(0, -0.25), (50, -0.25), (50, 0.25), (0, 0.25)], layer=(1, 0))   # waveguide
    >>> _ = c.add_polygon([(0, 1.0), (50, 1.0), (50, 3.0), (0, 3.0)], layer=(11, 0))      # metal, 0.75 away
    >>> len(check_separation(c, (11, 0), (1, 0), 1.0)), len(check_separation(c, (11, 0), (1, 0), 0.7))
    (1, 0)
    """
    separation = _positive("separation", separation)
    _check_metric(metric)
    pa, pb = _merged(cell, a), _merged(cell, b)
    if not pa or not pb:
        return []
    h = 0.5 * separation - _BAND
    out: list[DrcViolation] = []
    ba, bb = [_bbox(p) for p in pa], [_bbox(p) for p in pb]
    pairs = sweep_pairs(ba + bb, pad=h)
    na = len(pa)
    for i, j in pairs:
        if (i < na) == (j < na):
            continue
        ia, ib = (i, j - na) if i < na else (j, i - na)
        out += _viol(rule, a, separation, _interacting([pa[ia]], [pb[ib]], h, metric))
    return out


def check_extension(cell: Cell, a: tuple[int, int], b: tuple[int, int], extension: float, *,
                    metric: str = "euclidean", rule: str = "extension") -> list[DrcViolation]:
    """Where ``b`` extends less than ``extension`` beyond ``a`` (``a`` inside ``b``).

    Places where ``a`` itself leaves ``b`` (a rib becoming a strip at the
    end of its slab) are not violations.

    Examples
    --------
    >>> from photonix.layout import Cell
    >>> c = Cell("rib")
    >>> _ = c.add_polygon([(-10, -0.25), (10, -0.25), (10, 0.25), (-10, 0.25)], layer=(1, 0))   # rib core
    >>> _ = c.add_polygon([(-5, -1.25), (5, -1.25), (5, 1.25), (-5, 1.25)], layer=(2, 0))       # slab
    >>> len(check_extension(c, (1, 0), (2, 0), 1.0)), len(check_extension(c, (1, 0), (2, 0), 1.2))
    (0, 2)
    """
    extension = _positive("extension", extension)
    _check_metric(metric)
    pa, pb = _merged(cell, a), _merged(cell, b)
    ab = _bool(pa, pb, "and")
    if not ab:
        return []
    leave = _bool(pa, pb, "not")

    def grow(polys: list, d: float) -> list:
        if metric == "projection":
            return _bool(_grow_axis(polys, d, 0) + _grow_axis(polys, d, 1), [], "or")
        return _grow(polys, d, metric)

    region = _bool(grow(ab, extension - _BAND), pb, "not")
    if leave:
        region = _bool(region, grow(leave, extension), "not")
    return _viol(rule, a, extension, region)


def check_not_overlap(cell: Cell, a: tuple[int, int], b: tuple[int, int], *,
                      rule: str = "not_overlap") -> list[DrcViolation]:
    """Regions where layers ``a`` and ``b`` overlap."""
    return _viol(rule, a, 0.0, _bool(_merged(cell, a), _merged(cell, b), "and"))


def check_inside(cell: Cell, a: tuple[int, int], b: tuple[int, int], *, rule: str = "inside") -> list[DrcViolation]:
    """Parts of layer ``a`` that lie outside layer ``b``."""
    return _viol(rule, a, 0.0, _bool(_merged(cell, a), _merged(cell, b), "not"))


def _interior_angles(loop: np.ndarray, material_left: bool) -> np.ndarray:
    d_in = loop - np.roll(loop, 1, axis=0)
    d_out = np.roll(loop, -1, axis=0) - loop
    turn = np.arctan2(d_in[:, 0] * d_out[:, 1] - d_in[:, 1] * d_out[:, 0], np.sum(d_in * d_out, axis=1))
    return np.degrees(np.pi - (turn if material_left else -turn))


def check_min_angle(cell: Cell, layer: tuple[int, int], min_angle: float, *, rule: str = "min_angle",
                    marker: float = 0.05) -> list[DrcViolation]:
    """Vertices of ``layer`` whose interior angle is below ``min_angle`` degrees.

    Each violation is a small triangle (legs up to ``marker`` µm) at the vertex.

    Examples
    --------
    >>> from photonix.layout import Cell
    >>> c = Cell("t")
    >>> _ = c.add_polygon([(0, 0), (4, 0), (4, 4 * 0.5773502691896257)])     # 30-60-90 triangle
    >>> _ = c.add_polygon([(10, 0), (12, 0), (12, 2), (10, 2)])               # square
    >>> [round(v.value) for v in check_min_angle(c, (1, 0), 45.0)]
    [45]
    """
    if not (0 < min_angle < 180):
        raise ValueError("min_angle must lie in (0, 180) degrees")
    lay = layer_tuple(layer)
    out: list[DrcViolation] = []
    for poly in _merged(cell, lay):
        pts = np.asarray(poly.points, float)
        left = signed_area(pts) > 0
        for loop in boundary_loops(pts):
            ang = _interior_angles(loop, left)
            for k in np.nonzero(ang < min_angle - 1e-9)[0]:
                v, prv, nxt = loop[k], loop[k - 1], loop[(k + 1) % len(loop)]
                legs = [(p - v) / np.linalg.norm(p - v) * min(marker, 0.5 * np.linalg.norm(p - v)) for p in (prv, nxt)]
                out.append(DrcViolation(rule, lay, float(min_angle), np.array([v, v + legs[0], v + legs[1]])))
    return out


def check_offgrid_vertices(cell: Cell, layer: tuple[int, int], grid: float, *,
                           rule: str = "grid") -> list[DrcViolation]:
    """Drawn vertices of ``layer`` that are not on the ``grid`` (µm); one marker per vertex.

    Examples
    --------
    >>> from photonix.layout import Cell
    >>> c = Cell("t")
    >>> _ = c.add_polygon([(0, 0), (1, 0), (1, 1.0005), (0, 1)])
    >>> [tuple(float(x) for x in v.polygon.mean(axis=0).round(4)) for v in check_offgrid_vertices(c, (1, 0), 0.001)]
    [(1.0, 1.0005)]
    """
    grid = _positive("grid", grid)
    lay = layer_tuple(layer)
    seen: set[tuple[float, float]] = set()
    out: list[DrcViolation] = []
    half = max(0.5 * grid, _P)
    for pts, ly in cell.get_polygons():
        if layer_tuple(ly) != lay:
            continue
        q = np.asarray(pts, float) / grid
        bad = np.any(np.abs(q - np.round(q)) > 1e-6, axis=1)
        for v in np.asarray(pts, float)[bad]:
            key = (round(float(v[0]), 9), round(float(v[1]), 9))
            if key not in seen:
                seen.add(key)
                out.append(DrcViolation(rule, lay, grid, box_polygon(v, half)))
    return out


def check_max_width(cell: Cell, layer: tuple[int, int], max_width: float, *,
                    rule: str = "max_width") -> list[DrcViolation]:
    """Features of ``layer`` wider than ``max_width`` (reported whole, e.g. for metal slotting).

    Examples
    --------
    >>> from photonix.layout import Cell
    >>> c = Cell("m")
    >>> _ = c.add_polygon([(0, 0), (40, 0), (40, 10), (0, 10)], layer=(11, 0))
    >>> [round(v.area, 6) for v in check_max_width(c, (11, 0), 5.0)]
    [400.0]
    """
    max_width = _positive("max_width", max_width)
    lay = layer_tuple(layer)
    merged = _merged(cell, lay)
    h = 0.5 * max_width + _BAND
    core = [p for p in _offset(merged, -h, "miter")
            if p.perimeter() > 0 and 2 * p.area() / p.perimeter() > _NOISE]
    if not core:
        return []
    return _viol(rule, lay, max_width, _bool(_offset(core, h, "miter"), merged, "and"))


def check_hole_area(cell: Cell, layer: tuple[int, int], min_area: float, *,
                    rule: str = "hole_area") -> list[DrcViolation]:
    """Enclosed holes of ``layer`` smaller than ``min_area`` µm^2.

    Examples
    --------
    >>> from photonix.layout import Cell
    >>> c = Cell("h")
    >>> _ = c.add_polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    >>> _ = c.add_polygon([(3, 3), (3, 3.2), (3.2, 3.2), (3.2, 3)], layer=(1, 1))
    >>> d = Cell("d")
    >>> _ = d.add_polygon([(0, 0), (10, 0), (10, 3), (0, 3)])
    >>> _ = d.add_polygon([(0, 3.2), (10, 3.2), (10, 10), (0, 10)])
    >>> _ = d.add_polygon([(0, 3), (3, 3), (3, 3.2), (0, 3.2)])
    >>> _ = d.add_polygon([(3.2, 3), (10, 3), (10, 3.2), (3.2, 3.2)])          # 0.2 x 0.2 hole
    >>> [round(v.area, 9) for v in check_hole_area(d, (1, 0), 0.1)]
    [0.04]
    """
    min_area = _positive("min_area", min_area)
    lay = layer_tuple(layer)
    merged = _merged(cell, lay)
    if not merged:
        return []
    frame = _frame(merged, 1.0)
    fx0, fy0, fx1, fy1 = _bbox(frame[0])
    out: list[DrcViolation] = []
    for piece in _bool(frame, merged, "not"):
        x0, y0, x1, y1 = _bbox(piece)
        if min(x0 - fx0, y0 - fy0, fx1 - x1, fy1 - y1) < 1e-9:
            continue  # the outside
        if piece.area() < min_area:
            out.append(DrcViolation(rule, lay, min_area, np.asarray(piece.points, float)))
    return out


# --------------------------------------------------------------------------- #
# rules, decks, waivers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rule:
    """One design rule.

    Attributes
    ----------
    name
        Rule name, used as the violation's ``rule`` and marker category.
    kind
        One of :data:`RULE_KINDS`.
    layer
        Primary layer (the inner layer of an enclosure; ``a`` of a two-layer rule).
    other
        Second layer of enclosure (outer), separation, extension, not_overlap
        and inside rules.
    value
        Rule value: µm for distances and widths, µm^2 for areas, degrees for
        ``min_angle``, µm for ``grid``, a fraction for densities.
    metric
        Distance metric of space, enclosure, separation and extension rules.
    window
        Square window side (µm) of density rules.
    description
        Free text, written to marker databases and KLayout decks.

    Examples
    --------
    >>> r = Rule("M1.EN.V", "enclosure", (12, 0), (11, 0), 0.2, description="metal over via")
    >>> r.kind, r.other
    ('enclosure', (11, 0))
    """

    name: str
    kind: str
    layer: tuple[int, int]
    other: tuple[int, int] | None = None
    value: float = 0.0
    metric: str = "euclidean"
    window: float = 100.0
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in RULE_KINDS:
            raise ValueError(f"unknown rule kind {self.kind!r}; expected one of {sorted(RULE_KINDS)}")
        object.__setattr__(self, "layer", layer_tuple(self.layer))
        if RULE_KINDS[self.kind]:
            if self.other is None:
                raise ValueError(f"rule {self.name!r} ({self.kind}) needs a second layer 'other'")
            object.__setattr__(self, "other", layer_tuple(self.other))
        elif self.other is not None:
            raise ValueError(f"rule {self.name!r} ({self.kind}) takes no second layer")
        _check_metric(self.metric)
        v = float(self.value)
        object.__setattr__(self, "value", v)
        if self.kind in ("min_density", "max_density"):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"density rule {self.name!r} needs a value in [0, 1]")
            _positive("window", self.window)
        elif self.kind == "min_angle":
            if not 0 < v < 180:
                raise ValueError(f"min_angle rule {self.name!r} needs a value in (0, 180) degrees")
        elif self.kind not in ("not_overlap", "inside"):
            _positive(f"value of rule {self.name!r}", v)

    def check(self, cell: Cell) -> list[DrcViolation]:
        """Violations of this rule in ``cell``."""
        k, n, lay, oth, v, m = self.kind, self.name, self.layer, self.other, self.value, self.metric
        if k == "width":
            return [DrcViolation(n, x.layer, x.value, x.polygon) for x in check_min_width(cell, lay, v)]
        if k == "space":
            return check_space(cell, lay, v, metric=m, rule=n)
        if k == "area":
            return [DrcViolation(n, x.layer, x.value, x.polygon) for x in check_min_area(cell, lay, v)]
        assert oth is not None or not RULE_KINDS[k]
        if k == "enclosure":
            return check_enclosure(cell, lay, oth, v, metric=m, rule=n)  # type: ignore[arg-type]
        if k == "separation":
            return check_separation(cell, lay, oth, v, metric=m, rule=n)  # type: ignore[arg-type]
        if k == "extension":
            return check_extension(cell, lay, oth, v, metric=m, rule=n)  # type: ignore[arg-type]
        if k == "not_overlap":
            return check_not_overlap(cell, lay, oth, rule=n)  # type: ignore[arg-type]
        if k == "inside":
            return check_inside(cell, lay, oth, rule=n)  # type: ignore[arg-type]
        if k == "min_angle":
            return check_min_angle(cell, lay, v, rule=n)
        if k == "grid":
            return check_offgrid_vertices(cell, lay, v, rule=n)
        if k == "max_width":
            return check_max_width(cell, lay, v, rule=n)
        if k == "hole_area":
            return check_hole_area(cell, lay, v, rule=n)
        from .density import check_density

        lo, hi = (v, 1.0) if k == "min_density" else (0.0, v)
        return [DrcViolation(n, x.layer, v, x.polygon)
                for x in check_density(cell, lay, min_density=lo, max_density=hi, window=(self.window, self.window))]


@dataclass(frozen=True)
class Waiver:
    """A waived region: violations of ``rule`` lying entirely inside ``region`` are accepted.

    Attributes
    ----------
    rule
        Rule name (``"*"`` waives every rule).
    region
        Polygon ``(n, 2)`` in µm.
    reason
        Why it is waived (recorded in the report).
    layer
        Restrict the waiver to violations on this layer.
    """

    rule: str
    region: np.ndarray = field(compare=False)
    reason: str = ""
    layer: tuple[int, int] | None = None

    def covers(self, v: DrcViolation) -> bool:
        """Whether this waiver accepts violation ``v``."""
        if self.rule not in ("*", v.rule):
            return False
        if self.layer is not None and layer_tuple(self.layer) != v.layer:
            return False
        gd = gdstk()
        rest = gd.boolean(gd.Polygon(np.asarray(v.polygon, float)), gd.Polygon(np.asarray(self.region, float)),
                          "not", precision=_P)
        return sum(p.area() for p in rest) <= 1e-9


@dataclass
class DeckReport(DrcReport):
    """A :class:`~photonix.layout.DrcReport` from a :class:`RuleDeck`, with the waived violations."""

    waived: list[tuple[DrcViolation, Waiver]] = field(default_factory=list)

    def __str__(self) -> str:
        head = "DRC (photonix rule deck): " + ("clean" if self.clean else f"{len(self.violations)} violation(s)")
        lines = [head + (f", {len(self.waived)} waived" if self.waived else "")]
        for (rule, layer), n in sorted(self.summary().items()):
            lines.append(f"  {rule:<16s} layer {layer}: {n}")
        return "\n".join(lines)


def apply_waivers(report: DrcReport, waivers: Iterable[Waiver]) -> DeckReport:
    """Move violations covered by a waiver from ``violations`` to ``waived``.

    Examples
    --------
    >>> import numpy as np
    >>> from photonix.layout import Cell, run_drc_precheck
    >>> c = Cell("t")
    >>> _ = c.add_polygon([(0, 0), (10, 0), (10, 0.1), (0, 0.1)])
    >>> rep = run_drc_precheck(c, {(1, 0): {"min_width": 0.15}})
    >>> box = np.array([(-1, -1), (11, -1), (11, 1), (-1, 1)])
    >>> out = apply_waivers(rep, [Waiver("min_width", box, "test structure")])
    >>> out.clean, len(out.waived)
    (True, 1)
    """
    ws = list(waivers)
    kept, waived = [], []
    for v in report.violations:
        hit = next((w for w in ws if w.covers(v)), None)
        if hit is None:
            kept.append(v)
        else:
            waived.append((v, hit))
    prev = list(getattr(report, "waived", []))
    return DeckReport(kept, dict(report.rules), prev + waived)


_KL_METRIC = {"euclidean": "euclidian", "square": "square", "projection": "projection"}


def _rb_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("#", "\\#") + '"'


def _rb_num(v: float) -> str:
    return repr(float(v))


@dataclass
class RuleDeck:
    """An ordered set of :class:`Rule` objects.

    Examples
    --------
    >>> from photonix.layout import components as lc
    >>> deck = RuleDeck([Rule("WG.W", "width", (1, 0), value=0.15),
    ...                  Rule("WG.S", "space", (1, 0), value=0.15)])
    >>> rep = deck.run(lc.directional_coupler(gap=0.1))
    >>> rep.summary()
    {('WG.S', (1, 0)): 1}
    """

    rules: list[Rule] = field(default_factory=list)
    name: str = "photonix"

    def __post_init__(self) -> None:
        names = [r.name for r in self.rules]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"duplicate rule names: {dup}")

    def run(self, cell: Cell, *, waivers: Iterable[Waiver] = ()) -> DeckReport:
        """Check every rule on ``cell`` and apply ``waivers``."""
        report = DeckReport(rules={r.name: r for r in self.rules})
        for r in self.rules:
            report.violations.extend(r.check(cell))
        ws = list(waivers)
        return apply_waivers(report, ws) if ws else report

    # -- serialisation ------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        """Plain-data form (``json`` serialisable)."""
        return {"name": self.name, "rules": [{k: (list(v) if isinstance(v, tuple) else v)
                                               for k, v in asdict(r).items()} for r in self.rules]}

    def to_json(self, path: str | os.PathLike[str] | None = None) -> str:
        """JSON text of the deck (also written to ``path`` when given)."""
        text = json.dumps(self.to_dict(), indent=1, sort_keys=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, source: str | os.PathLike[str]) -> RuleDeck:
        """Deck from JSON text or a JSON file path."""
        text = str(source)
        if not text.lstrip().startswith("{"):
            text = Path(source).read_text(encoding="utf-8")
        data = json.loads(text)
        rules = []
        for d in data["rules"]:
            d = dict(d)
            d["layer"] = tuple(d["layer"])
            if d.get("other") is not None:
                d["other"] = tuple(d["other"])
            rules.append(Rule(**d))
        return cls(rules, data.get("name", "photonix"))

    @classmethod
    def from_pdk(cls, pdk: Any) -> RuleDeck:
        """Width/space/area rules of a :class:`~photonix.pdk.Pdk` (``add_design_rules``).

        Rule names are ``"<LAYER>.W"``, ``".S"`` and ``".A"``.

        Examples
        --------
        >>> from photonix.pdk import demo_pdk
        >>> [r.name for r in RuleDeck.from_pdk(demo_pdk()).rules]
        ['WG.W', 'WG.S', 'METAL.W', 'METAL.S']
        """
        kinds = {"min_width": ("width", "W"), "min_space": ("space", "S"), "min_area": ("area", "A")}
        rules = []
        for lname, table in pdk.design_rules.items():
            layer = pdk.layers[lname].tuple
            for key, value in table.items():
                if key not in kinds:
                    raise ValueError(f"unsupported PDK rule {key!r} on layer {lname!r}")
                kind, suffix = kinds[key]
                rules.append(Rule(f"{lname}.{suffix}", kind, layer, value=float(value),
                                  description=f"{key} {value:g} ({pdk.name})"))
        return cls(rules, str(pdk.name))

    # -- KLayout --------------------------------------------------------------- #
    def to_klayout_drc(self, path: str | os.PathLike[str] | None = None, *, report_var: str = "report",
                       input_var: str = "input") -> str:
        """An equivalent KLayout DRC script (Ruby DSL) for this deck.

        The script reads the layout from ``$<input_var>`` and writes its
        report to ``$<report_var>``, the runtime variables that
        :func:`photonix.layout.run_klayout_deck` supplies. Edge-pair results
        are converted to polygons so markers carry areas. Rules without a
        reliable KLayout equivalent (densities, ``min_angle``) are emitted as
        comments only.

        Examples
        --------
        >>> deck = RuleDeck([Rule("WG.W", "width", (1, 0), value=0.15)])
        >>> print(deck.to_klayout_drc().splitlines()[-1])
        l1_0.width(0.15.um, euclidian).polygons.output("WG.W", "width 0.15")
        """
        lines = [f"# KLayout DRC deck generated by photonix from rule deck {self.name!r}.",
                 "# Not a foundry deck: it mirrors photonix.verification.RuleDeck for cross-checking.",
                 f"source(${input_var})", f"report({_rb_str(self.name)}, ${report_var})", "flat"]
        layers = sorted({r.layer for r in self.rules} | {r.other for r in self.rules if r.other is not None})
        for lay in layers:
            lines.append(f"l{lay[0]}_{lay[1]} = input({lay[0]}, {lay[1]})")
        for r in self.rules:
            a = f"l{r.layer[0]}_{r.layer[1]}"
            b = f"l{r.other[0]}_{r.other[1]}" if r.other is not None else ""
            v, m = _rb_num(r.value), _KL_METRIC[r.metric]
            desc = _rb_str(r.description or f"{r.kind} {r.value:g}")
            name = _rb_str(r.name)
            expr = {
                "width": f"{a}.width({v}.um, {m}).polygons",
                "space": f"{a}.space({v}.um, {m}).polygons",
                "area": f"{a}.with_area(nil, {v}.um2)",
                "enclosure": f"{b}.enclosing({a}, {v}.um, {m}).polygons",
                "separation": f"{a}.separation({b}, {v}.um, {m}).polygons",
                "extension": f"(({a} & {b}).sized({v}.um) - {b} - ({a} - {b}).sized({v}.um))",
                "not_overlap": f"({a} & {b})",
                "inside": f"({a} - {b})",
                "grid": f"{a}.ongrid({v}.um)",
                "max_width": f"({a}.sized(-{_rb_num(r.value / 2)}.um).sized({_rb_num(r.value / 2)}.um) & {a})",
                "hole_area": f"{a}.holes.with_area(nil, {v}.um2)",
            }.get(r.kind)
            if expr is None:
                lines.append(f"# {r.name}: {r.kind} {r.value:g} is not translated (check it with photonix)")
                continue
            lines.append(f"{expr}.output({name}, {desc})")
        text = "\n".join(lines) + "\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text.rstrip("\n")


def compare_with_klayout(cell: Cell, deck: RuleDeck, *, workdir: str | os.PathLike[str],
                         executable: str | os.PathLike[str] | None = None, rtol: float = 0.01,
                         timeout: float | None = 600.0) -> dict[str, dict[str, Any]]:
    """Run ``deck`` with photonix and with KLayout on the same GDS and compare per rule.

    Writes ``layout.gds``, ``deck.drc`` and ``klayout.lyrdb`` to ``workdir``
    and runs KLayout through :func:`photonix.layout.run_klayout_deck`
    (``FileNotFoundError`` without a KLayout executable). For every
    translated rule the result holds the violation counts and total marker
    areas of both engines, ``presence`` (both flag the rule, or neither) and
    ``area_agrees`` (total areas within ``rtol``; meaningful for rules whose
    markers are the violating region -- width, area, overlap, inside,
    hole area -- whereas KLayout's space/separation/enclosure markers span
    the edge pairs rather than photonix's grown overlaps).
    """
    from photonix.layout.gds import write_gds
    from photonix.layout.klayout import run_klayout_deck

    from .rdb import read_lyrdb

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    gds = write_gds(cell, work / "layout.gds")
    drc = work / "deck.drc"
    deck.to_klayout_drc(drc)
    lyrdb = work / "klayout.lyrdb"
    run_klayout_deck(gds, drc, report_path=lyrdb, executable=executable, timeout=timeout)
    theirs = read_lyrdb(lyrdb)
    ours = deck.run(cell)
    out: dict[str, dict[str, Any]] = {}
    for r in deck.rules:
        if r.kind in ("min_density", "max_density", "min_angle"):
            continue
        mine = [v for v in ours.violations if v.rule == r.name]
        kl = [v for v in theirs.violations if v.rule == r.name]
        a_mine, a_kl = float(sum(v.area for v in mine)), float(sum(v.area for v in kl))
        out[r.name] = {
            "photonix_count": len(mine), "klayout_count": len(kl),
            "photonix_area": a_mine, "klayout_area": a_kl,
            "presence": bool(mine) == bool(kl),
            "area_agrees": math.isclose(a_mine, a_kl, rel_tol=rtol, abs_tol=1e-6),
        }
    return out

