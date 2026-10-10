"""Fast geometric design-rule *pre-check* (minimum width, spacing and area).

This catches the common drawing mistakes -- a sliver between two cells, a
waveguide narrower than the process allows, a gap the lithography cannot
resolve -- seconds after the layout is generated, before an expensive external
run. It is **not** sign-off: real decks contain hundreds of context-dependent
rules (density, enclosure, angle, notch, antenna, ...). Use
:func:`photonix.layout.run_drc` with the foundry deck for that.

Method
------
Polygons on a layer are first merged, so abutting cells form one shape.

* *Width*: a morphological opening (erode then dilate by ``w/2``) removes every
  feature narrower than ``w``; the removed area is reported. Mitred joins with
  the minimum mitre limit restore right-angle and obtuse corners exactly, while
  tips sharper than 60 degrees are bevelled and therefore reported.
* *Spacing between shapes*: each merged island is dilated by ``s/2`` with
  round joins (a Euclidean disc); any overlap of two dilated islands is a gap
  narrower than ``s`` -- including corner-to-corner (diagonal) gaps.
* *Notches within a shape*: a mitred closing (dilate then erode by ``s/2``) of
  each island fills Manhattan notches narrower than ``s``.
* *Area*: merged islands smaller than ``a`` are reported.

Features drawn exactly at the rule value pass (a 2 x 1e-4 µm tolerance band
absorbs the offset arithmetic), and slivers thinner than 2 % of the rule (or
1 nm) are treated as numerical noise.

Not checked: the narrowest point of a within-shape corner neck (two reflex
vertices of one polygon closer than ``w``) and any context-dependent rule.
Results are reported as violation polygons with their layer and rule.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .cell import Cell
from .gds import _gdstk

__all__ = [
    "DrcViolation", "DrcReport", "check_min_width", "check_min_space", "check_min_area",
    "run_drc_precheck",
]

_PRECISION = 1e-4  # µm
_MITER = 2.0       # gdstk's minimum mitre limit: right angles restored, tips < 60 deg bevelled
_ROUND_POINTS = 64.0


@dataclass(frozen=True)
class DrcViolation:
    """One rule violation: the offending region and what it broke."""

    rule: str
    layer: tuple[int, int]
    value: float
    polygon: np.ndarray

    @property
    def bbox(self) -> tuple[tuple[float, float], tuple[float, float]]:
        p = self.polygon
        return ((float(p[:, 0].min()), float(p[:, 1].min())), (float(p[:, 0].max()), float(p[:, 1].max())))

    @property
    def area(self) -> float:
        x, y = self.polygon[:, 0], self.polygon[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


@dataclass
class DrcReport:
    """All violations found by :func:`run_drc_precheck`."""

    violations: list[DrcViolation] = field(default_factory=list)
    rules: dict = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        """True when no rule was violated (by this pre-check)."""
        return not self.violations

    def summary(self) -> dict[tuple[str, tuple[int, int]], int]:
        """Violation counts keyed by ``(rule, layer)``."""
        out: dict[tuple[str, tuple[int, int]], int] = {}
        for v in self.violations:
            out[(v.rule, v.layer)] = out.get((v.rule, v.layer), 0) + 1
        return out

    def __str__(self) -> str:
        if self.clean:
            return "DRC pre-check: clean"
        lines = [f"DRC pre-check: {len(self.violations)} violation(s)"]
        for (rule, layer), n in sorted(self.summary().items()):
            lines.append(f"  {rule:<10s} layer {layer}: {n}")
        return "\n".join(lines)


def _layer(layer) -> tuple[int, int]:
    return (int(layer[0]), int(layer[1]))


def _merged(cell: Cell, layer: tuple[int, int]):
    gdstk = _gdstk()
    polys = [gdstk.Polygon(pts) for pts, lay in cell.get_polygons() if _layer(lay) == _layer(layer)]
    if not polys:
        return []
    return gdstk.boolean(polys, [], "or", precision=_PRECISION)


def _band(rule: float) -> float:
    """Half-offset for a rule, shrunk so a feature drawn exactly at the rule passes."""
    return 0.5 * rule - 2.0 * _PRECISION


def _noise(rule: float) -> float:
    return max(10.0 * _PRECISION, 0.02 * rule)


def _real(polys, min_thickness: float):
    """Drop numerical slivers thinner than ``min_thickness`` (2 * area / perimeter)."""
    out = []
    for p in polys:
        perim = p.perimeter()
        if perim > 0 and 2.0 * p.area() / perim > min_thickness:
            out.append(np.asarray(p.points))
    return out


def check_min_width(cell: Cell, layer: tuple[int, int], min_width: float) -> list[DrcViolation]:
    """Regions of ``layer`` narrower than ``min_width`` µm.

    Examples
    --------
    >>> from photonix.layout import Cell
    >>> c = Cell("t")
    >>> _ = c.add_polygon([(0, 0), (10, 0), (10, 0.5), (0, 0.5)])     # OK
    >>> _ = c.add_polygon([(0, 2), (10, 2), (10, 2.1), (0, 2.1)])     # 100 nm wide
    >>> [v.rule for v in check_min_width(c, (1, 0), 0.15)]
    ['min_width']
    """
    gdstk = _gdstk()
    merged = _merged(cell, layer)
    if not merged:
        return []
    half = _band(min_width)
    opened = gdstk.offset(gdstk.offset(merged, -half, join="miter", tolerance=_MITER, precision=_PRECISION),
                          half, join="miter", tolerance=_MITER, precision=_PRECISION)
    removed = gdstk.boolean(merged, opened, "not", precision=_PRECISION)
    return [DrcViolation("min_width", _layer(layer), float(min_width), p)
            for p in _real(removed, _noise(min_width))]


def check_min_space(cell: Cell, layer: tuple[int, int], min_space: float) -> list[DrcViolation]:
    """Gaps and notches on ``layer`` narrower than ``min_space`` µm.

    Examples
    --------
    >>> from photonix.layout import Cell
    >>> c = Cell("t")
    >>> _ = c.add_polygon([(0, 0), (10, 0), (10, 0.5), (0, 0.5)])
    >>> _ = c.add_polygon([(0, 0.6), (10, 0.6), (10, 1.1), (0, 1.1)])  # 100 nm gap
    >>> len(check_min_space(c, (1, 0), 0.15)), len(check_min_space(c, (1, 0), 0.08))
    (1, 0)
    >>> d = Cell("diag")                       # corner-to-corner, 71 nm apart
    >>> _ = d.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> _ = d.add_polygon([(1.05, 1.05), (2, 1.05), (2, 2), (1.05, 2)])
    >>> len(check_min_space(d, (1, 0), 0.15))
    1
    """
    gdstk = _gdstk()
    merged = _merged(cell, layer)
    if not merged:
        return []
    half = _band(min_space)
    noise = _noise(min_space)
    out: list[DrcViolation] = []
    # Notches inside each island (Manhattan closing).
    for island in merged:
        closed = gdstk.offset(gdstk.offset([island], half, join="miter", tolerance=_MITER, precision=_PRECISION),
                              -half, join="miter", tolerance=_MITER, precision=_PRECISION)
        filled = gdstk.boolean(closed, [island], "not", precision=_PRECISION)
        out.extend(DrcViolation("min_space", _layer(layer), float(min_space), p) for p in _real(filled, noise))
    # Gaps between islands, with a Euclidean (round) structuring element.
    grown = [gdstk.offset([p], half, join="round", tolerance=_ROUND_POINTS, precision=_PRECISION) for p in merged]
    boxes = [p.bounding_box() for p in merged]
    for a in range(len(merged)):
        (ax0, ay0), (ax1, ay1) = boxes[a]
        for b in range(a + 1, len(merged)):
            (bx0, by0), (bx1, by1) = boxes[b]
            if bx0 > ax1 + 2 * half or ax0 > bx1 + 2 * half or by0 > ay1 + 2 * half or ay0 > by1 + 2 * half:
                continue
            overlap = gdstk.boolean(grown[a], grown[b], "and", precision=_PRECISION)
            out.extend(DrcViolation("min_space", _layer(layer), float(min_space), p)
                       for p in _real(overlap, 0.0))
    return out


def check_min_area(cell: Cell, layer: tuple[int, int], min_area: float) -> list[DrcViolation]:
    """Merged islands on ``layer`` smaller than ``min_area`` µm^2."""
    out = []
    for p in _merged(cell, layer):
        if p.area() < min_area:
            out.append(DrcViolation("min_area", _layer(layer), float(min_area), np.asarray(p.points)))
    return out


_CHECKS = {"min_width": check_min_width, "min_space": check_min_space, "min_area": check_min_area}


def run_drc_precheck(cell: Cell, rules: Mapping[tuple[int, int], Mapping[str, float]]) -> DrcReport:
    """Run width/space/area pre-checks for every layer in ``rules``.

    Parameters
    ----------
    rules
        ``{layer: {"min_width": w, "min_space": s, "min_area": a}}`` with any
        subset of the three keys per layer (µm and µm^2).

    Examples
    --------
    >>> from photonix.layout import components as lc
    >>> report = run_drc_precheck(lc.directional_coupler(gap=0.1), {(1, 0): {"min_width": 0.15, "min_space": 0.15}})
    >>> report.clean, report.summary()
    (False, {('min_space', (1, 0)): 1})
    """
    report = DrcReport(rules={_layer(k): dict(v) for k, v in rules.items()})
    for layer, layer_rules in rules.items():
        for name, value in layer_rules.items():
            if name not in _CHECKS:
                raise ValueError(f"unknown rule {name!r}; expected one of {sorted(_CHECKS)}")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"rule {name} on {layer} must be positive and finite")
            report.violations.extend(_CHECKS[name](cell, _layer(layer), float(value)))
    return report
