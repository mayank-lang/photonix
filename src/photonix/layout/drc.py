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
  feature narrower than ``w``; the removed area is reported.
* *Spacing*: a closing (dilate then erode by ``s/2``) fills every gap or notch
  narrower than ``s``; the filled area is reported.
* *Area*: merged islands smaller than ``a`` are reported.

Offsets use mitred joins, which restore convex and reflex corners exactly, so
ordinary rectangles and Manhattan shapes produce no false corner violations.
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
_MITER = 100.0


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
    half = 0.5 * min_width
    opened = gdstk.offset(gdstk.offset(merged, -half, join="miter", tolerance=_MITER, precision=_PRECISION),
                          half, join="miter", tolerance=_MITER, precision=_PRECISION)
    removed = gdstk.boolean(merged, opened, "not", precision=_PRECISION)
    return [DrcViolation("min_width", _layer(layer), float(min_width), p)
            for p in _real(removed, 10 * _PRECISION)]


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
    """
    gdstk = _gdstk()
    merged = _merged(cell, layer)
    if not merged:
        return []
    half = 0.5 * min_space
    closed = gdstk.offset(gdstk.offset(merged, half, join="miter", tolerance=_MITER, precision=_PRECISION),
                          -half, join="miter", tolerance=_MITER, precision=_PRECISION)
    filled = gdstk.boolean(closed, merged, "not", precision=_PRECISION)
    return [DrcViolation("min_space", _layer(layer), float(min_space), p)
            for p in _real(filled, 10 * _PRECISION)]


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
