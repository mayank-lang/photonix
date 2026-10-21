"""Re-extract model parameters from drawn geometry and compare them with ``model_settings``.

Layout generators record the parameters their geometry implies
(:attr:`photonix.layout.Cell.model_settings`), and extraction hands those to
the circuit. :func:`check_parameters` closes the loop by measuring the drawn
polygons independently:

* **length** of a constant-width ribbon (straights, bends, routes, S-bends,
  heaters, modulators) and of a linear taper: ``area / width``. A band of
  half-width ``h`` around a C2 curve whose curvature radius exceeds ``h``
  has area ``2 h L`` exactly (Steiner's tube formula); for a polygonal arc of
  ``N`` segments of angle ``dtheta`` the drawn length is ``R N sin(dtheta)``,
  i.e. a relative deficit ``dtheta^2 / 6``.
* **gap** between two waveguide islands: the exact minimum edge-to-edge
  distance (vertex-to-segment both ways). For a ring drawn with vertices
  every ``dtheta`` the drawn gap exceeds the circle's by at most
  ``(R + w/2)(1 - cos(dtheta/2))``.
* **radius** of a ring (and a circular bend): Kasa algebraic circle fits of
  the outer and inner boundaries, averaged; for a racetrack, fits of the two
  end arcs, whose centre distance is ``length_x``.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from photonix.layout.cell import Cell

from ._geom import (
    MARKER_LAYERS,
    Leaf,
    all_loops,
    collect_leaves,
    inside,
    kasa_fit,
    layer_tuple,
    merge,
    optical_ports,
    polygon_distance,
)
from .connectivity import LvsIssue

__all__ = ["extract_parameters", "check_parameters"]

_CIRCLE_RESID = 1e-6   # rms radial residual (relative to R) below which a boundary is a circle


def _islands(cell: Cell, layer: tuple[int, int]) -> list[tuple[Any, list[np.ndarray]]]:
    pts = [p for p, lay in cell.get_polygons() if layer_tuple(lay) == layer]
    return [(m, all_loops([m])) for m in merge(pts, precision=1e-9)]


def _ring_params(loops: list[np.ndarray]) -> dict[str, float]:
    """Radius (and racetrack straight length) from the boundary loops of a closed ring."""
    if len(loops) < 2:
        return {}
    loops = sorted(loops, key=lambda lp: -abs(_area(lp)))
    outer, inner = loops[0], loops[1]
    ex = float(np.ptp(outer[:, 0]))
    ey = float(np.ptp(outer[:, 1]))
    lx_est = ex - ey
    if lx_est > 1e-3 * ey:
        xmid = 0.5 * (outer[:, 0].min() + outer[:, 0].max())
        margin = 0.025 * ey
        radii, centres = [], []
        for sgn in (1.0, -1.0):
            fits = []
            for lp in (outer, inner):
                sel = sgn * (lp[:, 0] - xmid) >= 0.5 * lx_est + margin
                if sel.sum() < 3:
                    return {}
                fits.append(kasa_fit(lp[sel]))
            radii.append(0.5 * (fits[0][1] + fits[1][1]))
            centres.append(0.5 * (fits[0][0] + fits[1][0]))
        return {"radius_geo": float(np.mean(radii)),
                "length_x_geo": float(np.linalg.norm(centres[0] - centres[1]))}
    co, ro, _ = kasa_fit(outer)
    ci, ri, _ = kasa_fit(inner)
    return {"radius_geo": 0.5 * (ro + ri), "length_x_geo": 0.0}


def _area(loop: np.ndarray) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _arc_radius(loop: np.ndarray, width: float) -> float | None:
    """Radius of a circular-arc ribbon: fit each concentric boundary arc, average."""
    c0, _r0, _res0 = kasa_fit(loop)
    d = np.linalg.norm(loop - c0, axis=1)
    split = 0.5 * (d.min() + d.max())
    outer, inner = loop[d > split], loop[d <= split]
    if len(outer) < 3 or len(inner) < 3:
        return None
    (_co, ro, eo), (_ci, ri, ei) = kasa_fit(outer), kasa_fit(inner)
    rm = 0.5 * (ro + ri)
    if max(eo, ei) > _CIRCLE_RESID * max(rm, 1.0) or abs((ro - ri) - width) > 1e-6 * max(rm, 1.0):
        return None
    return float(rm)


def extract_parameters(leaf: Cell, *, marker_layers: Iterable[tuple[int, int]] = MARKER_LAYERS) -> dict[str, float]:
    """Geometric parameters of one leaf cell, measured from its polygons.

    The waveguide layer is the layer of the cell's (non-marker) optical
    ports. Returned keys, where they apply:

    ``width_geo``
        Mean declared port width used for ``length_geo``.
    ``length_geo``
        ``area / width`` of a single two-port island (ribbon or linear taper).
    ``radius_geo``
        Circle-fit radius of a circular-arc ribbon or of a ring island
        (an island touched by no port).
    ``length_x_geo``
        Straight length of a racetrack ring.
    ``gap_geo``, ``gap2_geo``
        Minimum edge-to-edge distance between the islands of ``o1`` and
        ``o2`` of a coupler, or between a ring and the bus of ``o1``
        (``gap2_geo``: of ``o3``).

    Examples
    --------
    >>> from photonix.layout import components as lc
    >>> p = extract_parameters(lc.directional_coupler(gap=0.2))
    >>> round(p["gap_geo"], 12)
    0.2
    >>> r = extract_parameters(lc.ring_single(radius=10.0, gap=0.2))
    >>> round(r["radius_geo"], 9)
    10.0
    """
    ports = optical_ports(Leaf(leaf.name, leaf, (), dict(leaf.ports)), marker_layers)
    if not ports:
        return {}
    layer = layer_tuple(next(iter(ports.values())).layer)
    ports = {n: p for n, p in ports.items() if layer_tuple(p.layer) == layer}
    islands = _islands(leaf, layer)
    if not islands:
        return {}
    owner: dict[str, int] = {}
    for pn, p in ports.items():
        probe = np.asarray(p.center, float) - 1e-3 * p.direction
        for k, (poly, _loops) in enumerate(islands):
            if inside([probe], [poly])[0]:
                owner[pn] = k
                break
    out: dict[str, float] = {}
    ringed = [k for k in range(len(islands)) if k not in owner.values() and len(islands[k][1]) >= 2]
    if ringed:
        ring_loops = islands[ringed[0]][1]
        out.update(_ring_params(ring_loops))
        for key, pn in (("gap_geo", "o1"), ("gap2_geo", "o3")):
            if pn in owner:
                out[key] = polygon_distance(ring_loops, islands[owner[pn]][1])
        return out
    used = sorted(set(owner.values()))
    if len(ports) == 2 and len(islands) == 1 and len(used) == 1:
        w = float(np.mean([p.width for p in ports.values()]))
        area = float(sum(abs(_area(lp)) * (1 if k == 0 else -1)
                         for k, lp in enumerate(sorted(islands[0][1], key=lambda lp: -abs(_area(lp))))))
        out["width_geo"] = w
        out["length_geo"] = area / w
        widths = {round(p.width, 9) for p in ports.values()}
        if len(widths) == 1 and len(islands[0][1]) == 1:
            r = _arc_radius(islands[0][1][0], w)
            if r is not None:
                out["radius_geo"] = r
        return out
    if len(used) == 2 and "o1" in owner and "o2" in owner and owner["o1"] != owner["o2"]:
        out["gap_geo"] = polygon_distance(islands[owner["o1"]][1], islands[owner["o2"]][1])
    return out


def _implied(model: str, settings: dict[str, Any]) -> dict[str, float]:
    """Geometry implied by a cell's model settings (keys as in :func:`extract_parameters`)."""
    s = {k: v for k, v in settings.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    out: dict[str, float] = {}
    if model == "bend" and "radius" in s and "angle" in s:
        out["length_geo"] = abs(float(s["radius"]) * np.deg2rad(float(s["angle"])))
        out["radius_geo"] = float(s["radius"])
    elif model == "bend_euler" and "radius" in s and "angle" in s:
        out["length_geo"] = abs(float(s["radius"]) * np.deg2rad(float(s["angle"]))) * (1.0 + float(s.get("p", 0.5)))
    elif model in ("evanescent_coupler", "directional_coupler"):
        if "gap" in s:
            out["gap_geo"] = float(s["gap"])
    else:
        if "length" in s:
            out["length_geo"] = float(s["length"])
        for key in ("gap", "gap2", "radius", "length_x"):
            if key in s:
                out[f"{key}_geo"] = float(s[key])
    return out


def check_parameters(top: Cell, *, rtol: float = 2e-3, atol: float = 2e-3,
                     marker_layers: Iterable[tuple[int, int]] = MARKER_LAYERS) -> list[LvsIssue]:
    """Compare every leaf's ``model_settings`` with its re-extracted geometry.

    A setting ``v`` and its measured value ``g`` disagree when
    ``|g - v| > atol + rtol |v|``; each disagreement is a
    ``parameter_mismatch`` issue at the instance's first port with the
    measured value. Settings that geometry does not determine (index
    contrast, ``P_pi``, ...) and leaves whose geometry is not a recognised
    ribbon, coupler or ring are skipped. Bends are compared through their
    implied path length (``R |theta|``, Euler ``R |theta| (1 + p)``).

    Examples
    --------
    >>> from photonix.layout import Cell, components as lc
    >>> top = Cell("top")
    >>> wg = lc.straight(10.0)
    >>> wg.model_settings["length"] = 12.0          # stale setting
    >>> _ = top.add_ref(wg, name="w")
    >>> [(i.kind, i.objects, round(i.value, 9)) for i in check_parameters(top)]
    [('parameter_mismatch', ('w',), 10.0)]
    """
    if not (rtol >= 0 and atol >= 0):
        raise ValueError("rtol and atol must be non-negative")
    issues: list[LvsIssue] = []
    cache: dict[int, dict[str, float]] = {}
    for lf in collect_leaves(top):
        cell = lf.cell
        if not cell.model_settings:
            continue
        expected = _implied(cell.model or cell.name, cell.model_settings)
        if not expected:
            continue
        if id(cell) not in cache:
            cache[id(cell)] = extract_parameters(cell, marker_layers=marker_layers)
        got = cache[id(cell)]
        loc = next(iter(lf.ports.values())).center if lf.ports else (0.0, 0.0)
        for key, want in expected.items():
            if key not in got:
                continue
            g = got[key]
            if abs(g - want) > atol + rtol * abs(want):
                issues.append(LvsIssue(
                    "parameter_mismatch", (float(loc[0]), float(loc[1])), (lf.name,),
                    f"{key[:-4]} drawn as {g:.6g} um but model setting implies {want:.6g} um", float(g)))
    return issues
