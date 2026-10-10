"""Waveguide routing between ports.

* :func:`route` -- a sharp-cornered dogleg (schematic; corners are not
  physical waveguide bends).
* :func:`route_manhattan` -- a Manhattan route whose corners are filleted with
  circular or Euler bends of a given minimum radius, either found
  automatically (straight, L, Z and U topologies) or through explicit corner
  ``waypoints``.

Both return a single-polygon :class:`Cell` with ports ``o1`` (at ``port1``)
and ``o2`` (at ``port2``) facing *into* the ports they connect, and record
``model="route"`` with the drawn centre-line ``length`` and ``n_bends``, so
:func:`~photonix.layout.extract_netlist` simulates the real path.
"""
from __future__ import annotations

import numpy as np

from ._geometry import arc_curve, corner_offset, euler_curve, path_polygon, place_curve, polyline_length
from .cell import Cell, Port

__all__ = ["route", "route_manhattan", "RoutingError"]

_TOL = 1e-9


class RoutingError(ValueError):
    """Raised when no valid route exists for the requested geometry."""


def _route_cell(points, headings, port1: Port, port2: Port, width: float, layer,
                length: float, n_bends: int) -> Cell:
    c = Cell("route", model="route", model_settings={"length": float(length), "n_bends": int(n_bends)})
    c.add_polygon(path_polygon(points, width, headings), layer)
    c.add_port("o1", tuple(np.asarray(port1.center, dtype=float)), (port1.orientation + 180.0) % 360.0,
               width, layer)
    c.add_port("o2", tuple(np.asarray(port2.center, dtype=float)), (port2.orientation + 180.0) % 360.0,
               width, layer)
    return c


def route(port1: Port, port2: Port, *, width: float | None = None, layer=(1, 0)) -> Cell:
    """Route an L/Z manhattan waveguide with *sharp* corners between two ports.

    Produces a :class:`Cell` whose polygon connects ``port1.center`` to
    ``port2.center`` via a midpoint dogleg. Use :func:`route_manhattan` for a
    physical route with bends.

    Examples
    --------
    >>> from photonix.layout import Port, route
    >>> c = route(Port("a", (0, 0), 0), Port("b", (20, 10), 180))
    >>> len(c.polygons), c.model, c.model_settings["length"]
    (1, 'route', 30.0)
    """
    w = port1.width if width is None else width
    p1 = np.asarray(port1.center, dtype=float)
    p2 = np.asarray(port2.center, dtype=float)
    if p1.shape != (2,) or p2.shape != (2,) or not np.all(np.isfinite([p1, p2])):
        raise ValueError("Route port centers must be finite 2-D coordinates.")
    if np.array_equal(p1, p2):
        raise ValueError("Cannot route between two ports at the same position.")
    if p1[0] == p2[0] or p1[1] == p2[1]:
        points = np.array([p1, p2])
    else:
        mid_x = (p1[0] + p2[0]) / 2.0
        points = np.array([p1, (mid_x, p1[1]), (mid_x, p2[1]), p2])
    return _route_cell(points, None, port1, port2, w, layer, polyline_length(points), len(points) - 2)


def _axis_dir(port: Port) -> np.ndarray:
    d = port.direction
    r = np.round(d)
    if np.max(np.abs(d - r)) > 1e-9 or np.sum(np.abs(r)) != 1:
        raise RoutingError(f"route_manhattan needs axis-aligned ports; {port.name!r} faces {port.orientation} deg")
    return r


def _auto_corners(s: np.ndarray, h1: np.ndarray, e: np.ndarray, h2: np.ndarray, d: float) -> list[np.ndarray]:
    """Corner points for the straight/L/Z/U topologies (h2 = arrival heading)."""
    v = e - s
    lat = np.array([-h1[1], h1[0]])
    a = float(v @ h1)
    b = float(v @ lat)
    if np.allclose(h2, h1):
        if abs(b) < _TOL:
            if a > _TOL:
                return []
            raise RoutingError("port2 lies behind port1 on the same axis")
        if a >= 2 * d - _TOL and abs(b) >= 2 * d - _TOL:
            c1 = s + 0.5 * a * h1
            return [c1, c1 + b * lat]
        raise RoutingError(
            f"a Z route needs >= {2 * d:.4g} um both along and across (got {a:.4g}, {abs(b):.4g}); "
            "reduce the radius, use an sbend, or pass waypoints"
        )
    if np.allclose(h2, -h1):
        if abs(b) < 2 * d - _TOL:
            raise RoutingError(f"a U route needs >= {2 * d:.4g} um lateral separation (got {abs(b):.4g})")
        t = max(0.0, a) + d
        c1 = s + t * h1
        return [c1, c1 + b * lat]
    # perpendicular: single corner
    c = s + a * h1
    if a >= d - _TOL and float((e - c) @ h2) >= d - _TOL:
        return [c]
    raise RoutingError("no single-bend L route fits these ports; pass waypoints for a longer route")


def route_manhattan(
    port1: Port,
    port2: Port,
    *,
    radius: float = 5.0,
    bend: str = "circular",
    p: float = 0.5,
    width: float | None = None,
    layer=None,
    waypoints=None,
    npts: int = 48,
) -> Cell:
    """Route from ``port1`` to ``port2`` with filleted Manhattan bends.

    The route leaves ``port1`` along its outward direction and enters
    ``port2`` against its outward direction. Each 90-degree corner is replaced
    by a ``bend`` (``"circular"`` or ``"euler"`` with clothoid fraction ``p``)
    of minimum radius ``radius``; every straight segment must be long enough
    to hold its fillets, otherwise :class:`RoutingError` explains what to
    change.

    Parameters
    ----------
    waypoints
        Optional corner points between the two ports. Consecutive points must
        be axis-aligned and every corner a 90-degree turn. Without waypoints the
        router picks a straight, L (1 bend), Z or U (2 bends) topology.

    Examples
    --------
    >>> from photonix.layout import Port, route_manhattan
    >>> r = route_manhattan(Port("a", (0, 0), 0), Port("b", (30, 20), 180), radius=5.0)
    >>> r.model_settings["n_bends"]
    2
    >>> import math
    >>> L = 30 + 20 - 4 * 5.0 + 2 * (math.pi / 2 * 5.0)       # straights + two arcs
    >>> abs(r.model_settings["length"] - L) < 1e-9
    True
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if bend == "circular":
        curve = arc_curve(radius, 90.0, npts)
        bend_len = 0.5 * np.pi * radius
    elif bend == "euler":
        curve = euler_curve(radius, 90.0, p, npts)
        bend_len = 0.5 * np.pi * radius * (1.0 + p)
    else:
        raise ValueError("bend must be 'circular' or 'euler'")
    d = corner_offset(curve)
    w = port1.width if width is None else float(width)
    lay = port1.layer if layer is None else layer

    s = np.asarray(port1.center, dtype=float)
    e = np.asarray(port2.center, dtype=float)
    h1 = _axis_dir(port1)
    h2 = -_axis_dir(port2)
    if waypoints is None:
        corners = _auto_corners(s, h1, e, h2, d)
    else:
        corners = [np.asarray(wp, dtype=float) for wp in waypoints]
    path = [s, *corners, e]

    # Drop collinear/duplicate points and validate Manhattan turns.
    clean = [path[0]]
    for pt in path[1:]:
        if np.linalg.norm(pt - clean[-1]) < _TOL:
            continue
        if len(clean) >= 2:
            u = clean[-1] - clean[-2]
            v = pt - clean[-1]
            if abs(u[0] * v[1] - u[1] * v[0]) < _TOL * (np.linalg.norm(u) + np.linalg.norm(v)) and u @ v > 0:
                clean[-1] = pt
                continue
        clean.append(pt)
    path = clean
    dirs = []
    for a, b in zip(path[:-1], path[1:], strict=True):
        seg = b - a
        if min(abs(seg[0]), abs(seg[1])) > _TOL:
            raise RoutingError(f"segment {a} -> {b} is not axis-aligned")
        dirs.append(seg / np.linalg.norm(seg))
    if not np.allclose(dirs[0], h1):
        raise RoutingError("the first segment must leave port1 along its orientation")
    if not np.allclose(dirs[-1], h2):
        raise RoutingError("the last segment must enter port2 against its orientation")
    for u, v in zip(dirs[:-1], dirs[1:], strict=True):
        if abs(u @ v) > 1e-9:
            raise RoutingError("every corner must be a 90-degree turn (no U-turns in place)")

    n_corner = len(path) - 2
    for i in range(len(path) - 1):
        need = d * ((1 if i > 0 else 0) + (1 if i < len(path) - 2 else 0))
        seg_len = float(np.linalg.norm(path[i + 1] - path[i]))
        if seg_len < need - 1e-9:
            raise RoutingError(
                f"segment {i} is {seg_len:.4g} um but its bends need {need:.4g} um at radius {radius}"
            )

    pts = [path[0]]
    hds = [float(np.arctan2(dirs[0][1], dirs[0][0]))]
    for k in range(1, len(path) - 1):
        u, v = dirs[k - 1], dirs[k]
        start = path[k] - d * u
        heading = float(np.arctan2(u[1], u[0]))
        ccw = (u[0] * v[1] - u[1] * v[0]) > 0
        cp, ch = place_curve(curve, start, heading, mirror=not ccw)
        if np.linalg.norm(start - pts[-1]) > _TOL:
            pts.append(start)
            hds.append(heading)
        pts.extend(cp[1:])
        hds.extend(ch[1:])
    if np.linalg.norm(path[-1] - pts[-1]) > _TOL:
        pts.append(path[-1])
        hds.append(float(np.arctan2(dirs[-1][1], dirs[-1][0])))
    pts_arr = np.asarray(pts)
    pts_arr[-1] = path[-1]  # land exactly on the port

    total = sum(float(np.linalg.norm(b - a)) for a, b in zip(path[:-1], path[1:], strict=True))
    length = total - 2.0 * d * n_corner + bend_len * n_corner
    return _route_cell(pts_arr, np.asarray(hds), port1, port2, w, lay, length, n_corner)
