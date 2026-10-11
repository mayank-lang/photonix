"""Bundle (river) routing of N port pairs at a pitch, and S-bend fan-outs.

:func:`route_bundle` connects ``ports1[i]`` to its partner in ``ports2`` with N
filleted Manhattan routes that never cross; :func:`fanout` moves N parallel
ports to a new pitch with S-bends. Both return ordinary ``model="route"``
cells (one per route), so extraction simulates every drawn length.

Planarity (river routing, Leiserson & Pinter, SIAM J. Comput. 12, 1983)
--------------------------------------------------------------------------
Give every route segment of direction ``u`` the coordinate ``lambda = n . x``
of its line along the left normal ``n`` of ``u``. A set of Manhattan routes
that share their sequence of segment directions is planar exactly when the
routes appear in the same ``lambda`` order on every segment: at a left turn the
leftmost route is the inner one and turns first, at a right turn it is the
outer one and turns last. ``route_bundle`` therefore sorts ``ports1`` by
``lambda`` along their exit heading and ``ports2`` by ``lambda`` along their
arrival heading and pairs them rank by rank (``sort=True``); with
``sort=False`` it checks that the given pairing already has this order and
raises :class:`~photonix.layout.RoutingError` otherwise.

Topologies
----------
* **L** (perpendicular port groups): route ``k`` turns once at the
  intersection of its two port lines, ``lambda1_k n1 + lambda2_k n2``. For
  ports at pitch ``p`` turning through 90 degrees, the corners step by
  ``(-p, p)`` and, with equal radii, the lengths obey ``L_k = L_0 - 2 k p``.
* **Z / S** (parallel groups facing each other): routes offset by at least
  two corner offsets jog in columns ``pitch`` apart -- the route nearest the
  direction of travel jogs first, so jogs nest. Smaller offsets become S-bends
  across the common axial window; zero offsets stay straight.
* **U** (both groups facing the same way): nested U-turns, the outermost
  (longest lateral run) in the farthest column.
* **waypoints**: the bundle follows a Manhattan backbone through the given
  corner points, route ``k`` offset by ``(k - (N - 1)/2) * pitch`` from it.

Every route is drawn by :func:`~photonix.layout.route_manhattan` through its
corner points (or by the S-bend router), and the finished bundle is
*verified*: the exact distance between every pair of drawn centre lines must
be at least ``width + min_space``.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ._route_common import (
    TOL,
    axis_direction,
    centerline,
    check_bend,
    check_bundle_clearance,
    corner_offset_90,
    resolve_style,
    rot90,
    route_cell,
    same_angle,
)
from .cell import Cell, Port
from .routing import RoutingError, route_manhattan
from .routing_sbend import sbend_min_span, sbend_path

__all__ = ["route_bundle", "fanout"]


def _common_orientation(ports: Sequence[Port], what: str) -> float:
    o = ports[0].orientation
    for prt in ports[1:]:
        if not same_angle(prt.orientation, o):
            raise RoutingError(f"all {what} must share one orientation ({prt.name!r} faces "
                               f"{prt.orientation:g}, {ports[0].name!r} faces {o:g} deg)")
    return float(o)


def _min_spacing(lam: np.ndarray) -> float:
    if len(lam) < 2:
        return math.inf
    return float(np.min(np.diff(np.sort(lam))))


def _l_plan(P1, P2, h1, g, n1, n2, d, s0, s1) -> list[tuple]:
    plans = []
    for k, (a, b) in enumerate(zip(P1, P2, strict=True)):
        ca, cb = np.asarray(a.center, dtype=float), np.asarray(b.center, dtype=float)
        corner = float(ca @ n1) * n1 + float(cb @ n2) * n2
        run1 = float((corner - ca) @ h1)
        run2 = float((cb - corner) @ g)
        if run1 < d + s0 - TOL or run2 < d + s1 - TOL:
            raise RoutingError(
                f"L bundle: route {k} ({a.name!r} -> {b.name!r}) has runs of {run1:.4g} and {run2:.4g} um "
                f"but needs {d + s0:.4g} and {d + s1:.4g} um; move the port groups apart or pass waypoints"
            )
        plans.append(("corners", [corner]))
    return plans


def _z_plan(P1, P2, h1, n1, d, pitch, s0, s1, kind, p, r) -> list[tuple]:
    a1 = np.array([float(np.asarray(q.center) @ h1) for q in P1])
    a2 = np.array([float(np.asarray(q.center) @ h1) for q in P2])
    y1 = np.array([float(np.asarray(q.center) @ n1) for q in P1])
    y2 = np.array([float(np.asarray(q.center) @ n1) for q in P2])
    lo, hi = float(a1.max()) + s0, float(a2.min()) - s1
    if hi - lo <= TOL:
        raise RoutingError("ports2 must lie ahead of every port in ports1 (after the start/end straights)")
    delta = y2 - y1
    n = len(P1)
    plans: list[tuple] = [("corners", [])] * n
    # Jog order: in a group moving towards +n1 the route farthest along +n1 jogs first, and
    # vice versa, so every jog happens beside routes that have not jogged yet (nested).
    up = [k for k in range(n - 1, -1, -1) if delta[k] >= TOL]
    down = [k for k in range(n) if delta[k] <= -TOL]
    for group in (up, down):
        if not group:
            continue
        if all(abs(delta[k]) < 2 * d - TOL for k in group):
            # Only S-bends: share the whole axial window (gentlest bends, offsets interpolate monotonically).
            for k in group:
                plans[k] = ("sbend", lo - a1[k], hi - lo)
            continue
        # Mixed or jog-only group: lay the jogs out one after another in jog order. A jog takes a
        # column ``pitch`` from the previous one; an S-bend takes its minimum span, kept clear of the
        # neighbouring columns' fillets by max(pitch, d).
        items: list[tuple[int, str, float, float]] = []   # (route, kind, start, end) of each footprint
        cursor = None
        prev = None
        for k in group:
            if abs(delta[k]) >= 2 * d - TOL:
                x = (lo + d) if cursor is None else cursor + (pitch if prev == "z" else max(pitch, d))
                items.append((k, "z", x, x))
                cursor, prev = x, "z"
            else:
                span = sbend_min_span(delta[k], r, kind=kind, p=p) * (1.0 + 1e-9)
                xs0 = lo if cursor is None else cursor + (max(pitch, d) if prev == "z" else pitch)
                items.append((k, "s", xs0, xs0 + span))
                cursor, prev = xs0 + span, "s"
        first = items[0][2] - (d if items[0][1] == "z" else 0.0)
        last = items[-1][3] + (d if items[-1][1] == "z" else 0.0)
        extent = last - first
        if extent > hi - lo + TOL:
            raise RoutingError(
                f"{len(group)} jogging routes need {extent:.4g} um of axial room between the port groups, "
                f"only {hi - lo:.4g} um is available; move the groups apart or reduce the radius"
            )
        shift = lo - first + 0.5 * (hi - lo - extent)
        for k, kd, x0, x1 in items:
            if kd == "z":
                x = x0 + shift
                plans[k] = ("corners", [x * h1 + y1[k] * n1, x * h1 + y2[k] * n1])
            else:
                plans[k] = ("sbend", x0 + shift - a1[k], x1 - x0)
    return plans


def _u_plan(P1, P2, h1, n1, d, pitch, s0, s1) -> list[tuple]:
    a1 = np.array([float(np.asarray(q.center) @ h1) for q in P1])
    a2 = np.array([float(np.asarray(q.center) @ h1) for q in P2])
    y1 = np.array([float(np.asarray(q.center) @ n1) for q in P1])
    y2 = np.array([float(np.asarray(q.center) @ n1) for q in P2])
    lat = np.abs(y2 - y1)
    if np.any(lat < 2 * d - TOL):
        k = int(np.argmin(lat))
        raise RoutingError(f"U bundle: route {k} turns back over {lat[k]:.4g} um laterally, needs >= {2 * d:.4g} um")
    order = np.argsort(-lat, kind="stable")              # outermost first
    lo_y, hi_y = np.minimum(y1, y2), np.maximum(y1, y2)
    for a, b in zip(order[:-1], order[1:], strict=True):
        if not (lo_y[a] < lo_y[b] - TOL and hi_y[b] < hi_y[a] - TOL):
            raise RoutingError("U bundle: the routes' lateral spans are not nested, so they would cross")
    n = len(P1)
    extra = {int(k): (n - 1 - r) * pitch for r, k in enumerate(order)}
    base = max(max(a1[k] + s0, a2[k] + s1) + d - extra[k] for k in range(n))
    plans = []
    for k in range(n):
        x = base + extra[k]
        plans.append(("corners", [x * h1 + y1[k] * n1, x * h1 + y2[k] * n1]))
    return plans


def _backbone_plan(waypoints, P1, P2, h1, g, d, pitch, s0, s1) -> list[tuple]:
    W = [np.asarray(w, dtype=float).reshape(2) for w in waypoints]
    if len(W) < 2:
        raise RoutingError("a bundle backbone needs at least two waypoints")
    dirs = []
    for a, b in zip(W[:-1], W[1:], strict=True):
        seg = b - a
        if np.linalg.norm(seg) < TOL or min(abs(seg[0]), abs(seg[1])) > TOL:
            raise RoutingError(f"backbone segment {a} -> {b} is not a non-empty axis-aligned run")
        dirs.append(np.round(seg / np.linalg.norm(seg)))
    seg_dirs = [h1, *dirs, g]
    for u, v in zip(seg_dirs[:-1], seg_dirs[1:], strict=True):
        if abs(float(u @ v)) > 1e-9:
            raise RoutingError("every backbone corner, including the first and last, must be a 90-degree turn")
    normals = [rot90(u) for u in seg_dirs]
    lam_bb = [float(rot90(u) @ w) for u, w in zip(dirs, W[:-1], strict=True)]
    n = len(P1)
    plans = []
    for k in range(n):
        ca, cb = np.asarray(P1[k].center, dtype=float), np.asarray(P2[k].center, dtype=float)
        off = (k - 0.5 * (n - 1)) * pitch
        lines = [float(ca @ normals[0])] + [lam + off for lam in lam_bb] + [float(cb @ normals[-1])]
        corners = [lines[j] * normals[j] + lines[j + 1] * normals[j + 1] for j in range(len(seg_dirs) - 1)]
        verts = [ca, *corners, cb]
        last = len(seg_dirs) - 1
        for j, u in enumerate(seg_dirs):
            run = float((verts[j + 1] - verts[j]) @ u)
            need = (d + s0) if j == 0 else (d + s1) if j == last else 2 * d
            if run < need - TOL:
                raise RoutingError(
                    f"bundle route {k} ({P1[k].name!r} -> {P2[k].name!r}): backbone segment {j} runs "
                    f"{run:.4g} um, needs >= {need:.4g} um in its own direction; move the waypoints"
                )
        plans.append(("corners", corners))
    return plans


def route_bundle(
    ports1: Sequence[Port],
    ports2: Sequence[Port],
    *,
    pitch: float | None = None,
    radius: float | None = None,
    bend: str = "euler",
    p: float = 0.5,
    sort: bool = True,
    start_straight: float = 0.0,
    end_straight: float = 0.0,
    waypoints=None,
    min_space: float | None = None,
    width: float | None = None,
    layer: Any = None,
    xs: Any = None,
    npts: int = 48,
) -> list[Cell]:
    """Route N port pairs as a non-crossing bundle.

    All ``ports1`` must share one axis-aligned orientation, and so must all
    ``ports2``; the topology (L, Z/S or U) follows from the two orientations,
    or from a Manhattan ``waypoints`` backbone (see the module docstring).

    Parameters
    ----------
    pitch
        Centre-to-centre spacing of the routes where the router chooses it
        (jog columns, U columns, backbone segments). Default: the smaller of
        the two port groups' minimum spacings.
    radius, bend, p
        Bend radius (default ``xs.radius`` or 5 µm), ``"euler"`` (default) or
        ``"circular"``, and the Euler fraction.
    sort
        Pair the port groups rank by rank so the bundle is planar. With
        ``False``, ``ports1[i]`` connects to ``ports2[i]`` and a non-planar
        pairing raises :class:`~photonix.layout.RoutingError`.
    start_straight, end_straight
        Minimum straight run after ``ports1`` and before ``ports2``.
    waypoints
        Corner points of the bundle's centre line (at least two).
    min_space
        Required edge-to-edge gap between routes; checked on the drawn
        geometry (default 0: routes must merely not overlap). ``pitch`` must
        also be at least ``width + min_space``.
    xs
        Optional cross-section supplying ``width``, ``layer`` and ``radius``.

    Returns
    -------
    list[Cell]
        One ``model="route"`` cell per pair, in the order of ``ports1``.

    Examples
    --------
    >>> from photonix.layout import Port, route_bundle
    >>> ins = [Port(f"i{k}", (0.0, 2.0 * k), 0) for k in range(4)]
    >>> outs = [Port(f"o{k}", (60.0 - 2.0 * k, 50.0), 270) for k in range(4)]
    >>> routes = route_bundle(ins, outs, radius=10.0)
    >>> [round(r.model_settings["length"] - routes[0].model_settings["length"], 9) for r in routes]
    [0.0, -4.0, -8.0, -12.0]
    """
    ports1, ports2 = list(ports1), list(ports2)
    n = len(ports1)
    if n == 0 or n != len(ports2):
        raise ValueError(f"need two equally long, non-empty port lists (got {len(ports1)} and {len(ports2)})")
    check_bend(bend, p)
    w, lay, r = resolve_style(ports1[0], width=width, layer=layer, radius=radius, xs=xs)
    d, _bl = corner_offset_90(r, bend, p, npts)
    _common_orientation(ports1, "ports1")
    _common_orientation(ports2, "ports2")
    h1 = axis_direction(ports1[0])
    out2 = axis_direction(ports2[0])
    if h1 is None or out2 is None:
        raise RoutingError("route_bundle needs axis-aligned ports; use route_auto per pair for other angles")
    g = -out2
    n1, n2 = rot90(h1), rot90(g)
    lam1 = np.array([float(np.asarray(q.center, dtype=float) @ n1) for q in ports1])
    lam2 = np.array([float(np.asarray(q.center, dtype=float) @ n2) for q in ports2])
    sp1, sp2 = _min_spacing(lam1), _min_spacing(lam2)
    if min(sp1, sp2) < TOL:
        raise RoutingError("two ports of one group lie on the same line; a bundle cannot separate them")
    order1 = np.argsort(lam1, kind="stable")
    order2 = np.argsort(lam2, kind="stable")
    if not sort:
        if not np.array_equal(order1, order2):
            raise RoutingError(
                "non-planar pairing: ports1[i] -> ports2[i] would cross; pass sort=True to pair by rank"
            )
    pitch_v = (min(sp1, sp2) if n > 1 else 2.0 * w) if pitch is None else float(pitch)
    space = 0.0 if min_space is None else float(min_space)
    if not (math.isfinite(pitch_v) and pitch_v > w):
        raise ValueError(f"pitch must exceed the route width {w:g} um, got {pitch_v!r}")
    if pitch_v < w + space - TOL:
        raise RoutingError(f"pitch {pitch_v:g} um leaves less than min_space {space:g} um between routes")
    P1 = [ports1[i] for i in order1]
    P2 = [ports2[j] for j in order2]
    s0, s1 = float(start_straight), float(end_straight)
    if s0 < 0 or s1 < 0:
        raise ValueError("start_straight and end_straight must be non-negative")

    if waypoints is not None:
        plans = _backbone_plan(waypoints, P1, P2, h1, g, d, pitch_v, s0, s1)
    elif abs(float(h1 @ g)) < 0.5:
        plans = _l_plan(P1, P2, h1, g, n1, n2, d, s0, s1)
    elif float(h1 @ g) > 0:
        plans = _z_plan(P1, P2, h1, n1, d, pitch_v, s0, s1, bend, float(p), r)
    else:
        plans = _u_plan(P1, P2, h1, n1, d, pitch_v, s0, s1)

    cells: list[Cell] = []
    for k, plan in enumerate(plans):
        a, b = P1[k], P2[k]
        try:
            if plan[0] == "corners":
                cell = route_manhattan(a, b, radius=r, bend=bend, p=p, width=w, layer=lay,
                                       waypoints=plan[1], npts=npts)
            else:
                s = np.asarray(a.center, dtype=float)
                e = np.asarray(b.center, dtype=float)
                pts, hds, length, nb = sbend_path(s, e, h1, kind=bend, p=float(p), min_radius=r,
                                                  start_straight=plan[1], span=plan[2], npts=npts)
                cell = route_cell(pts, hds, a, b, w, lay, length, nb)
        except RoutingError as exc:
            raise RoutingError(f"bundle route {k} ({a.name!r} -> {b.name!r}): {exc}") from exc
        cell.name = "route_bundle"
        cells.append(cell)
    check_bundle_clearance([centerline(c) for c in cells], w + space, what="bundle routes (by rank)")
    out: list[Cell | None] = [None] * n
    for k, i in enumerate(order1):
        out[int(i)] = cells[k]
    return [c for c in out if c is not None]


def fanout(
    ports: Sequence[Port],
    *,
    pitch: float,
    dx: float,
    min_radius: float | None = None,
    kind: str = "euler",
    p: float = 0.5,
    center: float | None = None,
    min_space: float | None = None,
    width: float | None = None,
    layer: Any = None,
    xs: Any = None,
    npts: int = 48,
) -> tuple[list[Cell], list[Port]]:
    """Fan N parallel ports out (or in) to a new ``pitch`` with S-bends.

    Every route runs straight to the plane of the front-most port, then makes
    an S-bend of axial span ``dx`` to its new lateral position. The new
    positions are centred on ``center`` (a coordinate along the ports' left
    normal; default: the middle of the port group) and keep the ports' order.

    Parameters
    ----------
    pitch, dx
        New centre-to-centre spacing and the S-bends' axial span (µm).
    min_radius, kind, p
        S-bend radius bound (default ``xs.radius`` or 5 µm), profile
        (``"euler"``, ``"circular"`` or ``"cosine"``) and Euler fraction.
    min_space
        Required gap between neighbouring routes, checked on the drawn
        geometry (default 0: no overlap).

    Returns
    -------
    (list[Cell], list[Port])
        The routes and the new ports, both in the order of ``ports``; each
        new port faces the same way as the port it continues and carries its
        name, ready to be routed onwards.

    Examples
    --------
    >>> from photonix.layout import Port, fanout
    >>> ports = [Port(f"p{k}", (0.0, 127.0 * k), 0) for k in range(3)]
    >>> routes, new = fanout(ports, pitch=10.0, dx=400.0, min_radius=50.0)
    >>> [tuple(round(v, 6) for v in q.center) for q in new]
    [(400.0, 117.0), (400.0, 127.0), (400.0, 137.0)]
    """
    ports = list(ports)
    if not ports:
        raise ValueError("fanout needs at least one port")
    orient = _common_orientation(ports, "fan-out ports")
    w, lay, r = resolve_style(ports[0], width=width, layer=layer, radius=min_radius, xs=xs)
    pitch_v, dx_v = float(pitch), float(dx)
    space = 0.0 if min_space is None else float(min_space)
    if not (math.isfinite(pitch_v) and pitch_v > w):
        raise ValueError(f"pitch must exceed the route width {w:g} um")
    if pitch_v < w + space - TOL:
        raise RoutingError(f"pitch {pitch_v:g} um leaves less than min_space {space:g} um between routes")
    if not (math.isfinite(dx_v) and dx_v > 0):
        raise ValueError("dx must be positive")
    h = ports[0].direction
    nrm = rot90(h)
    centres = np.array([np.asarray(q.center, dtype=float) for q in ports])
    lam = centres @ nrm
    ax = centres @ h
    if _min_spacing(lam) < TOL:
        raise RoutingError("two fan-out ports lie on the same line")
    order = np.argsort(lam, kind="stable")
    mid = 0.5 * (float(lam.min()) + float(lam.max())) if center is None else float(center)
    target = np.empty(len(ports))
    for k, i in enumerate(order):
        target[i] = mid + (k - 0.5 * (len(ports) - 1)) * pitch_v
    a0 = float(ax.max())
    cells: list[Cell] = []
    new_ports: list[Port] = []
    for i, prt in enumerate(ports):
        end = (a0 + dx_v) * h + target[i] * nrm
        try:
            pts, hds, length, nb = sbend_path(centres[i], end, h, kind=kind, p=float(p), min_radius=r,
                                              start_straight=a0 - float(ax[i]), span=dx_v, npts=npts)
        except RoutingError as exc:
            raise RoutingError(f"fan-out route {prt.name!r}: {exc}") from exc
        dest = Port(prt.name, (float(end[0]), float(end[1])), (orient + 180.0) % 360.0, w, lay)
        cell = route_cell(pts, hds, prt, dest, w, lay, length, nb, name="route_fanout")
        cells.append(cell)
        new_ports.append(Port(prt.name, (float(end[0]), float(end[1])), orient, w, lay, prt.port_type))
    check_bundle_clearance([centerline(cells[i]) for i in order], w + space, what="fan-out routes (by rank)")
    return cells, new_ports
