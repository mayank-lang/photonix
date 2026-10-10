"""Length-matched routes: meanders that bring a route to an exact target length.

:func:`route_length_matched` draws the shortest route between two ports (the
:func:`~photonix.layout.route_auto` Manhattan plan, a minimum-radius S-bend for
small offsets between facing ports, or the given ``waypoints``), then inserts
detours on its longest straight run until the filleted centre-line length
equals ``target_length``.

Rectangular trombones
---------------------
A bump of height ``h`` on a straight run adds ``2 h`` of polyline and four
90-degree corners, each of which shortens the filleted path by
``2 d - L_bend`` (``d`` = corner offset; ``2 R - pi R / 2`` for circular
bends), so

    dL = 2 h - 4 (2 d - L_bend),   h = (dL / n + 4 (2 d - L_bend)) / 2

for ``n`` equal bumps. Legs need ``h >= 2 d`` and bumps are ``2 d`` wide with
``2 d`` gaps, so ``n`` bumps occupy ``4 d n`` of the run plus the tangent
lengths at its ends. ``max_height`` caps ``h`` by adding bumps.

Shallow bumps
-------------
Extra lengths below the smallest trombone (``4 (L_bend - d)``, ``h = 2 d``)
use one trapezoid of four bends of angle ``theta``, with ramps and top each
``2 T(theta)`` long (``T`` = tangent length):

    dL(theta) = 4 T (1 - cos theta) + 4 (L_bend(theta) - 2 T),

which rises continuously from 0 and reaches the trombone minimum at 90
degrees. ``theta`` is found by bisection, so any ``dL > 0`` is reachable.

The final length is checked against the target (``1e-9`` relative), and the
route is checked never to come within ``width + min_space`` of itself.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._route_common import (
    TOL,
    Bend,
    axis_direction,
    check_bend,
    clean_vertices,
    corner_offset_90,
    fillet_path,
    make_bend,
    path_length,
    plan_bends,
    resolve_style,
    rot90,
    route_cell,
    same_angle,
    self_clearance,
)
from .cell import Cell, Port
from .routing import RoutingError, route_manhattan
from .routing_auto import _is_manhattan, best_route_path, default_bend_penalty

__all__ = ["route_length_matched"]


def _min_sbend_vertices(s: np.ndarray, e: np.ndarray, u: np.ndarray, radius: float, bend: str, p: float,
                        npts: int) -> tuple[list[np.ndarray], list[Bend]]:
    """Vertices of the shortest S-bend at ``radius`` (turn angle by bisection), then a straight."""
    n = rot90(u)
    D = float((e - s) @ n)
    sgn = 1.0 if D > 0 else -1.0

    def lateral(th: float) -> float:
        b = make_bend(radius, th, bend, p, npts)
        return (b.t_in + b.t_out) * math.sin(th)

    lo, hi = 1e-9, 0.5 * math.pi
    if lateral(hi) < abs(D):
        raise RoutingError("internal error: offset too large for an S-bend")  # pragma: no cover
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if lateral(mid) < abs(D):
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    th = hi
    b1 = make_bend(radius, sgn * th, bend, p, npts)
    b2 = make_bend(radius, -sgn * th, bend, p, npts)
    c1 = s + b1.t_in * u
    c2 = c1 + (b1.t_out + b2.t_in) * (math.cos(th) * u + sgn * math.sin(th) * n)
    # Land exactly on the port axis: absorb the bisection residual into the ramp direction.
    c2 = c2 + (D - float((c2 - s) @ n)) * n
    return [s, c1, c2, e], [b1, b2]


def _shallow_bump(radius: float, bend: str, p: float, npts: int, dL: float) -> tuple[float, Bend, Bend]:
    """Turn angle of a trapezoid bump adding ``dL``, with its up and down bends (canonical, ccw)."""

    def extra(th: float) -> float:
        b = make_bend(radius, th, bend, p, npts)
        T = 0.5 * (b.t_in + b.t_out)
        return 4.0 * T * (1.0 - math.cos(th)) + 4.0 * (b.length - b.t_in - b.t_out)

    lo, hi = 1e-6, 0.5 * math.pi
    if extra(lo) > dL:
        raise RoutingError(
            f"the requested extra length {dL:.3g} um is below what one bend pair can resolve "
            f"({extra(lo):.3g} um); this is far below fabrication accuracy"
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if extra(mid) < dL:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    th = 0.5 * (lo + hi)
    return th, make_bend(radius, th, bend, p, npts), make_bend(radius, -th, bend, p, npts)


def route_length_matched(
    port1: Port,
    port2: Port,
    *,
    target_length: float,
    radius: float | None = None,
    bend: str = "circular",
    p: float = 0.5,
    n_meanders: int | None = None,
    max_height: float | None = None,
    side: int = 1,
    waypoints=None,
    min_space: float | None = None,
    width: float | None = None,
    layer: Any = None,
    xs: Any = None,
    npts: int = 48,
) -> Cell:
    """Route ``port1`` to ``port2`` with a filleted centre-line length of exactly ``target_length``.

    Parameters
    ----------
    target_length
        Required centre-line length (µm); at least the shortest route's.
    radius, bend, p
        Bend radius (default ``xs.radius`` or 5 µm), ``"circular"`` (default)
        or ``"euler"`` bends, Euler fraction.
    n_meanders
        Number of trombones (default: the fewest that respect ``max_height``).
    max_height
        Largest trombone height (µm); default unlimited.
    side
        ``+1`` puts the meanders on the left of the direction of travel, ``-1``
        on the right; the other side, and other straight runs, are tried if the
        preferred placement would touch the route itself.
    waypoints
        Optional corner points of the base route (any angles).
    min_space
        Gap the route must keep from itself (default one width).
    xs
        Optional cross-section supplying ``width``, ``layer`` and ``radius``.

    Returns
    -------
    Cell
        ``model="route"`` whose ``length`` equals ``target_length``.

    Examples
    --------
    >>> from photonix.layout import Port, route_length_matched
    >>> a, b = Port("a", (0, 0), 0), Port("b", (200, 40), 180)
    >>> r = route_length_matched(a, b, target_length=400.0, radius=5.0)
    >>> abs(r.model_settings["length"] - 400.0) < 1e-9
    True
    """
    check_bend(bend, p)
    if side not in (1, -1):
        raise ValueError("side must be +1 (left) or -1 (right)")
    if n_meanders is not None and int(n_meanders) < 1:
        raise ValueError("n_meanders must be a positive integer")
    if max_height is not None and not float(max_height) > 0:
        raise ValueError("max_height must be positive")
    w, lay, r = resolve_style(port1, width=width, layer=layer, radius=radius, xs=xs)
    target = float(target_length)
    if not math.isfinite(target) or target <= 0:
        raise ValueError("target_length must be positive and finite")
    clearance = w + (w if min_space is None else float(min_space))
    d, bl = corner_offset_90(r, bend, p, npts)
    s = np.asarray(port1.center, dtype=float)
    e = np.asarray(port2.center, dtype=float)

    base_bends: list[Bend] | None = None
    if waypoints is not None:
        verts = clean_vertices([s, *[np.asarray(q, dtype=float) for q in waypoints], e])
    else:
        u = port1.direction
        lateral = float((e - s) @ rot90(u))
        axial = float((e - s) @ u)
        if same_angle(port2.orientation, port1.orientation + 180.0) and TOL <= abs(lateral) < 2 * d - TOL \
                and axial > 0:
            verts, base_bends = _min_sbend_vertices(s, e, u, r, bend, float(p), npts)
        else:
            _path, verts = best_route_path(port1, port2, radius=r, bend=bend, p=float(p), max_bends=4,
                                           bend_penalty=default_bend_penalty(r), npts=npts, clearance=clearance)
            verts = clean_vertices(verts)
    if base_bends is None:
        base_bends = plan_bends(verts, r, bend, float(p), npts)
    L0 = path_length(verts, base_bends)
    dL = target - L0
    if dL < -1e-9 * max(1.0, target):
        raise RoutingError(f"target_length {target:.6g} um is shorter than the shortest route ({L0:.6g} um)")

    def build(vs: list[np.ndarray], bends: list[Bend]) -> Cell:
        path = fillet_path(vs, r, bend=bend, p=float(p), npts=npts, bends=bends)
        if self_clearance(path.points, clearance) is not None:
            raise RoutingError("route comes back too close to itself")
        if _is_manhattan(vs) and axis_direction(port1) is not None and axis_direction(port2) is not None \
                and all(abs(abs(b.angle) - 0.5 * math.pi) < 1e-9 for b in bends):
            cell = route_manhattan(port1, port2, radius=r, bend=bend, p=p, width=w, layer=lay,
                                   waypoints=vs[1:-1], npts=npts)
        else:
            cell = route_cell(path.points, path.headings, port1, port2, w, lay, path.length, path.n_bends)
        cell.name = "route_length_matched"
        got = cell.model_settings["length"]
        if abs(got - target) > 1e-9 * max(1.0, target):
            raise RoutingError(f"internal error: matched length {got!r} != target {target!r}")  # pragma: no cover
        return cell

    if dL <= 1e-12 * max(1.0, target):
        return build(verts, base_bends)

    dL_rect_min = 4.0 * (bl - d)
    th = T = 0.0                             # shallow-bump angle and tangent length (shallow mode only)
    shrink = 4.0 * (2.0 * d - bl)            # length lost to the four fillets of one trombone
    if dL >= dL_rect_min - 1e-12:
        if n_meanders is not None:
            counts = [int(n_meanders)]
        else:
            n_lo = 1
            if max_height is not None:
                per = 2.0 * float(max_height) - shrink
                if per <= dL_rect_min * (1 - 1e-12) or max_height < 2 * d - TOL:
                    raise RoutingError(f"max_height must be at least {2 * d:.6g} um (two corner offsets)")
                n_lo = max(1, math.ceil(dL / per - 1e-12))
            n_hi = max(n_lo, int(math.floor(dL / dL_rect_min + 1e-12)))
            counts = list(range(n_lo, n_hi + 1))
        mode = "rect"
    else:
        if n_meanders not in (None, 1):
            raise RoutingError(f"an extra length of {dL:.4g} um < {dL_rect_min:.4g} um fits only one shallow bump")
        counts = [1]
        mode = "shallow"
        th, b_up, b_dn = _shallow_bump(r, bend, float(p), npts, dL)
        T = 0.5 * (b_up.t_in + b_up.t_out)

    # Straight runs of the base route: (free straight length, index, run length, tangent at start/end).
    runs = []
    for i in range(len(verts) - 1):
        ell = float(np.linalg.norm(verts[i + 1] - verts[i]))
        A = base_bends[i - 1].t_out if i > 0 else 0.0
        B = base_bends[i].t_in if i < len(verts) - 2 else 0.0
        runs.append((ell - A - B, i, ell, A, B))
    by_room = sorted(runs, key=lambda t: -t[0])

    def bumps_on(i: int, n: int, sd: int, h: float) -> tuple[list[np.ndarray], list[Bend]]:
        """Corner points and bends of ``n`` bumps centred on run ``i`` (side ``sd``)."""
        _free, _i, ell, A, B = runs[i]
        a, b = verts[i], verts[i + 1]
        u = (b - a) / ell
        nrm = sd * rot90(u)
        pts: list[np.ndarray] = []
        bends: list[Bend] = []
        if mode == "rect":
            x = A + d + 0.5 * (ell - (A + B + 4.0 * d * n))
            up = make_bend(r, sd * 0.5 * math.pi, bend, float(p), npts)
            dn = make_bend(r, -sd * 0.5 * math.pi, bend, float(p), npts)
            for _k in range(n):
                pts += [a + x * u, a + x * u + h * nrm, a + (x + 2 * d) * u + h * nrm, a + (x + 2 * d) * u]
                bends += [up, dn, dn, up]
                x += 4.0 * d
        else:
            x = A + T + 0.5 * (ell - (A + B + 4.0 * T + 4.0 * T * math.cos(th)))
            ramp = 2.0 * T
            c1 = a + x * u
            c2 = c1 + ramp * (math.cos(th) * u + math.sin(th) * nrm)
            c3 = c2 + ramp * u
            c4 = c3 + ramp * (math.cos(th) * u - math.sin(th) * nrm)
            c4 = a + float((c4 - a) @ u) * u      # back on the run exactly
            pts = [c1, c2, c3, c4]
            up = make_bend(r, sd * th, bend, float(p), npts)
            dn = make_bend(r, -sd * th, bend, float(p), npts)
            bends = [up, dn, dn, up]
        return pts, bends

    def capacity(i: int) -> int:
        _free, _i, ell, A, B = runs[i]
        if mode == "rect":
            return int(math.floor((ell - A - B + TOL) / (4.0 * d)))
        return 1 if ell - A - B >= 4.0 * T + 4.0 * T * math.cos(th) - TOL else 0

    # Placements to try: every run alone (most room first), then the bumps spread over several runs.
    plans: list[tuple[int, dict[int, int]]] = []
    for n in counts:
        for _free, i, _ell, _A, _B in by_room:
            if capacity(i) >= n:
                plans.append((n, {i: n}))
        if mode == "rect":
            alloc: dict[int, int] = {}
            left = n
            for _free, i, _ell, _A, _B in by_room:
                take = min(capacity(i), left)
                if take > 0:
                    alloc[i] = take
                    left -= take
            if left == 0 and len(alloc) > 1:
                plans.append((n, alloc))
    reasons: list[str] = []
    for n, alloc in plans:
        h = 0.5 * (dL / n + shrink) if mode == "rect" else 0.0
        if mode == "rect" and h < 2 * d - 1e-9:
            reasons.append(f"{n} trombones would be {h:.4g} um tall, below the {2 * d:.4g} um minimum")
            continue
        if mode == "rect" and max_height is not None and h > float(max_height) + 1e-9:
            continue
        for sd in (side, -side):
            vs: list[np.ndarray] = [verts[0]]
            bends_all: list[Bend] = []
            for i in range(len(verts) - 1):
                if i > 0:
                    bends_all.append(base_bends[i - 1])
                if i in alloc:
                    pts_i, bends_i = bumps_on(i, alloc[i], sd, h)
                    vs += pts_i
                    bends_all += bends_i
                vs.append(verts[i + 1])
            try:
                return build(vs, bends_all)
            except RoutingError as exc:
                if "internal error" in str(exc):
                    raise
                reasons.append(f"runs {sorted(alloc)}, side {sd:+d}: {exc}")
    if not plans:
        room = max(t[0] for t in runs)
        reasons.append(f"the longest straight run has only {room:.4g} um of free length")
    detail = f" ({reasons[0]})" if reasons else ""
    raise RoutingError(f"could not place meanders adding {dL:.6g} um on the route's straight runs{detail}")
