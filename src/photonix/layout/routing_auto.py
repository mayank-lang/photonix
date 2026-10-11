"""Automatic single routes for any pair of ports.

:func:`route_auto` picks the topology instead of making the caller choose one:

1. Ports facing each other on one line: a straight.
2. Ports facing each other with a lateral offset: an S-bend
   (:func:`~photonix.layout.route_sbend`) over the whole axial distance, if
   its radius stays above ``radius``.
3. Otherwise the shortest filleted Manhattan path with at most ``max_bends``
   90-degree corners -- this covers the L, Z and U routes of
   :func:`~photonix.layout.route_manhattan` and adds the 3-bend and 4-bend
   loops needed when ``port2`` lies behind ``port1`` or too close for a
   single corner.
4. Ports that are not axis-aligned first turn onto the nearest Manhattan
   heading through a bend of the exact angle. When both ports share an
   orientation modulo 90 degrees, the route is Manhattan in their rotated
   frame instead and needs no such bends.

Topology search
---------------
For a turn sequence (``L``/``R`` per corner) the segment directions are fixed,
so the segment lengths ``l_i`` only have to satisfy one linear equation per
axis, ``sum_i sigma_i l_i = delta``, with ``l_i >= m_i`` (each run must hold
the tangent lengths of the bends at its two ends). The minimum of
``sum_i l_i`` is then closed form: start every run at its minimum and add the
remaining signed slack to the runs that point the needed way (split equally,
which reproduces ``route_manhattan``'s centred Z jog). All ``2^k`` sequences up
to ``max_bends`` are scored by ``routed length + bend_penalty * n_bends`` and the
cheapest one that does not come back within ``width + min_space`` of itself is
drawn.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._route_common import (
    TOL,
    axis_direction,
    check_bend,
    clean_vertices,
    corner_offset_90,
    fillet_path,
    make_bend,
    resolve_style,
    rot90,
    route_cell,
    same_angle,
    self_clearance,
)
from .cell import Cell, Port
from .routing import RoutingError, route_manhattan
from .routing_sbend import sbend_path

__all__ = ["route_auto", "default_bend_penalty"]


def default_bend_penalty(radius: float) -> float:
    """Default cost per bend, ``2 R (2 - pi/2)`` µm of equivalent route length.

    Used as a tie-breaker by :func:`route_auto` and
    :func:`~photonix.layout.route_astar`: among routes of similar length the one
    with fewer bends wins, but no route takes a long detour to save a bend.

    Examples
    --------
    >>> round(default_bend_penalty(5.0), 6)
    4.292037
    """
    return 2.0 * float(radius) * (2.0 - 0.5 * math.pi)


def _turn(h: np.ndarray, t: int) -> np.ndarray:
    return rot90(h) if t > 0 else -rot90(h)


def _axis_solve(sig: list[float], mins: list[float], delta: float) -> list[float] | None:
    """Minimum-total lengths ``l_i >= mins`` with ``sum sig_i l_i = delta`` (``None`` if impossible)."""
    if not sig:
        return [] if abs(delta) <= TOL * max(1.0, abs(delta)) else None
    base = sum(s * m for s, m in zip(sig, mins, strict=True))
    slack = delta - base
    out = list(mins)
    if abs(slack) <= TOL * max(1.0, abs(delta)):
        return out
    want = 1.0 if slack > 0 else -1.0
    idx = [i for i, s in enumerate(sig) if s == want]
    if not idx:
        return None
    for i in idx:
        out[i] += abs(slack) / len(idx)
    return out


def manhattan_corners(start: np.ndarray, h0: np.ndarray, end: np.ndarray, h_end: np.ndarray, *,
                      a_start: float, a_end: float, d: float, max_bends: int) -> list[tuple[list[np.ndarray], float]]:
    """All shortest-run Manhattan corner lists from ``start`` (heading ``h0``) to ``end`` (arriving ``h_end``).

    ``a_start``/``a_end`` are tangent lengths already used at the two ends (0
    for a port, the fix-up bend's tangent otherwise); ``d`` is the 90-degree
    corner offset. Returns ``(corners, polyline_length)`` per feasible turn
    sequence; the corners exclude ``start`` and ``end``.
    """
    out = []
    for k in range(max_bends + 1):
        for turns in itertools.product((1, -1), repeat=k):
            heads = [h0]
            for t in turns:
                heads.append(_turn(heads[-1], t))
            if not np.allclose(heads[-1], h_end):
                continue
            mins = [(a_start if i == 0 else d) + (a_end if i == k else d) for i in range(k + 1)]
            lengths = [0.0] * (k + 1)
            ok = True
            for ax in (0, 1):
                idx = [i for i in range(k + 1) if heads[i][ax] != 0]
                sol = _axis_solve([float(heads[i][ax]) for i in idx], [mins[i] for i in idx],
                                  float(end[ax] - start[ax]))
                if sol is None:
                    ok = False
                    break
                for i, val in zip(idx, sol, strict=True):
                    lengths[i] = val
            if not ok or min(lengths) <= TOL:
                continue
            corners = []
            c = np.asarray(start, dtype=float)
            for i in range(k):
                c = c + lengths[i] * heads[i]
                corners.append(c)
            out.append((corners, float(sum(lengths))))
    return out


@dataclass(frozen=True)
class _Fix:
    """A fix-up bend between a non-Manhattan port and the nearest axis heading."""

    corner: np.ndarray | None   # polyline corner (None: the port is already axis-aligned)
    heading: np.ndarray         # Manhattan heading (frame) after/before the fix-up
    allowance: float            # tangent length the fix-up bend takes from the Manhattan run
    excess: float               # bend length - (t_in + t_out)


def _fix_options(point: np.ndarray, angle_deg: float, radius: float, bend: str, p: float, npts: int,
                 arriving: bool) -> list[_Fix]:
    """Ways to join a port heading ``angle_deg`` (frame) to the Manhattan grid."""
    k = angle_deg / 90.0
    if abs(k - round(k)) < 1e-9:
        th = math.radians(round(k) * 90.0)
        return [_Fix(None, np.round(np.array([math.cos(th), math.sin(th)])), 0.0, 0.0)]
    out = []
    for m in (math.floor(k) * 90.0, (math.floor(k) + 1) * 90.0):
        th = math.radians(m)
        axis = np.round(np.array([math.cos(th), math.sin(th)]))
        port_dir = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))])
        if arriving:   # Manhattan run (axis) -> port heading
            b = make_bend(radius, math.radians(angle_deg - m), bend, p, npts)
            corner = point - b.t_out * port_dir
            out.append(_Fix(corner, axis, b.t_in, b.length - b.t_in - b.t_out))
        else:          # port heading -> Manhattan run (axis)
            b = make_bend(radius, math.radians(m - angle_deg), bend, p, npts)
            corner = point + b.t_in * port_dir
            out.append(_Fix(corner, axis, b.t_out, b.length - b.t_in - b.t_out))
    return out


def plan_routes(port1: Port, port2: Port, *, radius: float, bend: str, p: float, max_bends: int,
                bend_penalty: float, npts: int) -> list[tuple[float, list[np.ndarray]]]:
    """Candidate vertex lists (port1 ... port2, global coordinates), cheapest first."""
    if max_bends < 0 or max_bends > 6:
        raise ValueError("max_bends must lie in [0, 6]")
    d, bl = corner_offset_90(radius, bend, p, npts)
    s = np.asarray(port1.center, dtype=float)
    e = np.asarray(port2.center, dtype=float)
    o1 = port1.orientation % 360.0
    arrive = (port2.orientation + 180.0) % 360.0
    # Route in the ports' own Manhattan frame when they share an orientation modulo 90 degrees.
    rel = (o1 - port2.orientation) % 90.0
    phi = o1 % 90.0 if min(rel, 90.0 - rel) < 1e-9 else 0.0
    if min(phi, 90.0 - phi) < 1e-9:
        phi = 0.0
    c, sn = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    to_frame = np.array([[c, sn], [-sn, c]])
    from_frame = to_frame.T
    ef = to_frame @ (e - s)
    starts = _fix_options(np.zeros(2), o1 - phi, radius, bend, p, npts, arriving=False)
    ends = _fix_options(ef, arrive - phi, radius, bend, p, npts, arriving=True)
    cands = []
    for fs in starts:
        for fe in ends:
            p0 = np.zeros(2) if fs.corner is None else fs.corner
            p1 = ef if fe.corner is None else fe.corner
            for corners, poly in manhattan_corners(p0, fs.heading, p1, fe.heading, a_start=fs.allowance,
                                                   a_end=fe.allowance, d=d, max_bends=max_bends):
                verts_f = [np.zeros(2)] + ([fs.corner] if fs.corner is not None else []) + corners \
                    + ([fe.corner] if fe.corner is not None else []) + [ef]
                n_fix = (fs.corner is not None) + (fe.corner is not None)
                lead = (float(np.linalg.norm(fs.corner)) if fs.corner is not None else 0.0) + \
                    (float(np.linalg.norm(ef - fe.corner)) if fe.corner is not None else 0.0)
                length = poly + lead - len(corners) * (2.0 * d - bl) + fs.excess + fe.excess
                cost = length + bend_penalty * (len(corners) + n_fix)
                verts = [s + from_frame @ v for v in verts_f]
                cands.append((cost, verts))
    cands.sort(key=lambda cv: (round(cv[0], 9), len(cv[1])))
    return cands


def _is_manhattan(verts: list[np.ndarray]) -> bool:
    for a, b in zip(verts[:-1], verts[1:], strict=True):
        if min(abs(b[0] - a[0]), abs(b[1] - a[1])) > TOL:
            return False
    return True


def best_route_path(port1: Port, port2: Port, *, radius: float, bend: str, p: float, max_bends: int,
                    bend_penalty: float, npts: int, clearance: float):
    """The cheapest self-clear filleted path and its vertex list (raises if none)."""
    cands = plan_routes(port1, port2, radius=radius, bend=bend, p=p, max_bends=max_bends,
                        bend_penalty=bend_penalty, npts=npts)
    reasons = []
    for _cost, verts in cands:
        try:
            path = fillet_path(verts, radius, bend=bend, p=p, npts=npts)
        except RoutingError as exc:
            reasons.append(str(exc))
            continue
        if self_clearance(path.points, clearance) is not None:
            reasons.append("route comes back too close to itself")
            continue
        return path, verts
    detail = f" ({reasons[0]})" if reasons else ""
    raise RoutingError(
        f"no route with at most {max_bends} bends of radius {radius:g} fits between these ports{detail}; "
        "increase max_bends, reduce the radius, or move the ports"
    )


def route_auto(
    port1: Port,
    port2: Port,
    *,
    radius: float | None = None,
    bend: str = "euler",
    p: float = 0.5,
    allow_sbend: bool = True,
    max_bends: int = 4,
    bend_penalty: float | None = None,
    min_space: float | None = None,
    width: float | None = None,
    layer: Any = None,
    xs: Any = None,
    npts: int = 48,
) -> Cell:
    """Route ``port1`` to ``port2`` choosing the topology automatically.

    Handles every relative placement: straight, S-bend, L, Z, U, the 3-bend
    route for ports too close for one corner, the 4-bend loop for a ``port2``
    behind ``port1``, and ports at any angle (see the module docstring).

    Parameters
    ----------
    radius
        Minimum bend radius (default ``xs.radius`` or 5 µm). S-bends keep their
        radius of curvature at or above it.
    bend, p
        ``"euler"`` (default) or ``"circular"`` corners, Euler fraction ``p``.
        The S-bend uses the same profile.
    allow_sbend
        Use an S-bend for laterally offset facing ports when it fits.
    max_bends
        Most 90-degree corners to consider (0-6).
    bend_penalty
        Extra cost per bend in µm (default :func:`default_bend_penalty`).
    min_space
        Required gap between non-adjacent parts of the route (default: one
        route width).
    xs
        Optional cross-section supplying ``width``, ``layer`` and ``radius``.

    Returns
    -------
    Cell
        ``model="route"`` with the exact centre-line ``length`` and ``n_bends``.

    Examples
    --------
    >>> from photonix.layout import Port, route_auto
    >>> a = Port("a", (0, 0), 0)
    >>> route_auto(a, Port("b", (30, 3), 180), radius=10.0).name      # 3 um offset over 30 um
    'route_sbend'
    >>> loop = route_auto(a, Port("b", (-20, 0), 180), radius=5.0)    # port2 behind port1
    >>> loop.model_settings["n_bends"]
    4
    """
    check_bend(bend, p)
    w, lay, r = resolve_style(port1, width=width, layer=layer, radius=radius, xs=xs)
    bp = default_bend_penalty(r) if bend_penalty is None else float(bend_penalty)
    if bp < 0:
        raise ValueError("bend_penalty must be non-negative")
    clearance = w + (w if min_space is None else float(min_space))
    s = np.asarray(port1.center, dtype=float)
    e = np.asarray(port2.center, dtype=float)
    if np.linalg.norm(e - s) < TOL:
        raise RoutingError("port1 and port2 coincide")
    if same_angle(port2.orientation, port1.orientation + 180.0):
        u = port1.direction
        axial = float((e - s) @ u)
        lateral = float((e - s) @ rot90(u))
        if axial > TOL and abs(lateral) < TOL:
            return route_cell(np.array([s, e]), np.full(2, math.atan2(u[1], u[0])), port1, port2, w, lay,
                              axial, 0, name="route_auto")
        if axial > TOL and allow_sbend:
            try:
                pts, hds, length, nb = sbend_path(s, e, u, kind=bend, p=float(p), min_radius=r,
                                                  start_straight=0.0, span=axial, npts=npts)
            except RoutingError:
                pass
            else:
                return route_cell(pts, hds, port1, port2, w, lay, length, nb, name="route_sbend")
    path, verts = best_route_path(port1, port2, radius=r, bend=bend, p=float(p), max_bends=int(max_bends),
                                  bend_penalty=bp, npts=npts, clearance=clearance)
    verts = clean_vertices(verts)
    if _is_manhattan(verts) and axis_direction(port1) is not None and axis_direction(port2) is not None:
        cell = route_manhattan(port1, port2, radius=r, bend=bend, p=p, width=w, layer=lay,
                               waypoints=verts[1:-1], npts=npts)
        cell.name = "route_auto"
        return cell
    return route_cell(path.points, path.headings, port1, port2, w, lay, path.length, path.n_bends,
                      name="route_auto")
