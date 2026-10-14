"""Obstacle-avoiding Manhattan routing by A* search on a grid.

:func:`route_astar` finds the cheapest filleted Manhattan route between two
axis-aligned ports that keeps ``keepout`` µm of clearance (edge to edge)
from every obstacle polygon, then draws it with
:func:`~photonix.layout.route_manhattan`.

Search (Hart, Nilsson & Raphael, IEEE Trans. SSC 4, 1968; Lee, IRE Trans. EC 10, 1961)
------------------------------------------------------------------------------------
* **Graph.** States are ``(node, heading)`` on a rectilinear grid of spacing
  ``grid``. The grid also contains the exact lines of both port axes and the
  lines that hug every obstacle's bounding box, so routes land exactly on
  the ports and can follow rectangular obstacles at exactly the clearance.
* **Moves.** A straight step to the next node costs its length. A turn at a
  node costs ``bend_penalty`` and is followed by a straight run of at least
  ``2 d`` (``d`` = corner offset of the bend), so every run between two
  corners can hold both fillets; the first corner is at least ``d`` from
  ``port1`` and the last at least ``d`` from ``port2``.
* **Cost.** ``corner-polyline length + bend_penalty * n_bends``. Because
  filleting a corner shortens the path by ``2 d - L_bend``, this equals
  ``routed length + (bend_penalty + 2 d - L_bend) * n_bends``.
* **Heuristic.** Manhattan distance to the goal plus ``bend_penalty`` times
  the fewest turns any route from that state still needs (0 to 4, from the
  goal's position and heading relative to the state). Both terms are lower
  bounds, so the heuristic is admissible and A* returns the same cost as
  Dijkstra's algorithm (``heuristic=False``).

Clearance
---------
Obstacles are rasterised by their exact Euclidean distance at every node. A
straight step is allowed only between nodes at least ``r + grid/2`` from all
obstacles (``r = keepout + width/2`` is the centre-line clearance), which
bounds the distance along the whole step because distance is 1-Lipschitz.
A turn is allowed only if every node within ``grid/sqrt(2)`` of the drawn
fillet curve is at least ``r + grid/sqrt(2)`` away from all obstacles; every
point of the curve has such a node within ``grid/sqrt(2)``, so this covers the
whole drawn bend (and nothing of the corner that the bend never touches). The
finished route is then *verified* exactly: the minimum distance from its drawn
centre line to every obstacle edge must be at least ``r``.

The two straight leads that leave ``port1`` and enter ``port2`` (length
``max(d, r + grid/sqrt(2))``) are checked exactly against every obstacle
except the polygons that touch or contain that port itself (within
``width/2`` of its centre), because a port must abut the device it belongs
to. A lead that runs into any other obstacle raises
:class:`~photonix.layout.RoutingError`; so does the straight route between
two collinear facing ports when it crosses an obstacle that touches neither
port.

Self-crossing
-------------
A grid search cannot see its own path, so when the goal needs a loop the
cheapest grid route may cross itself. Such a result is never returned: the
router then takes the cheapest closed-form topology of
:func:`~photonix.layout.route_auto` (same cost function, up to six bends) that
is self-clear and clears every obstacle, and raises
:class:`~photonix.layout.RoutingError` if there is none. That fallback is
correct but not optimal around obstacles.
"""
from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ._route_common import (
    TOL,
    axis_direction,
    centerline,
    check_bend,
    clean_vertices,
    corner_offset_90,
    fillet_path,
    resolve_style,
    route_cell,
    segment_distances,
    self_clearance,
)
from .cell import Cell, Port
from .routing import RoutingError, route_manhattan
from .routing_auto import default_bend_penalty, plan_routes

__all__ = ["route_astar"]

_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))     # heading index -> grid step
_LINE_TOL = 1e-7


def _heading_index(v: np.ndarray) -> int:
    for k, (dx, dy) in enumerate(_DIRS):
        if v[0] == dx and v[1] == dy:
            return k
    raise RoutingError("internal error: not an axis direction")  # pragma: no cover


def _obstacle_polygons(obstacles, layers) -> list[np.ndarray]:
    if isinstance(obstacles, Cell):
        keep = None if layers is None else {(int(lay[0]), int(lay[1])) for lay in layers}
        polys = [np.asarray(pts, dtype=float) for pts, lay in obstacles.get_polygons()
                 if keep is None or (int(lay[0]), int(lay[1])) in keep]
    else:
        polys = [np.asarray(pts, dtype=float) for pts in obstacles]
    for pts in polys:
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.all(np.isfinite(pts)):
            raise ValueError("every obstacle must be a finite (n >= 3, 2) polygon")
    return polys


def _point_polygon_distance(px: np.ndarray, py: np.ndarray, poly: np.ndarray, chunk: int = 4_000_000) -> np.ndarray:
    """Euclidean distance from points to a polygon region (0 inside; even-odd rule)."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    n_pts = px.size
    out = np.full(n_pts, np.inf)
    inside = np.zeros(n_pts, dtype=bool)
    step = max(1, chunk // max(len(a), 1))
    for s in range(0, n_pts, step):
        x = px[s:s + step, None]
        y = py[s:s + step, None]
        ax, ay, bx, by = a[None, :, 0], a[None, :, 1], b[None, :, 0], b[None, :, 1]
        dx, dy = bx - ax, by - ay
        den = dx * dx + dy * dy
        t = np.clip(((x - ax) * dx + (y - ay) * dy) / np.where(den > 0, den, 1.0), 0.0, 1.0)
        dist = np.hypot(x - (ax + t * dx), y - (ay + t * dy))
        out[s:s + step] = dist.min(axis=1)
        cond = (ay > y) != (by > y)
        xint = ax + (y - ay) * dx / np.where(dy != 0, dy, 1.0)
        inside[s:s + step] = (np.sum(cond & (x < xint), axis=1) % 2) == 1
    out[inside] = 0.0
    return out


def _grid_lines(lo: float, hi: float, anchor: float, step: float, extra: list[float]) -> np.ndarray:
    k0 = math.ceil((lo - anchor) / step)
    k1 = math.floor((hi - anchor) / step)
    base = anchor + step * np.arange(k0, k1 + 1)
    vals = np.sort(np.concatenate([base, np.asarray([v for v in extra if lo - TOL <= v <= hi + TOL])]))
    keep = np.concatenate([[True], np.diff(vals) > _LINE_TOL])
    return vals[keep]


def _min_turns(fwd: float, lat: float, rel: int) -> int:
    """Fewest 90-degree turns from a state to the goal (lower bound, run lengths ignored).

    ``fwd``/``lat`` locate the goal along / to the left of the current heading;
    ``rel`` is the goal heading relative to the current one (0 same, 1 left,
    2 opposite, 3 right).
    """
    eps = 1e-9
    if rel == 0:
        if abs(lat) <= eps and fwd >= -eps:
            return 0
        return 2 if fwd >= -eps else 4
    if rel == 2:
        return 2 if abs(lat) > eps else 4
    if rel == 1:
        return 1 if (fwd >= -eps and lat >= -eps) else 3
    return 1 if (fwd >= -eps and lat <= eps) else 3


def route_astar(
    port1: Port,
    port2: Port,
    *,
    obstacles: Cell | Sequence[np.ndarray],
    keepout: float = 1.0,
    grid: float = 1.0,
    radius: float | None = None,
    bend: str = "circular",
    p: float = 0.5,
    bend_penalty: float | None = None,
    obstacle_layers: Sequence[tuple[int, int]] | None = None,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    width: float | None = None,
    layer: Any = None,
    xs: Any = None,
    heuristic: bool = True,
    max_nodes: int = 2_000_000,
    npts: int = 48,
) -> Cell:
    """Route between two axis-aligned ports around obstacles (A* on a grid).

    Parameters
    ----------
    obstacles
        A :class:`~photonix.layout.Cell` (its flattened polygons, optionally only
        on ``obstacle_layers``) or a list of ``(n, 2)`` polygons.
    keepout
        Required edge-to-edge gap between the route and any obstacle (µm).
    grid
        Grid spacing (µm). Smaller is tighter and slower; the route can hug a
        rectangular obstacle at exactly ``keepout`` but keeps up to
        ``grid/sqrt(2)`` extra from curved or oblique ones.
    radius, bend, p
        Bend radius (default ``xs.radius`` or 5 µm), ``"circular"`` (default)
        or ``"euler"`` bends, Euler fraction.
    bend_penalty
        Cost per bend added to the corner-polyline length (default
        :func:`~photonix.layout.default_bend_penalty`, ``2 R (2 - pi/2)``).
    bounds
        Search window ``((xmin, ymin), (xmax, ymax))``; default: everything
        involved, padded by ``3 d + r`` plus two grid cells.
    heuristic
        ``False`` runs Dijkstra's algorithm (``h = 0``); the cost is the same.
    max_nodes
        Abort after this many state expansions.

    Returns
    -------
    Cell
        A ``route_manhattan`` cell (``model="route"``, exact ``length`` and
        ``n_bends``) whose drawn centre line has been checked against the
        clearance.

    Examples
    --------
    >>> import numpy as np
    >>> from photonix.layout import Port, route_astar
    >>> wall = np.array([(20, -10), (40, -10), (40, 10), (20, 10)], dtype=float)
    >>> r = route_astar(Port("a", (0, 0), 0), Port("b", (60, 0), 180), obstacles=[wall],
    ...                 keepout=1.0, grid=1.0, radius=5.0)
    >>> r.model_settings["n_bends"]
    4
    """
    check_bend(bend, p)
    w, lay, R = resolve_style(port1, width=width, layer=layer, radius=radius, xs=xs)
    keepout, grid = float(keepout), float(grid)
    if not (math.isfinite(keepout) and keepout >= 0):
        raise ValueError("keepout must be non-negative")
    if not (math.isfinite(grid) and grid > 0):
        raise ValueError("grid must be positive")
    bp = default_bend_penalty(R) if bend_penalty is None else float(bend_penalty)
    if not (math.isfinite(bp) and bp >= 0):
        raise ValueError("bend_penalty must be non-negative")
    d, _bl = corner_offset_90(R, bend, p, npts)
    h1v = axis_direction(port1)
    out2 = axis_direction(port2)
    if h1v is None or out2 is None:
        raise RoutingError("route_astar needs axis-aligned ports")
    hgv = -out2
    polys = _obstacle_polygons(obstacles, obstacle_layers)
    r_c = keepout + 0.5 * w                       # centre-line clearance
    r_s = r_c + 0.5 * grid                        # straight-step node threshold
    r_t = r_c + grid / math.sqrt(2.0)             # turn-box node threshold
    lead = max(d, r_t)
    s = np.asarray(port1.center, dtype=float)
    e = np.asarray(port2.center, dtype=float)
    S = s + lead * h1v
    G = e - lead * hgv
    # Polygons a port touches or lies in belong to its device: only they are exempt along that port's lead.
    ex1 = _touching(polys, s, 0.5 * w)
    ex2 = _touching(polys, e, 0.5 * w)
    exempt = (ex1, ex2)

    # Collinear facing ports whose leads overlap: the straight route is all lead.
    if np.allclose(hgv, h1v):
        fwd = float((e - s) @ h1v)
        lat = float((e - s) @ np.array([-h1v[1], h1v[0]]))
        if abs(lat) < TOL and 0 < fwd <= 2 * lead + TOL:
            line = np.array([s, e])
            gap = _line_gap(line, [q for k, q in enumerate(polys) if k not in ex1 and k not in ex2], r_c)
            if gap is not None:
                raise RoutingError(
                    f"the straight route between the collinear facing ports passes {gap:.4g} um from an "
                    f"obstacle (centre-line clearance {r_c:.4g} um)"
                )
            return route_cell(line, np.full(2, math.atan2(h1v[1], h1v[0])), port1, port2,
                              w, lay, fwd, 0, name="route_astar")

    for name, origin, tip, own in (("port1's lead-out", s, S, ex1), ("port2's lead-in", e, G, ex2)):
        gap = _line_gap(np.array([origin, tip]), [q for k, q in enumerate(polys) if k not in own], r_c)
        if gap is not None:
            raise RoutingError(
                f"{name} (the straight {lead:.4g} um before the first bend) passes {gap:.4g} um from an "
                f"obstacle (centre-line clearance {r_c:.4g} um)"
            )

    # ---- grid ---------------------------------------------------------------
    pts_all = [s, e, S, G] + [q for poly in polys for q in poly]
    arr = np.asarray(pts_all, dtype=float)
    pad = 3.0 * d + r_t + 2.0 * grid
    if bounds is None:
        (x0, y0), (x1, y1) = arr.min(axis=0) - pad, arr.max(axis=0) + pad
    else:
        (x0, y0), (x1, y1) = bounds
        for q in (s, e, S, G):
            if not (x0 <= q[0] <= x1 and y0 <= q[1] <= y1):
                raise ValueError("bounds must contain both ports and their lead-out points")
    ex = [s[0], e[0], S[0], G[0]]
    ey = [s[1], e[1], S[1], G[1]]
    for poly in polys:
        (px0, py0), (px1, py1) = poly.min(axis=0), poly.max(axis=0)
        ex += [px0 - r_s, px1 + r_s]
        ey += [py0 - r_s, py1 + r_s]
    xs_ = _grid_lines(float(x0), float(x1), float(s[0]), grid, ex)
    ys_ = _grid_lines(float(y0), float(y1), float(s[1]), grid, ey)
    nx, ny = len(xs_), len(ys_)
    if nx * ny * 4 > 4 * max_nodes + 4_000_000:
        raise RoutingError(f"the search grid has {nx} x {ny} nodes; use a coarser grid or tighter bounds")

    # ---- obstacle distance field ---------------------------------------------
    dist = np.full((nx, ny), np.inf)
    for poly in polys:
        (px0, py0), (px1, py1) = poly.min(axis=0), poly.max(axis=0)
        i0, i1 = np.searchsorted(xs_, px0 - r_t, "left"), np.searchsorted(xs_, px1 + r_t, "right")
        j0, j1 = np.searchsorted(ys_, py0 - r_t, "left"), np.searchsorted(ys_, py1 + r_t, "right")
        if i0 >= i1 or j0 >= j1:
            continue
        gx, gy = np.meshgrid(xs_[i0:i1], ys_[j0:j1], indexing="ij")
        dd = _point_polygon_distance(gx.ravel(), gy.ravel(), poly).reshape(gx.shape)
        dist[i0:i1, j0:j1] = np.minimum(dist[i0:i1, j0:j1], dd)
    blocked_s = dist < r_s - 1e-9
    blocked_t = dist < r_t - 1e-9

    def index_of(q: np.ndarray) -> tuple[int, int]:
        i = int(np.argmin(np.abs(xs_ - q[0])))
        j = int(np.argmin(np.abs(ys_ - q[1])))
        if abs(xs_[i] - q[0]) > 1e-6 or abs(ys_[j] - q[1]) > 1e-6:
            raise RoutingError("internal error: lead-out point is not on the grid")  # pragma: no cover
        return i, j

    si, sj = index_of(S)
    gi, gj = index_of(G)
    if blocked_s[si, sj]:
        raise RoutingError("port1's lead-out point lies within the keepout of an obstacle")
    if blocked_s[gi, gj]:
        raise RoutingError("port2's lead-in point lies within the keepout of an obstacle")

    # ---- turn feasibility ---------------------------------------------------
    # Every point of a drawn bend has a grid node within grid/sqrt(2), so the bend clears r_c when every
    # node that near the curve is at least r_t from all obstacles. Fast path: the whole corner box (plus
    # one cell) is free (summed-area table). Otherwise only the blocked nodes of the box are tested
    # against the actual fillet curve, lazily and memoised.
    sat = np.zeros((nx + 1, ny + 1), dtype=np.int64)
    sat[1:, 1:] = np.cumsum(np.cumsum(blocked_t, axis=0), axis=1)
    near = grid / math.sqrt(2.0) + 1e-7
    curves: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for sx in (1, -1):
        for sy in (1, -1):
            # Corner at the origin, entering along (-sx, 0) and leaving along (0, sy): bend in the (sx, sy) quadrant.
            arc = fillet_path(np.array([(2 * sx * d, 0.0), (0.0, 0.0), (0.0, 2 * sy * d)]), R, bend=bend, p=p,
                              npts=npts).points[1:-1]
            curves[(sx, sy)] = (arc[:-1], arc[1:])

    def span(coords: np.ndarray, sign: int) -> tuple[np.ndarray, np.ndarray]:
        lo_v = coords - grid if sign > 0 else coords - d - grid
        hi_v = coords + d + grid if sign > 0 else coords + grid
        return (np.searchsorted(coords, lo_v - 1e-9, "left"),
                np.clip(np.searchsorted(coords, hi_v + 1e-9, "right") - 1, 0, len(coords) - 1))

    xspan = {sg: span(xs_, sg) for sg in (1, -1)}
    yspan = {sg: span(ys_, sg) for sg in (1, -1)}
    turn_ok: dict[tuple[int, int], np.ndarray] = {}
    for sx in (1, -1):
        ia, ib = xspan[sx]
        for sy in (1, -1):
            ja, jb = yspan[sy]
            A, B = ia[:, None], ib[:, None] + 1
            C, D = ja[None, :], jb[None, :] + 1
            count = sat[B, D] - sat[A, D] - sat[B, C] + sat[A, C]
            turn_ok[(sx, sy)] = (count == 0) & ~blocked_s
    turn_memo: dict[tuple[tuple[int, int], int, int], bool] = {}

    def turn_free(key: tuple[int, int], i: int, j: int) -> bool:
        """Exact-curve turn test at node (i, j) for a corner box the fast path rejected."""
        mk = (key, i, j)
        hit = turn_memo.get(mk)
        if hit is not None:
            return hit
        ok = not bool(blocked_s[i, j])
        if ok:
            ia, ib = xspan[key[0]]
            ja, jb = yspan[key[1]]
            sub = blocked_t[ia[i]:ib[i] + 1, ja[j]:jb[j] + 1]
            bi, bj = np.nonzero(sub)
            if bi.size:
                pts = np.stack([xs_[ia[i] + bi] - xs_[i], ys_[ja[j] + bj] - ys_[j]], axis=1)
                c0, c1 = curves[key]
                dd = segment_distances(pts, pts, c0, c1)
                ok = not bool(np.any(dd <= near))
        turn_memo[mk] = ok
        return ok

    # ---- straight-run checks and minimum post-turn runs ---------------------
    cx = np.zeros((nx + 1, ny), dtype=np.int64)
    cx[1:, :] = np.cumsum(blocked_s, axis=0)
    cy = np.zeros((nx, ny + 1), dtype=np.int64)
    cy[:, 1:] = np.cumsum(blocked_s, axis=1)
    run = 2.0 * d
    tgt = {
        0: np.searchsorted(xs_, xs_ + run - 1e-9, "left"),
        2: np.searchsorted(xs_, xs_ - run + 1e-9, "right") - 1,
        1: np.searchsorted(ys_, ys_ + run - 1e-9, "left"),
        3: np.searchsorted(ys_, ys_ - run + 1e-9, "right") - 1,
    }
    tgt_l = {k: v.tolist() for k, v in tgt.items()}
    xs_l, ys_l = xs_.tolist(), ys_.tolist()
    bs = blocked_s.tolist()
    cx_l, cy_l = cx.tolist(), cy.tolist()
    tok = {k: v.tolist() for k, v in turn_ok.items()}

    def run_free(i: int, j: int, h: int, i2: int, j2: int) -> bool:
        """No blocked node strictly after (i, j) up to and including (i2, j2) along heading h."""
        if h == 0:
            return cx_l[i2 + 1][j] - cx_l[i + 1][j] == 0
        if h == 2:
            return cx_l[i][j] - cx_l[i2][j] == 0
        if h == 1:
            return cy_l[i][j2 + 1] - cy_l[i][j + 1] == 0
        return cy_l[i][j] - cy_l[i][j2] == 0

    h_start = _heading_index(h1v)
    h_goal = _heading_index(hgv)
    xg, yg = xs_l[gi], ys_l[gj]

    def hfun(i: int, j: int, h: int) -> float:
        if not heuristic:
            return 0.0
        dx, dy = xg - xs_l[i], yg - ys_l[j]
        ux, uy = _DIRS[h]
        fwd = dx * ux + dy * uy
        lat = -dx * uy + dy * ux
        return abs(dx) + abs(dy) + bp * _min_turns(fwd, lat, (h_goal - h) % 4)

    n_states = nx * ny * 4
    gcost = np.full(n_states, np.inf)
    parent = np.full(n_states, -1, dtype=np.int64)

    def sid(i: int, j: int, h: int) -> int:
        return (i * ny + j) * 4 + h

    start = sid(si, sj, h_start)
    goal = sid(gi, gj, h_goal)
    gcost[start] = 0.0
    heap: list[tuple[float, float, int]] = [(hfun(si, sj, h_start), 0.0, start)]

    def relax(frm: int, i: int, j: int, h: int, cost: float) -> None:
        st = sid(i, j, h)
        if cost < gcost[st] - 1e-12:
            gcost[st] = cost
            parent[st] = frm
            heapq.heappush(heap, (cost + hfun(i, j, h), cost, st))

    expanded = 0
    found = False
    while heap:
        _f, gc, st = heapq.heappop(heap)
        if gc > gcost[st] + 1e-12:
            continue
        if st == goal:
            found = True
            break
        expanded += 1
        if expanded > max_nodes:
            raise RoutingError(f"A* expanded more than max_nodes={max_nodes} states without reaching port2")
        node, h = divmod(st, 4)
        i, j = divmod(node, ny)
        di, dj = _DIRS[h]
        ni, nj = i + di, j + dj
        if 0 <= ni < nx and 0 <= nj < ny and not bs[ni][nj]:
            relax(st, ni, nj, h, gc + abs(xs_l[ni] - xs_l[i]) + abs(ys_l[nj] - ys_l[j]))
        for h2 in ((h + 1) % 4, (h + 3) % 4):
            ux, uy = _DIRS[h]
            vx, vy = _DIRS[h2]
            key = (vx - ux, vy - uy)
            if not tok[key][i][j] and not turn_free(key, i, j):
                continue
            if i == gi and j == gj and h2 == h_goal:          # corner on the goal itself
                relax(st, i, j, h2, gc + bp)
                continue
            # The goal on this ray, nearer than the minimum post-turn run: land on it directly.
            if h2 == h_goal and ((vx != 0 and j == gj and (gi - i) * vx > 0) or
                                 (vy != 0 and i == gi and (gj - j) * vy > 0)):
                dist_g = abs(xs_l[gi] - xs_l[i]) + abs(ys_l[gj] - ys_l[j])
                if dist_g < run + 1e-9 and run_free(i, j, h2, gi, gj):
                    relax(st, gi, gj, h2, gc + bp + dist_g)
            if vx != 0:
                t = tgt_l[h2][i]
                if 0 <= t < nx and run_free(i, j, h2, t, j):
                    relax(st, t, j, h2, gc + bp + abs(xs_l[t] - xs_l[i]))
            else:
                t = tgt_l[h2][j]
                if 0 <= t < ny and run_free(i, j, h2, i, t):
                    relax(st, i, t, h2, gc + bp + abs(ys_l[t] - ys_l[j]))
    if not found:
        raise RoutingError(
            "no obstacle-free route exists on this grid; enlarge bounds, refine the grid, or reduce "
            "the keepout or radius"
        )

    # ---- reconstruct the corner list ---------------------------------------
    chain = []
    st = goal
    while st != -1:
        chain.append(int(st))
        st = int(parent[st])
    chain.reverse()
    corners: list[tuple[float, float]] = []
    for a, b in zip(chain[:-1], chain[1:], strict=True):
        if a % 4 != b % 4:
            na = a // 4
            corners.append((xs_l[na // ny], ys_l[na % ny]))
    cell = route_manhattan(port1, port2, radius=R, bend=bend, p=p, width=w, layer=lay,
                           waypoints=corners, npts=npts)
    self_gap = w + (keepout if keepout > 0 else w)
    if self_clearance(centerline(cell), self_gap) is not None:
        # The grid search cannot see its own path, so a looping route may cross itself. Fall back
        # to the closed-form topologies of route_auto (same cost function), keeping the cheapest
        # one that is self-clear and clears every obstacle.
        cell = _closed_form_fallback(port1, port2, R=R, bend=bend, p=p, w=w, lay=lay, npts=npts,
                                     bend_penalty=bp + 2.0 * d - _bl, self_gap=self_gap, polys=polys,
                                     r_c=r_c, lead=lead, h1=h1v, hg=hgv, exempt=exempt)
    cell.name = "route_astar"
    gap = _clearance_violation(centerline(cell), polys, r_c, s, h1v, lead, e, hgv, exempt)
    if gap is not None:
        raise RoutingError(  # pragma: no cover - guarded by the rasterisation bounds
            f"internal error: A* route passes {gap:.4g} um from an obstacle (centre-line clearance {r_c:.4g} um)"
        )
    return cell


def _closed_form_fallback(port1: Port, port2: Port, *, R: float, bend: str, p: float, w: float, lay, npts: int,
                          bend_penalty: float, self_gap: float, polys: list[np.ndarray], r_c: float,
                          lead: float, h1: np.ndarray, hg: np.ndarray,
                          exempt: tuple[set[int], set[int]]) -> Cell:
    s = np.asarray(port1.center, dtype=float)
    e = np.asarray(port2.center, dtype=float)
    for _cost, verts in plan_routes(port1, port2, radius=R, bend=bend, p=p, max_bends=6,
                                    bend_penalty=bend_penalty, npts=npts):
        try:
            path = fillet_path(verts, R, bend=bend, p=p, npts=npts)
        except RoutingError:
            continue
        if self_clearance(path.points, self_gap) is not None:
            continue
        if _clearance_violation(path.points, polys, r_c, s, h1, lead, e, hg, exempt) is not None:
            continue
        return route_manhattan(port1, port2, radius=R, bend=bend, p=p, width=w, layer=lay,
                               waypoints=clean_vertices(verts)[1:-1], npts=npts)
    raise RoutingError(
        "the cheapest grid route crosses itself and no closed-form alternative clears the obstacles; "
        "pass intermediate ports (route in two legs) or move the ports"
    )


def _touching(polys: list[np.ndarray], point: np.ndarray, half_width: float) -> set[int]:
    """Indices of the polygons within ``half_width`` of ``point`` (the device a port belongs to)."""
    out = set()
    for k, poly in enumerate(polys):
        dd = float(_point_polygon_distance(np.array([point[0]]), np.array([point[1]]), poly)[0])
        if dd <= half_width + 1e-6:
            out.add(k)
    return out


def _split_lead(line: np.ndarray, origin: np.ndarray, direction: np.ndarray,
                lead: float) -> tuple[np.ndarray, np.ndarray]:
    """Split ``line`` into its straight lead (``lead`` µm from ``origin`` along ``direction``) and the rest.

    The two pieces share their joint vertex.
    """
    nrm = np.array([-direction[1], direction[0]])

    def axis_t(q: np.ndarray) -> float | None:
        rel = q - origin
        return float(rel @ direction) if abs(float(rel @ nrm)) < 1e-7 else None

    i = 0
    while i + 1 < len(line):
        t = axis_t(line[i + 1])
        if t is None or t > lead + 1e-9:
            break
        i += 1
    if i + 1 < len(line) and axis_t(line[i + 1]) is not None:     # lead ends inside a straight segment
        tip = origin + lead * direction
        return np.concatenate([line[:i + 1], [tip]]), np.concatenate([[tip], line[i + 1:]])
    return line[:i + 1], line[i:]


def _line_gap(line: np.ndarray, polys: Sequence[np.ndarray], r_c: float) -> float | None:
    """The smallest distance below ``r_c`` from a polyline to any of ``polys`` (0 inside), or None."""
    if len(line) < 2:
        return None
    a0, a1 = line[:-1], line[1:]
    lo, hi = line.min(axis=0) - r_c, line.max(axis=0) + r_c
    worst = None
    for poly in polys:
        if np.any(poly.max(axis=0) < lo) or np.any(poly.min(axis=0) > hi):
            continue
        dmin = float(np.min(segment_distances(a0, a1, poly, np.roll(poly, -1, axis=0))))
        inside = float(_point_polygon_distance(line[:, 0], line[:, 1], poly).min())
        gap = min(dmin, inside)
        if gap < r_c - 1e-6 and (worst is None or gap < worst):
            worst = gap
    return worst


def _clearance_violation(line: np.ndarray, polys: list[np.ndarray], r_c: float, s: np.ndarray, h1: np.ndarray,
                         lead: float, e: np.ndarray, hg: np.ndarray,
                         exempt: tuple[set[int], set[int]] = (set(), set())) -> float | None:
    """Exact check of a drawn centre line: the smallest obstacle distance below ``r_c``, or None.

    The body is checked against every polygon; each lead only against the polygons its port does not
    touch (``exempt[0]`` for ``port1``, ``exempt[1]`` for ``port2``).
    """
    lead1, rest = _split_lead(np.asarray(line, dtype=float), s, h1, lead)
    lead2, core = _split_lead(rest[::-1], e, -hg, lead)
    gaps = [_line_gap(core[::-1], polys, r_c),
            _line_gap(lead1, [q for k, q in enumerate(polys) if k not in exempt[0]], r_c),
            _line_gap(lead2[::-1], [q for k, q in enumerate(polys) if k not in exempt[1]], r_c)]
    found = [g for g in gaps if g is not None]
    return min(found) if found else None
