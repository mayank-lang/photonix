"""Centre-line machinery shared by the advanced routers (private).

* :func:`make_bend` -- a circular or Euler bend of *any* turn angle, with the
  tangent lengths that place it on a corner of a polyline.
* :func:`fillet_path` -- a polyline whose corners (of any angle) are replaced by
  bends; the general form of :func:`~photonix.layout.route_manhattan`'s
  90-degree filleting, used for S-bends, non-Manhattan ports and shallow
  length-matching bumps.
* :func:`route_cell` -- the single-polygon ``model="route"`` cell every router
  returns (same port and settings conventions as ``route_manhattan``).
* :func:`centerline` -- recovers the drawn centre line from a route polygon, so
  clearance checks run on what was drawn, not on what was intended.
* :func:`polyline_distance` / :func:`self_clearance` -- exact segment distances
  and a self-approach test, used to verify that bundles keep their spacing and
  that routes never fold onto themselves.

Everything here is plain NumPy: layout is not in the differentiable path.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._geometry import arc_curve, euler_curve, path_polygon, place_curve
from .cell import Cell, Port
from .routing import RoutingError

__all__: list[str] = []  # private module

TOL = 1e-9
ANGLE_TOL = 1e-9  # radians


# --------------------------------------------------------------------------- #
# Style resolution (width / layer / radius from kwargs, a cross-section or the port)
# --------------------------------------------------------------------------- #
def resolve_style(port: Port, *, width: float | None, layer: Any, radius: float | None, xs: Any,
                  default_radius: float = 5.0) -> tuple[float, tuple[int, int], float]:
    """Explicit kwargs win, then the cross-section ``xs``, then the port (or the default radius)."""
    w = width if width is not None else (xs.width if xs is not None else port.width)
    lay = layer if layer is not None else (xs.layer if xs is not None else port.layer)
    r = radius if radius is not None else (xs.radius if xs is not None else default_radius)
    w, r = float(w), float(r)
    if not (math.isfinite(w) and w > 0):
        raise ValueError(f"route width must be positive and finite, got {w!r}")
    if not (math.isfinite(r) and r > 0):
        raise ValueError(f"bend radius must be positive and finite, got {r!r}")
    return w, (int(lay[0]), int(lay[1])), r


def check_bend(bend: str, p: float) -> None:
    if bend not in ("circular", "euler"):
        raise ValueError(f"bend must be 'circular' or 'euler', got {bend!r}")
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError(f"Euler fraction p must lie in [0, 1], got {p!r}")


# --------------------------------------------------------------------------- #
# Small vector helpers
# --------------------------------------------------------------------------- #
def unit(angle_deg: float) -> np.ndarray:
    th = math.radians(angle_deg)
    return np.array([math.cos(th), math.sin(th)])


def rot90(v: np.ndarray) -> np.ndarray:
    """Left normal (counter-clockwise rotation by 90 degrees)."""
    return np.array([-v[1], v[0]])


def cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def heading_of(v: np.ndarray) -> float:
    return float(math.atan2(v[1], v[0]))


def axis_direction(port: Port, tol: float = 1e-9) -> np.ndarray | None:
    """The port's outward direction rounded to an axis, or ``None`` when it is not axis-aligned."""
    d = port.direction
    r = np.round(d)
    if np.max(np.abs(d - r)) > tol or np.sum(np.abs(r)) != 1:
        return None
    return r


def same_angle(a_deg: float, b_deg: float, tol_deg: float = 1e-6) -> bool:
    return abs(((a_deg - b_deg) + 180.0) % 360.0 - 180.0) <= tol_deg


# --------------------------------------------------------------------------- #
# Bends of arbitrary angle
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Bend:
    """A canonical bend: starts at the origin heading +x and turns counter-clockwise by ``|angle|``.

    ``t_in`` is the distance from the bend start to the intersection of its end
    tangents (the polyline corner), ``t_out`` the distance from that corner to
    the bend end; both are measured on the drawn (discretised) curve so that a
    bend placed by them lands exactly where the next one starts.
    """

    pts: np.ndarray
    hds: np.ndarray
    angle: float      # signed turn angle, radians (positive = counter-clockwise)
    radius: float     # minimum radius of curvature
    t_in: float
    t_out: float
    length: float     # analytic centre-line length


def bend_npts(npts: int, angle: float) -> int:
    """Points for a bend of ``angle`` radians, keeping the per-segment turn of ``npts`` per 90 degrees."""
    return max(4, int(math.ceil(npts * abs(angle) / (0.5 * math.pi))))


def make_bend(radius: float, angle: float, bend: str = "circular", p: float = 0.5, npts: int = 48) -> Bend:
    """A bend of minimum radius ``radius`` turning ``angle`` radians (``0 < |angle| < pi``)."""
    th = abs(float(angle))
    if not ANGLE_TOL < th < math.pi - 1e-6:
        raise RoutingError(f"cannot fillet a {math.degrees(angle):.6g} degree corner (fold-back or no turn)")
    n = bend_npts(npts, th)
    deg = math.degrees(th)
    if bend == "circular" or p == 0.0:
        pts, hds = arc_curve(radius, deg, n)
        length = radius * th
    else:
        pts, hds = euler_curve(radius, deg, p, n)
        length = radius * th * (1.0 + p)
    x, y = float(pts[-1, 0]), float(pts[-1, 1])
    t_out = y / math.sin(th)
    t_in = x - y / math.tan(th) if abs(th - 0.5 * math.pi) > 1e-15 else x
    return Bend(pts, hds, float(angle), float(radius), float(t_in), float(t_out), float(length))


def corner_offset_90(radius: float, bend: str, p: float, npts: int = 48) -> tuple[float, float]:
    """``(d, bend_length)`` of a 90-degree corner -- identical to ``route_manhattan``'s numbers."""
    b = make_bend(radius, 0.5 * math.pi, bend, p, npts)
    return b.t_in, b.length


# --------------------------------------------------------------------------- #
# Filleted polylines
# --------------------------------------------------------------------------- #
def clean_vertices(vertices) -> list[np.ndarray]:
    """Drop duplicate and collinear (same-direction) vertices; reject in-place fold-backs."""
    raw = [np.asarray(v, dtype=float).reshape(2) for v in vertices]
    if len(raw) < 2:
        raise RoutingError("a route needs at least two vertices")
    out = [raw[0]]
    for pt in raw[1:]:
        if np.linalg.norm(pt - out[-1]) < TOL:
            continue
        if len(out) >= 2:
            u = out[-1] - out[-2]
            v = pt - out[-1]
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if abs(cross(u, v)) <= ANGLE_TOL * nu * nv and float(u @ v) > 0:
                out[-1] = pt
                continue
        out.append(pt)
    if len(out) < 2:
        raise RoutingError("route start and end coincide")
    return out


@dataclass(frozen=True)
class FilletedPath:
    """A drawn centre line with exact headings, analytic length and bend count."""

    points: np.ndarray
    headings: np.ndarray
    length: float
    n_bends: int
    vertices: tuple[np.ndarray, ...]
    bends: tuple[Bend, ...]


def plan_bends(vertices: Sequence[np.ndarray], radii, bend: str, p: float, npts: int) -> list[Bend]:
    """The bend for every interior vertex of an already-cleaned vertex list."""
    n = len(vertices)
    if np.ndim(radii) == 0:
        radii = [float(radii)] * max(n - 2, 0)
    radii = list(radii)
    if len(radii) != max(n - 2, 0):
        raise ValueError(f"need one radius per corner ({n - 2}), got {len(radii)}")
    bends = []
    for k in range(1, n - 1):
        u = vertices[k] - vertices[k - 1]
        v = vertices[k + 1] - vertices[k]
        ang = math.atan2(cross(u, v), float(u @ v))
        bends.append(make_bend(radii[k - 1], ang, bend, p, npts))
    return bends


def path_length(vertices: Sequence[np.ndarray], bends: Sequence[Bend]) -> float:
    """Filleted centre-line length: polyline minus the tangent runs plus the analytic bend lengths."""
    poly = sum(float(np.linalg.norm(b - a)) for a, b in zip(vertices[:-1], vertices[1:], strict=True))
    return poly - sum(b.t_in + b.t_out for b in bends) + sum(b.length for b in bends)


def check_room(vertices: Sequence[np.ndarray], bends: Sequence[Bend], radius_note: str = "") -> None:
    """Every straight run must hold the tangent lengths of the bends at its two ends."""
    n = len(vertices)
    for i in range(n - 1):
        need = (bends[i - 1].t_out if i > 0 else 0.0) + (bends[i].t_in if i < n - 2 else 0.0)
        seg = float(np.linalg.norm(vertices[i + 1] - vertices[i]))
        if seg < need - TOL * max(1.0, need):
            raise RoutingError(
                f"segment {i} is {seg:.6g} um but its bends need {need:.6g} um{radius_note}"
            )


def fillet_path(vertices, radii, *, bend: str = "circular", p: float = 0.5, npts: int = 48,
                check: bool = True, bends: Sequence[Bend] | None = None) -> FilletedPath:
    """Replace every corner of ``vertices`` with a bend of the matching angle.

    ``radii`` is one minimum radius for all corners or one per corner. The
    first and last vertices are pinned exactly; straight runs connect the
    bends. Raises :class:`RoutingError` if a run is too short for its bends.
    ``bends`` optionally supplies the bend objects (one per corner) when the
    caller derived the vertices from them, so drawing reuses the exact same
    discretised curves.
    """
    v = clean_vertices(vertices)
    if bends is None:
        bends = plan_bends(v, radii, bend, p, npts)
    else:
        bends = list(bends)
        if len(bends) != len(v) - 2:
            raise RoutingError(f"internal error: {len(bends)} bends for {len(v) - 2} corners")
        for k, b in enumerate(bends, start=1):
            u = v[k] - v[k - 1]
            w = v[k + 1] - v[k]
            ang = math.atan2(cross(u, w), float(u @ w))
            if abs(ang - b.angle) > 1e-7:
                raise RoutingError("internal error: supplied bend does not match its corner angle")
    if check:
        check_room(v, bends)
    dirs = [(b - a) / np.linalg.norm(b - a) for a, b in zip(v[:-1], v[1:], strict=True)]
    pts: list[np.ndarray] = [v[0]]
    hds: list[float] = [heading_of(dirs[0])]
    for k, b in enumerate(bends, start=1):
        u = dirs[k - 1]
        start = v[k] - b.t_in * u
        cp, ch = place_curve((b.pts, b.hds), start, heading_of(u), mirror=b.angle < 0)
        if np.linalg.norm(start - pts[-1]) > TOL:
            pts.append(start)
            hds.append(heading_of(u))
        pts.extend(cp[1:])
        hds.extend(ch[1:])
    if np.linalg.norm(v[-1] - pts[-1]) > TOL:
        pts.append(v[-1])
        hds.append(heading_of(dirs[-1]))
    arr = np.asarray(pts)
    arr[0], arr[-1] = v[0], v[-1]
    return FilletedPath(arr, np.asarray(hds), path_length(v, bends), len(bends), tuple(v), tuple(bends))


def join_pieces(pieces: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate centre-line pieces, dropping points duplicated at the joints."""
    pts = [pieces[0][0][0]]
    hds = [float(pieces[0][1][0])]
    for p_arr, h_arr in pieces:
        for q, h in zip(p_arr, h_arr, strict=True):
            if np.linalg.norm(q - pts[-1]) > TOL:
                pts.append(q)
                hds.append(float(h))
    return np.asarray(pts), np.asarray(hds)


# --------------------------------------------------------------------------- #
# Route cells
# --------------------------------------------------------------------------- #
def route_cell(points, headings, port1: Port, port2: Port, width: float, layer, length: float,
               n_bends: int, name: str = "route") -> Cell:
    """A one-polygon ``model="route"`` cell; ports ``o1``/``o2`` mate with ``port1``/``port2``."""
    c = Cell(name, model="route", model_settings={"length": float(length), "n_bends": int(n_bends)})
    c.add_polygon(path_polygon(points, width, headings), layer)
    c.add_port("o1", tuple(np.asarray(port1.center, dtype=float)), (port1.orientation + 180.0) % 360.0,
               width, layer)
    c.add_port("o2", tuple(np.asarray(port2.center, dtype=float)), (port2.orientation + 180.0) % 360.0,
               width, layer)
    return c


def centerline(cell: Cell) -> np.ndarray:
    """The centre line of a route cell, recovered from its drawn polygon.

    Route polygons are ``path_polygon`` outputs: the left offset of the centre
    line followed by the reversed right offset, so the centre line is the mean
    of the two halves. The recovery is checked (every rung must be one route
    width long) rather than assumed.
    """
    if len(cell.polygons) != 1:
        raise ValueError(f"expected a single-polygon route cell, got {len(cell.polygons)} polygons")
    poly = np.asarray(cell.polygons[0][0], dtype=float)
    n = len(poly) // 2
    if len(poly) != 2 * n or n < 2:
        raise ValueError("route polygon does not have the two-sided path structure")
    left, right = poly[:n], poly[n:][::-1]
    width = cell.ports["o1"].width if "o1" in cell.ports else None
    rung = np.linalg.norm(left - right, axis=1)
    if width is not None and np.max(np.abs(rung - width)) > 1e-6 * max(1.0, width):
        raise ValueError("route polygon is not a constant-width path; cannot recover its centre line")
    return 0.5 * (left + right)


# --------------------------------------------------------------------------- #
# Exact distances between polylines
# --------------------------------------------------------------------------- #
def _point_segment(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = np.sum(ab * ab, axis=-1)
    t = np.where(denom > 0, np.sum((pt - a) * ab, axis=-1) / np.where(denom > 0, denom, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.linalg.norm(pt - proj, axis=-1)


def segment_distances(a0, a1, b0, b1) -> np.ndarray:
    """Pairwise minimum distances between segment arrays ``(m,2)`` and ``(n,2)`` -> ``(m, n)``."""
    A0, A1 = a0[:, None, :], a1[:, None, :]
    B0, B1 = b0[None, :, :], b1[None, :, :]
    d = np.minimum(np.minimum(_point_segment(A0, B0, B1), _point_segment(A1, B0, B1)),
                   np.minimum(_point_segment(B0, A0, A1), _point_segment(B1, A0, A1)))

    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])

    o1, o2 = orient(A0, A1, B0), orient(A0, A1, B1)
    o3, o4 = orient(B0, B1, A0), orient(B0, B1, A1)
    crossing = (o1 * o2 < 0) & (o3 * o4 < 0)
    return np.where(crossing, 0.0, d)


def polyline_distance(p: np.ndarray, q: np.ndarray, chunk: int = 2048) -> float:
    """Exact minimum distance between two polylines (``(m, 2)`` and ``(n, 2)`` vertex arrays)."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    best = math.inf
    qa, qb = q[:-1], q[1:]
    for i in range(0, len(p) - 1, chunk):
        pa, pb = p[i:i + chunk + 1][:-1], p[i:i + chunk + 1][1:]
        best = min(best, float(np.min(segment_distances(pa, pb, qa, qb))))
    return best


def _densify(points: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline so consecutive points are at most ``step`` apart; also return arc length."""
    seg = np.diff(points, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    out = [points[:1]]
    s_out = [np.zeros(1)]
    s0 = 0.0
    for a, d, ell in zip(points[:-1], seg, lens, strict=True):
        n = max(1, int(math.ceil(ell / step)))
        t = np.arange(1, n + 1) / n
        out.append(a + t[:, None] * d)
        s_out.append(s0 + t * ell)
        s0 += ell
    return np.concatenate(out), np.concatenate(s_out)


def self_clearance(points: np.ndarray, threshold: float) -> float | None:
    """Smallest distance at which a centre line comes back near itself, or ``None`` if it never does.

    Two points closer than ``threshold`` count as a self-approach only when
    they are more than ``(pi/2) * threshold`` apart along the curve: for a
    curve whose radius of curvature exceeds ``threshold / 2`` any closer pair is
    just a neighbour on the same stretch. The test runs on a resampled curve
    (step ``threshold / 8``), so it reports approaches to within that step.
    """
    from scipy.spatial import cKDTree

    step = threshold / 8.0
    pts, s = _densify(np.asarray(points, dtype=float), step)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(threshold, output_type="ndarray")
    if len(pairs) == 0:
        return None
    far = np.abs(s[pairs[:, 0]] - s[pairs[:, 1]]) > 0.5 * math.pi * threshold + 2.0 * step
    if not np.any(far):
        return None
    sel = pairs[far]
    return float(np.min(np.linalg.norm(pts[sel[:, 0]] - pts[sel[:, 1]], axis=1)))


def bbox_gap(p: np.ndarray, q: np.ndarray) -> float:
    """Distance between the bounding boxes of two point sets (0 when they overlap)."""
    dx = max(0.0, float(q[:, 0].min() - p[:, 0].max()), float(p[:, 0].min() - q[:, 0].max()))
    dy = max(0.0, float(q[:, 1].min() - p[:, 1].max()), float(p[:, 1].min() - q[:, 1].max()))
    return math.hypot(dx, dy)


def check_bundle_clearance(lines: Sequence[np.ndarray], min_center_distance: float, what: str = "routes") -> None:
    """Raise :class:`RoutingError` if any two centre lines come closer than ``min_center_distance``."""
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if bbox_gap(lines[i], lines[j]) >= min_center_distance:
                continue
            dist = polyline_distance(lines[i], lines[j])
            if dist < min_center_distance - 1e-6:
                raise RoutingError(
                    f"{what} {i} and {j} come within {dist:.4g} um centre-to-centre "
                    f"(need >= {min_center_distance:.4g} um)"
                )
