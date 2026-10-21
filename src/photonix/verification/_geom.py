"""Shared geometry helpers for the verification package (plain NumPy + gdstk).

* :func:`collect_leaves` walks a layout with the same leaf/composite rules and
  instance names as :func:`photonix.layout.extract_netlist`, but keeps every
  leaf (also the port-less ones: rings, fill tiles, pads) together with its
  placement chain, so polygons and ports can be mapped to the top cell.
* :func:`boundary_loops` turns gdstk's keyhole polygons (holes joined to the
  outline by a zero-width cut) back into separate outline and hole loops.
* :func:`polygon_distance` is the exact minimum distance between two polygon
  boundaries (vertex-to-segment, both ways).
* :func:`kasa_fit` is the algebraic least-squares circle fit of Kasa (IEEE
  Trans. Instrum. Meas. IM-25, 8, 1976).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from photonix.layout.cell import OPTICAL, Cell, Port, Reference, _transform_points
from photonix.layout.gds import _gdstk

__all__: list[str] = []

PRECISION = 1e-4          # µm, the boolean grid of photonix.layout.drc
MARKER_LAYERS: tuple[tuple[int, int], ...] = ((203, 0),)   # fibre-port markers of the component library


def gdstk():
    """The gdstk module (raises an ImportError with an install hint)."""
    return _gdstk()


def layer_tuple(layer: Sequence[int]) -> tuple[int, int]:
    return (int(layer[0]), int(layer[1]))


@dataclass(frozen=True)
class Leaf:
    """A leaf instance of a layout: its cell, placement chain and top-frame ports."""

    name: str
    cell: Cell
    chain: tuple[Reference, ...]
    ports: dict[str, Port]

    def polygons(self, layers: Iterable[tuple[int, int]] | None = None) -> list[tuple[np.ndarray, tuple[int, int]]]:
        """Flattened polygons of the leaf in top-cell coordinates."""
        keep = None if layers is None else {layer_tuple(x) for x in layers}
        out = []
        for pts, lay in self.cell.get_polygons():
            lay = layer_tuple(lay)
            if keep is not None and lay not in keep:
                continue
            out.append((to_top(pts, self.chain), lay))
        return out


def to_top(points: np.ndarray, chain: Sequence[Reference]) -> np.ndarray:
    """Map local points through a placement chain (innermost reference last)."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    for ref in reversed(chain):
        p = _transform_points(p, ref)
    return p


def port_to_top(port: Port, chain: Sequence[Reference]) -> Port:
    for ref in reversed(chain):
        port = port.moved(ref.origin[0], ref.origin[1], ref.rotation, ref.mirror)
    return port


def _is_leaf(cell: Cell) -> bool:
    return cell.model is not None or not cell.references


def collect_leaves(top: Cell) -> list[Leaf]:
    """Every leaf instance of ``top`` (extraction naming), port-less leaves included.

    Polygons drawn directly in a composite cell form a pseudo-instance named
    ``"<path><geometry>"`` without ports. A top cell without references is a
    single leaf named after itself.
    """
    if not top.references:
        return [Leaf(top.name, top, (), {pn: p for pn, p in top.ports.items()})]
    out: list[Leaf] = []

    def walk(cell: Cell, chain: tuple[Reference, ...], prefix: str) -> None:
        if cell.polygons:
            path = prefix[:-1] if prefix else top.name
            out.append(Leaf(f"{path}<geometry>", Cell(f"{cell.name}<geometry>", list(cell.polygons)), chain, {}))
        for i, ref in enumerate(cell.references):
            tag = ref.name or f"ref{i}"
            name = f"{prefix}{tag}"
            sub = ref.cell
            sub_chain = (*chain, ref)
            if not _is_leaf(sub):
                walk(sub, sub_chain, f"{name}.")
                continue
            out.append(Leaf(name, sub, sub_chain, {pn: port_to_top(p, sub_chain) for pn, p in sub.ports.items()}))

    walk(top, (), "")
    return out


def optical_ports(leaf: Leaf, marker_layers: Iterable[tuple[int, int]] = MARKER_LAYERS) -> dict[str, Port]:
    markers = {layer_tuple(m) for m in marker_layers}
    return {pn: p for pn, p in leaf.ports.items() if p.port_type == OPTICAL and layer_tuple(p.layer) not in markers}


def merge(points_list: Sequence[np.ndarray], precision: float = PRECISION) -> list[Any]:
    """Union of polygons given as point arrays (gdstk polygons, keyholes for holes)."""
    gd = gdstk()
    polys = [gd.Polygon(np.asarray(p, dtype=float)) for p in points_list if len(p) >= 3]
    if not polys:
        return []
    return gd.boolean(polys, [], "or", precision=precision)


def signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def boundary_loops(points: np.ndarray, digits: int = 9) -> list[np.ndarray]:
    """Split one (possibly keyhole) polygon into its closed boundary loops.

    Directed edges that occur together with their exact reverse are the
    zero-width cuts by which gdstk joins holes to the outline; they are
    removed and the remaining edges are chained into loops.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 3:
        return []
    key = [tuple(np.round(p, digits)) for p in pts]
    edges = [(key[i], key[(i + 1) % n]) for i in range(n) if key[i] != key[(i + 1) % n]]
    count: dict[tuple, int] = {}
    for e in edges:
        count[e] = count.get(e, 0) + 1
    for e in list(count):
        rev = (e[1], e[0])
        k = min(count[e], count.get(rev, 0))
        if k:
            count[e] -= k
            count[rev] -= k
    remaining: list[tuple] = []
    for e in edges:  # keep the original edge order
        if count.get(e, 0) > 0:
            remaining.append(e)
            count[e] -= 1
    if not remaining:
        return []
    nxt: dict[tuple, list[tuple]] = {}
    for a, b in remaining:
        nxt.setdefault(a, []).append(b)
    loops: list[np.ndarray] = []
    used = 0
    while used < len(remaining):
        start = next(a for a, bs in nxt.items() if bs)
        loop = [start]
        cur = start
        while True:
            b = nxt[cur].pop(0)
            used += 1
            if b == start:
                break
            loop.append(b)
            cur = b
            if not nxt.get(cur):
                break  # pragma: no cover - open chain (malformed input)
        if len(loop) >= 3:
            arr = _drop_collinear(np.asarray(loop, dtype=float))
            if len(arr) >= 3:
                loops.append(arr)
    return loops


def _drop_collinear(loop: np.ndarray, rtol: float = 1e-9) -> np.ndarray:
    """Remove vertices lying on the straight segment between their neighbours.

    Boolean engines attach a hole to its outline at an arbitrary point of an
    edge; that extra vertex is not a vertex of the drawn shape.
    """
    while len(loop) > 3:
        a = loop - np.roll(loop, 1, axis=0)
        b = np.roll(loop, -1, axis=0) - loop
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        dot = np.sum(a * b, axis=1)
        scale = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        flat = (np.abs(cross) <= rtol * scale) & (dot > 0)
        if not flat.any():
            break
        loop = loop[~flat]
    return loop


def all_loops(polygons: Iterable[Any]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p in polygons:
        out.extend(boundary_loops(np.asarray(getattr(p, "points", p))))
    return out


def point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(P, S)`` exact distances from ``P`` points to ``S`` segments ``a -> b``."""
    p = np.asarray(points, dtype=float)[:, None, :]
    ab = (b - a)[None, :, :]
    ap = p - a[None, :, :]
    den = np.sum(ab * ab, axis=-1)
    t = np.where(den > 0, np.sum(ap * ab, axis=-1) / np.where(den > 0, den, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    d = ap - t[..., None] * ab
    return np.sqrt(np.sum(d * d, axis=-1))


def _segments(loop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return loop, np.roll(loop, -1, axis=0)


def polygon_distance(loops_a: Sequence[np.ndarray], loops_b: Sequence[np.ndarray], chunk: int = 2048) -> float:
    """Exact minimum distance between the boundaries of two polygon sets.

    For disjoint polygons this is their edge-to-edge spacing: the minimum
    over the vertices of each set of the distance to the segments of the
    other (the closest pair of two disjoint segments always involves an
    end point).
    """
    def side(src: Sequence[np.ndarray], dst: Sequence[np.ndarray]) -> float:
        verts = np.concatenate([np.asarray(x, float) for x in src])
        aa, bb = zip(*(_segments(np.asarray(x, float)) for x in dst), strict=True)
        a, b = np.concatenate(aa), np.concatenate(bb)
        best = np.inf
        for k in range(0, len(verts), chunk):
            best = min(best, float(point_segment_distance(verts[k:k + chunk], a, b).min()))
        return best

    if not loops_a or not loops_b:
        return float("inf")
    return min(side(loops_a, loops_b), side(loops_b, loops_a))


def distance_to_boundary(point: Sequence[float], loops: Sequence[np.ndarray]) -> float:
    if not loops:
        return float("inf")
    aa, bb = zip(*(_segments(np.asarray(x, float)) for x in loops), strict=True)
    return float(point_segment_distance(np.asarray([point], float), np.concatenate(aa), np.concatenate(bb)).min())


def kasa_fit(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Algebraic (Kasa) circle fit: ``(centre, radius, rms radial residual)``."""
    p = np.asarray(points, dtype=float)
    m = p.mean(axis=0)
    q = p - m
    a = np.column_stack([q, np.ones(len(q))])
    rhs = -(q * q).sum(axis=1)
    sol, *_ = np.linalg.lstsq(a, rhs, rcond=None)
    c = np.array([-sol[0] / 2.0, -sol[1] / 2.0])
    r = float(np.sqrt(max(c @ c - sol[2], 0.0)))
    resid = float(np.sqrt(np.mean((np.linalg.norm(q - c, axis=1) - r) ** 2)))
    return c + m, r, resid


def line_chord(point: np.ndarray, direction: np.ndarray, loops: Sequence[np.ndarray]) -> tuple[float, float] | None:
    """Interval ``(s_lo, s_hi)`` of the line ``point + s direction`` inside the loops around ``s = 0``."""
    hits: list[float] = []
    p = np.asarray(point, float)
    t = np.asarray(direction, float)
    for loop in loops:
        a, b = _segments(np.asarray(loop, float))
        e = b - a
        den = t[0] * e[:, 1] - t[1] * e[:, 0]
        ok = np.abs(den) > 1e-15
        ap = a - p
        s = np.where(ok, (ap[:, 0] * e[:, 1] - ap[:, 1] * e[:, 0]) / np.where(ok, den, 1.0), np.nan)
        u = np.where(ok, (ap[:, 0] * t[1] - ap[:, 1] * t[0]) / np.where(ok, den, 1.0), np.nan)
        sel = ok & (u >= 0.0) & (u < 1.0)
        hits.extend(float(v) for v in s[sel])
    lo = [s for s in hits if s < 0]
    hi = [s for s in hits if s > 0]
    if not lo or not hi:
        return None
    return max(lo), min(hi)


def inside(points: Sequence[Sequence[float]], polygons: Sequence[Any]) -> tuple[bool, ...]:
    if not polygons:
        return tuple(False for _ in points)
    return tuple(bool(v) for v in gdstk().inside([tuple(map(float, p)) for p in points], list(polygons)))


def bbox_of(points: np.ndarray) -> tuple[float, float, float, float]:
    p = np.asarray(points, float)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def sweep_pairs(boxes: Sequence[tuple[float, float, float, float]], pad: float = 0.0) -> list[tuple[int, int]]:
    """Index pairs of overlapping (padded) boxes by sort-and-sweep on x."""
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    active: list[int] = []
    out: list[tuple[int, int]] = []
    for i in order:
        x0, y0, x1, y1 = boxes[i]
        active = [j for j in active if boxes[j][2] + pad >= x0 - pad]
        for j in active:
            _, by0, _, by1 = boxes[j]
            if by0 - pad <= y1 + pad and y0 - pad <= by1 + pad:
                out.append((min(i, j), max(i, j)))
        active.append(i)
    return out


def box_polygon(center: Sequence[float], half: float) -> np.ndarray:
    x, y = float(center[0]), float(center[1])
    return np.array([(x - half, y - half), (x + half, y - half), (x + half, y + half), (x - half, y + half)])
