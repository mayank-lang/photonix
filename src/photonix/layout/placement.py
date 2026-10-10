"""Placement and floorplanning: grids, rectangle packing, alignment and die-level checks.

* :func:`grid` places cells on a uniform lattice (pitch = largest cell + spacing).
* :func:`pack` bin-packs cells into dies or fields with the MaxRects
  algorithm and the Best-Short-Side-Fit rule (J. Jylänki, "A Thousand Ways to
  Pack the Bin -- A Practical Approach to Two-Dimensional Rectangle Bin
  Packing", 2010).
* :func:`align` and :func:`distribute` line references up by their bounding
  boxes.
* :class:`DieCell` describes a die (outline, dicing lane, keep-out band,
  optical-facet sides) and :func:`check_floorplan` reports blocks outside it,
  geometry in its keep-out, overlapping blocks and edge couplers whose tips
  miss the facet.

Bounding boxes are exact: they come from every transformed vertex, so a
rotated cell's box is the box of its rotated geometry, not a rotated box.
Placement moves references only (no geometry is copied), so the placed
cells keep their models and extraction still sees the drawn devices.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .cell import OPTICAL, Cell, Port, Reference
from .labels import ANCHORS, LabeledCell

__all__ = [
    "grid", "pack", "align", "distribute", "reference_bbox", "DieCell", "FloorplanIssue",
    "check_floorplan", "FACET_SIDES",
]

#: Die sides, as accepted by :class:`DieCell` (``"N"``, ``"E"``, ``"S"``, ``"W"``).
FACET_SIDES = ("N", "E", "S", "W")
_SIDE_ALIASES = {"n": "N", "north": "N", "top": "N", "e": "E", "east": "E", "right": "E",
                 "s": "S", "south": "S", "bottom": "S", "w": "W", "west": "W", "left": "W"}
_SIDE_ORIENTATION = {"E": 0.0, "N": 90.0, "W": 180.0, "S": 270.0}

BBox = tuple[tuple[float, float], tuple[float, float]]


def _side(name: str) -> str:
    key = str(name).lower()
    if key not in _SIDE_ALIASES:
        raise ValueError(f"unknown die side {name!r}; use N/E/S/W (or north/top, ...)")
    return _SIDE_ALIASES[key]


def _transform(pts: np.ndarray, origin, rotation: float, mirror: bool) -> np.ndarray:
    p = np.array(pts, dtype=float, copy=True)
    if mirror:
        p[:, 1] = -p[:, 1]
    th = math.radians(rotation)
    c, s = math.cos(th), math.sin(th)
    out = np.empty_like(p)
    out[:, 0] = c * p[:, 0] - s * p[:, 1] + origin[0]
    out[:, 1] = s * p[:, 0] + c * p[:, 1] + origin[1]
    return out


def _cell_points(cell: Cell) -> np.ndarray:
    polys = cell.get_polygons()
    if not polys:
        return np.zeros((0, 2))
    return np.concatenate([np.asarray(p, dtype=float) for p, _ in polys])


def _box(points: np.ndarray) -> BBox:
    if not len(points):
        return ((0.0, 0.0), (0.0, 0.0))
    lo, hi = points.min(axis=0) + 0.0, points.max(axis=0) + 0.0  # + 0.0: no negative zeros
    return ((float(lo[0]), float(lo[1])), (float(hi[0]), float(hi[1])))


def reference_bbox(ref: Reference) -> BBox:
    """Exact bounding box of a placed reference in its parent's coordinates.

    Examples
    --------
    >>> from photonix.layout import Cell, components as lc
    >>> top = Cell("top")
    >>> r = top.add_ref(lc.straight(10.0), origin=(5.0, 0.0), rotation=90.0)
    >>> (x0, y0), (x1, y1) = reference_bbox(r)
    >>> round(x0, 9), round(x1, 9), round(y1, 9), abs(y0) < 1e-12
    (4.75, 5.25, 10.0, True)
    """
    return _box(_transform(_cell_points(ref.cell), ref.origin, ref.rotation, ref.mirror))


def _anchor(bbox: BBox, anchor: str) -> np.ndarray:
    if anchor not in ANCHORS:
        raise ValueError(f"align must be one of {ANCHORS}, got {anchor!r}")
    (x0, y0), (x1, y1) = bbox
    x = {"w": x0, "e": x1}.get(anchor[-1], 0.5 * (x0 + x1))
    y = {"s": y0, "n": y1}.get(anchor[0], 0.5 * (y0 + y1))
    return np.array([x, y])


def _unique(name: str, used: set[str]) -> str:
    out, k = name, 1
    while out in used:
        out = f"{name}${k}"
        k += 1
    used.add(out)
    return out


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #
def grid(
    cells: Sequence[Cell],
    *,
    shape: tuple[int, int] | None = None,
    spacing: float | tuple[float, float] = (10.0, 10.0),
    align: str = "c",
    names: Sequence[str] | None = None,
    name: str = "grid",
) -> Cell:
    """Place ``cells`` on a uniform rows x columns lattice, in reading order.

    The pitch is the largest cell width (height) plus ``spacing``; slot
    ``(row, col)`` is centred at ``(col * pitch_x, -row * pitch_y)`` (row 0 at
    the top). Each cell's bounding-box ``align`` point (``"c"``, ``"sw"``,
    ``"n"``, ...) is placed on the same point of its slot, so with ``"c"`` the
    box centres lie exactly on the lattice.

    Parameters
    ----------
    shape
        ``(rows, columns)``; default is the most square grid with
        ``columns = ceil(sqrt(n))``.
    names
        Reference names; default ``"<cell name>_<row>_<col>"``.

    Examples
    --------
    >>> from photonix.layout import components as lc
    >>> g = grid([lc.straight(10.0), lc.straight(20.0), lc.straight(5.0)], shape=(1, 3), spacing=5.0)
    >>> [tuple(round(v, 9) for v in r.origin) for r in g.references]       # centres at 0, 25, 50
    [(-5.0, 0.0), (15.0, 0.0), (47.5, 0.0)]
    """
    cells = list(cells)
    n = len(cells)
    if n == 0:
        raise ValueError("grid() needs at least one cell")
    sx, sy = (float(spacing), float(spacing)) if np.isscalar(spacing) else map(float, spacing)  # type: ignore[arg-type]
    if not all(math.isfinite(v) and v >= 0 for v in (sx, sy)):
        raise ValueError("spacing must be non-negative and finite")
    if shape is None:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    else:
        rows, cols = int(shape[0]), int(shape[1])
        if rows < 1 or cols < 1 or rows * cols < n:
            raise ValueError(f"shape {shape} cannot hold {n} cells")
    if names is not None and len(names) != n:
        raise ValueError("names must have one entry per cell")
    boxes = [c.bbox() for c in cells]
    wmax = max(b[1][0] - b[0][0] for b in boxes)
    hmax = max(b[1][1] - b[0][1] for b in boxes)
    px_, py_ = wmax + sx, hmax + sy
    out = Cell(name)
    used: set[str] = set()
    for k, (cell, box) in enumerate(zip(cells, boxes, strict=True)):
        r, c = divmod(k, cols)
        centre = np.array([c * px_, -r * py_])
        slot = ((centre[0] - wmax / 2, centre[1] - hmax / 2), (centre[0] + wmax / 2, centre[1] + hmax / 2))
        origin = _anchor(slot, align) - _anchor(box, align)
        ref_name = names[k] if names is not None else f"{cell.name}_{r}_{c}"
        out.add_ref(cell, origin=(float(origin[0]), float(origin[1])), name=_unique(ref_name, used))
    return out


# --------------------------------------------------------------------------- #
# MaxRects packing
# --------------------------------------------------------------------------- #
class _MaxRects:
    """MaxRects free-rectangle bookkeeping for one bin (Jylänki 2010, section 4)."""

    def __init__(self, width: float, height: float, eps: float, heuristic: str = "bssf"):
        self.free: list[tuple[float, float, float, float]] = [(0.0, 0.0, width, height)]
        self.eps = eps
        self.heuristic = heuristic

    def _score(self, fx: float, fy: float, fw: float, fh: float, rw: float, rh: float) -> tuple[float, ...]:
        left_h, left_v = fw - rw, fh - rh
        short, long_ = min(left_h, left_v), max(left_h, left_v)
        if self.heuristic == "bssf":      # Best Short Side Fit
            return (short, long_, fy, fx)
        if self.heuristic == "blsf":      # Best Long Side Fit
            return (long_, short, fy, fx)
        if self.heuristic == "baf":       # Best Area Fit
            return (fw * fh - rw * rh, short, fy, fx)
        return (fy + rh, fx, fy, fx)      # "bl": Bottom-Left (lowest top edge, then leftmost)

    def find(self, w: float, h: float, rotate: bool):
        """Best free position for a ``w x h`` box under the bin's heuristic (``None`` if it fits nowhere)."""
        best = None
        options = [(w, h, False)] + ([(h, w, True)] if rotate and abs(w - h) > self.eps else [])
        for fx, fy, fw, fh in self.free:
            for rw, rh, rotated in options:
                if rw <= fw + self.eps and rh <= fh + self.eps:
                    score = (*self._score(fx, fy, fw, fh, rw, rh), rotated)
                    if best is None or score < best[0]:
                        best = (score, fx, fy, rw, rh, rotated)
        return None if best is None else best[1:]

    def place(self, x: float, y: float, w: float, h: float) -> None:
        e = self.eps
        out = []
        for fx, fy, fw, fh in self.free:
            if x >= fx + fw - e or x + w <= fx + e or y >= fy + fh - e or y + h <= fy + e:
                out.append((fx, fy, fw, fh))
                continue
            if x > fx + e:
                out.append((fx, fy, x - fx, fh))
            if x + w < fx + fw - e:
                out.append((x + w, fy, fx + fw - x - w, fh))
            if y > fy + e:
                out.append((fx, fy, fw, y - fy))
            if y + h < fy + fh - e:
                out.append((fx, y + h, fw, fy + fh - y - h))

        def inside(a, b) -> bool:
            return a[0] >= b[0] - e and a[1] >= b[1] - e and a[0] + a[2] <= b[0] + b[2] + e \
                and a[1] + a[3] <= b[1] + b[3] + e

        pruned = []
        for i, a in enumerate(out):
            if any(j != i and inside(a, b) and (not inside(b, a) or j < i) for j, b in enumerate(out)):
                continue
            pruned.append(a)
        self.free = pruned


def pack(
    cells: Sequence[Cell],
    *,
    max_size: tuple[float, float],
    spacing: float = 10.0,
    allow_rotation: bool = False,
    heuristic: Literal["bssf", "blsf", "baf", "bl"] = "bssf",
    name: str = "pack",
) -> list[Cell]:
    """Bin-pack cells into as many ``max_size`` bins as needed (MaxRects, Best Short Side Fit by default).

    Each cell's bounding box is inflated by ``spacing`` (half on every side,
    the bin by the same amount so cells may touch its edge), the boxes are
    packed in order of decreasing area (ties keep the input order) and a new
    bin opens when a box fits in none of the open ones. Results are
    deterministic for a given input.

    Parameters
    ----------
    max_size
        Bin ``(width, height)`` in µm. Every placed bounding box lies in
        ``[0, width] x [0, height]``.
    spacing
        Minimum edge-to-edge distance between packed bounding boxes.
    allow_rotation
        Also try each cell rotated by 90 degrees. This rotates its ports too.
    heuristic
        MaxRects placement rule (Jylänki 2010): ``"bssf"`` Best Short Side
        Fit (default, usually the best), ``"blsf"`` Best Long Side Fit,
        ``"baf"`` Best Area Fit or ``"bl"`` Bottom-Left.

    Returns
    -------
    list[Cell]
        One cell per bin (``"<name>_<i>"``), holding references named
        ``"<cell name>_<input index>"``.

    Examples
    --------
    >>> from photonix.layout import Cell
    >>> sq = Cell("sq").add_polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    >>> bins = pack([sq] * 16, max_size=(40.0, 40.0), spacing=0.0)
    >>> len(bins), len(bins[0].references)                  # 16 squares fill the bin exactly
    (1, 16)
    """
    cells = list(cells)
    width, height = float(max_size[0]), float(max_size[1])
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("max_size must be positive and finite")
    if not math.isfinite(spacing) or spacing < 0:
        raise ValueError("spacing must be non-negative and finite")
    if heuristic not in ("bssf", "blsf", "baf", "bl"):
        raise ValueError(f"heuristic must be 'bssf', 'blsf', 'baf' or 'bl', got {heuristic!r}")
    s = float(spacing)
    eps = 1e-9 * max(width, height, 1.0)
    boxes = [c.bbox() for c in cells]
    sizes = [(b[1][0] - b[0][0], b[1][1] - b[0][1]) for b in boxes]
    order = sorted(range(len(cells)), key=lambda k: (-(sizes[k][0] * sizes[k][1]), -max(sizes[k]), k))
    bins: list[_MaxRects] = []
    placed: list[list[tuple[int, float, float, bool]]] = []
    for k in order:
        w, h = sizes[k][0] + s, sizes[k][1] + s
        spot, b_idx = None, -1
        for i, b in enumerate(bins):
            spot = b.find(w, h, allow_rotation)
            if spot is not None:
                b_idx = i
                break
        if spot is None:
            fresh = _MaxRects(width + s, height + s, eps, heuristic)
            spot = fresh.find(w, h, allow_rotation)
            if spot is None:
                raise ValueError(f"cell {cells[k].name!r} ({sizes[k][0]:g} x {sizes[k][1]:g} um) does not fit in "
                                 f"a {width:g} x {height:g} um bin")
            bins.append(fresh)
            placed.append([])
            b_idx = len(bins) - 1
        x, y, rw, rh, rotated = spot
        bins[b_idx].place(x, y, rw, rh)
        placed[b_idx].append((k, x, y, rotated))
    out = []
    for b_idx, items in enumerate(placed):
        cell = Cell(f"{name}_{b_idx}")
        used: set[str] = set()
        for k, x, y, rotated in sorted(items):
            rot = 90.0 if rotated else 0.0
            (bx0, by0), _ = _box(_transform(_cell_points(cells[k]), (0.0, 0.0), rot, False))
            cell.add_ref(cells[k], origin=(x - bx0, y - by0), rotation=rot,
                         name=_unique(f"{cells[k].name}_{k}", used))
        out.append(cell)
    return out


# --------------------------------------------------------------------------- #
# align / distribute
# --------------------------------------------------------------------------- #
def _shift(ref: Reference, dx: float, dy: float) -> None:
    ref.origin = (float(ref.origin[0] + dx), float(ref.origin[1] + dy))


def align(refs: Sequence[Reference], edge: str, *, value: float | None = None) -> list[Reference]:
    """Move references so their bounding boxes share an edge or a centre line.

    ``edge`` is ``"left"``, ``"right"``, ``"bottom"``, ``"top"`` (default
    target: the outermost such edge of the group), or ``"x"`` / ``"y"`` for
    the vertical / horizontal centre line (default target: the first
    reference's). ``value`` overrides the target coordinate. References are
    moved in place and returned.

    Examples
    --------
    >>> from photonix.layout import Cell, components as lc
    >>> top = Cell("top")
    >>> a = top.add_ref(lc.straight(10.0), origin=(3.0, 0.0))
    >>> b = top.add_ref(lc.straight(20.0), origin=(-1.0, 8.0))
    >>> [r.origin[0] for r in align([a, b], "left")]
    [-1.0, -1.0]
    """
    if edge not in ("left", "right", "bottom", "top", "x", "y"):
        raise ValueError("edge must be left, right, bottom, top, x or y")
    refs = list(refs)
    if not refs:
        return refs
    boxes = [reference_bbox(r) for r in refs]
    if edge in ("left", "right", "x"):
        coord = [b[0][0] if edge == "left" else b[1][0] if edge == "right" else 0.5 * (b[0][0] + b[1][0])
                 for b in boxes]
    else:
        coord = [b[0][1] if edge == "bottom" else b[1][1] if edge == "top" else 0.5 * (b[0][1] + b[1][1])
                 for b in boxes]
    if value is None:
        value = {"left": min, "bottom": min, "right": max, "top": max}.get(edge, lambda c: c[0])(coord)
    for ref, c in zip(refs, coord, strict=True):
        if edge in ("left", "right", "x"):
            _shift(ref, float(value) - c, 0.0)
        else:
            _shift(ref, 0.0, float(value) - c)
    return refs


def distribute(refs: Sequence[Reference], axis: str = "x", spacing: float = 0.0) -> list[Reference]:
    """Space references along ``axis`` with ``spacing`` µm between consecutive bounding boxes.

    The first reference stays; each next one is moved along the axis so its
    lower edge sits ``spacing`` above (right of) the previous one's upper
    edge, in the given order. The other coordinate is unchanged.

    Examples
    --------
    >>> from photonix.layout import Cell, components as lc
    >>> top = Cell("top")
    >>> refs = [top.add_ref(lc.straight(L)) for L in (10.0, 5.0, 8.0)]
    >>> [r.origin[0] for r in distribute(refs, "x", spacing=2.0)]
    [0.0, 12.0, 19.0]
    """
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    if not math.isfinite(spacing):
        raise ValueError("spacing must be finite")
    refs = list(refs)
    i = 0 if axis == "x" else 1
    edge = None
    for ref in refs:
        lo, hi = reference_bbox(ref)
        if edge is not None:
            delta = edge + spacing - lo[i]
            _shift(ref, delta if i == 0 else 0.0, delta if i == 1 else 0.0)
            hi = (hi[0] + (delta if i == 0 else 0.0), hi[1] + (delta if i == 1 else 0.0))
        edge = hi[i]
    return refs


# --------------------------------------------------------------------------- #
# Dies and floorplan checks
# --------------------------------------------------------------------------- #
@dataclass(repr=False)
class DieCell(LabeledCell):
    """A die frame that knows its outline, dicing lane, keep-out band and facets.

    The die occupies ``[0, W] x [0, H]`` (``die_size``), the cut line. A
    ``dicing_lane`` wide ring inside it is reserved for the saw (when wafer
    streets are shared between dies, this is the die's half of the street),
    and devices must keep a further ``keepout`` from the lane: the usable area
    is :meth:`usable_box`. On ``facet_sides`` edge couplers may cross both to
    reach the optical facet at :meth:`facet_position`: the die outline, or
    the inner wall of the facet trench when one is drawn
    (``facet_trench_width > 0``). Build one with
    :func:`photonix.layout.library.die_frame`.
    """

    die_size: tuple[float, float] = (5000.0, 5000.0)
    dicing_lane: float = 100.0
    keepout: float = 50.0
    facet_sides: tuple[str, ...] = ()
    facet_trench_width: float = 0.0

    def usable_box(self) -> BBox:
        """``((x0, y0), (x1, y1))`` of the area inside the lane and the keep-out band."""
        m = self.dicing_lane + self.keepout
        w, h = self.die_size
        return ((m, m), (w - m, h - m))

    def facet_position(self, side: str) -> float:
        """Coordinate (x for E/W, y for N/S) of the optical facet on ``side``."""
        side = _side(side)
        w, h = self.die_size
        t = self.facet_trench_width
        return {"E": w - t, "W": t, "N": h - t, "S": t}[side]


@dataclass(frozen=True)
class FloorplanIssue:
    """One floorplan problem found by :func:`check_floorplan`.

    ``kind`` is ``"outside_die"``, ``"keepout"``, ``"overlap"`` or ``"facet"``;
    ``instances`` names the references involved (dotted paths for nested
    leaves) and ``bbox`` locates the problem in die coordinates.
    """

    kind: str
    message: str
    instances: tuple[str, ...]
    bbox: BBox


def _leaves(name: str, cell: Cell, origin, rotation: float, mirror: bool):
    """(path name, cell, origin, rotation, mirror) of every leaf of a placed ``cell``.

    A composite (references and no model) is descended into, as extraction
    does; its own polygons, if any, are reported as a geometry-only leaf under
    the composite's name.
    """
    origin = (float(origin[0]), float(origin[1]))
    if not (cell.references and cell.model is None):
        yield name, cell, origin, rotation % 360.0, mirror
        return
    if cell.polygons:
        yield name, Cell(cell.name, polygons=list(cell.polygons)), origin, rotation % 360.0, mirror
    for i, ref in enumerate(cell.references):
        # compose the parent transform (mirror -> rotate -> translate) with the reference's
        o = _transform(np.array([ref.origin]), origin, rotation, mirror)[0]
        rot = (-ref.rotation if mirror else ref.rotation) + rotation
        yield from _leaves(f"{name}.{ref.name or f'ref{i}'}", ref.cell, o, rot, mirror != ref.mirror)


def _overlap(a: BBox, b: BBox, tol: float) -> BBox | None:
    x0, y0 = max(a[0][0], b[0][0]), max(a[0][1], b[0][1])
    x1, y1 = min(a[1][0], b[1][0]), min(a[1][1], b[1][1])
    return ((x0, y0), (x1, y1)) if x1 - x0 > tol and y1 - y0 > tol else None


def check_floorplan(
    top: Cell,
    die: DieCell,
    *,
    keepout: float | None = None,
    facet_tol: float = 1.0,
    facet_models: Sequence[str] = ("edge_coupler",),
    facet_port: str = "o2",
    include_routes: bool = False,
    tol: float = 1e-6,
) -> list[FloorplanIssue]:
    """Check block placement in ``top`` against ``die``.

    Checks
    ------
    ``outside_die``
        A top-level reference's bounding box leaves ``[0, W] x [0, H]``.
    ``keepout``
        A leaf instance (or ``top``'s own geometry) reaches into the dicing
        lane or the keep-out band, i.e. outside :meth:`DieCell.usable_box`
        (with ``keepout`` overriding the die's). Facet devices (cells whose
        ``model`` is in ``facet_models``) are exempt: they must cross the band.
    ``overlap``
        Two top-level references' bounding boxes overlap by more than
        ``tol`` in both directions. Route cells (``model == "route"``), whose
        boxes span the blocks they join, are skipped unless ``include_routes``.
    ``facet``
        A facet device's ``facet_port`` does not face a die edge, faces a
        side that is not in ``die.facet_sides``, or lies more than
        ``facet_tol`` from :meth:`DieCell.facet_position`.

    References whose cell is ``die`` (or any :class:`DieCell`) are the frame
    and are not checked.

    Examples
    --------
    >>> from photonix.layout import Cell, components as lc
    >>> from photonix.layout.library import die_frame
    >>> die = die_frame((1000.0, 1000.0), dicing_lane=50.0, keepout=50.0, facet_sides=("E",))
    >>> top = Cell("chip")
    >>> _ = top.add_ref(die, name="frame")
    >>> _ = top.add_ref(lc.straight(100.0), origin=(150.0, 500.0), name="wg")
    >>> _ = top.add_ref(lc.edge_coupler(200.0), origin=(798.0, 600.0), name="ec")   # tip 2 um short
    >>> [(i.kind, i.instances) for i in check_floorplan(top, die)]
    [('facet', ('ec',))]
    """
    if not math.isfinite(facet_tol) or facet_tol < 0:
        raise ValueError("facet_tol must be non-negative and finite")
    w, h = die.die_size
    margin = die.dicing_lane + (die.keepout if keepout is None else float(keepout))
    usable = ((margin, margin), (w - margin, h - margin))
    outline = ((0.0, 0.0), (float(w), float(h)))
    facet_sides = {_side(s) for s in die.facet_sides}
    facet_set = set(facet_models)
    issues: list[FloorplanIssue] = []

    def outside(box: BBox, area: BBox) -> bool:
        return (box[0][0] < area[0][0] - tol or box[0][1] < area[0][1] - tol
                or box[1][0] > area[1][0] + tol or box[1][1] > area[1][1] + tol)

    blocks = [(r.name or f"ref{i}", r) for i, r in enumerate(top.references)
              if r.cell is not die and not isinstance(r.cell, DieCell)]
    boxes = {name: reference_bbox(r) for name, r in blocks}
    for name, _r in blocks:
        if outside(boxes[name], outline):
            issues.append(FloorplanIssue("outside_die", f"{name!r} extends beyond the {w:g} x {h:g} um die",
                                         (name,), boxes[name]))

    own = _box(np.concatenate([p for p, _ in top.polygons])) if top.polygons else None
    if own is not None and outside(own, usable):
        issues.append(FloorplanIssue("keepout", f"geometry drawn directly in {top.name!r} enters the keep-out",
                                     (top.name,), own))
    facets = []
    for name, ref in blocks:
        for leaf_name, leaf, origin, rot, mir in _leaves(name, ref.cell, ref.origin, ref.rotation, ref.mirror):
            if leaf.model in facet_set:
                facets.append((leaf_name, leaf, origin, rot, mir))
                continue
            box = _box(_transform(_cell_points(leaf), origin, rot, mir))
            if box != ((0.0, 0.0), (0.0, 0.0)) and outside(box, usable):
                issues.append(FloorplanIssue("keepout", f"{leaf_name!r} enters the dicing lane / keep-out band "
                                             f"(usable area {usable})", (leaf_name,), box))

    names = [n for n, r in blocks if include_routes or r.cell.model != "route"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ov = _overlap(boxes[a], boxes[b], tol)
            if ov is not None:
                issues.append(FloorplanIssue("overlap", f"{a!r} and {b!r} overlap", (a, b), ov))

    for leaf_name, leaf, origin, rot, mir in facets:
        if facet_port not in leaf.ports:
            issues.append(FloorplanIssue("facet", f"facet device {leaf_name!r} has no port {facet_port!r}",
                                         (leaf_name,), _box(_transform(_cell_points(leaf), origin, rot, mir))))
            continue
        p: Port = leaf.ports[facet_port].moved(origin[0], origin[1], rot, mir)
        where = (p.center, p.center)
        side = next((s for s, o in _SIDE_ORIENTATION.items()
                     if abs((p.orientation - o + 180.0) % 360.0 - 180.0) <= 1e-6), None)
        if side is None or p.port_type != OPTICAL:
            issues.append(FloorplanIssue("facet", f"{leaf_name}.{facet_port} faces {p.orientation:g} deg, not a die "
                                         "edge", (leaf_name,), where))
            continue
        if side not in facet_sides:
            issues.append(FloorplanIssue("facet", f"{leaf_name}.{facet_port} faces the {side} side, which is not a "
                                         f"facet side {sorted(facet_sides)}", (leaf_name,), where))
            continue
        coord = p.center[0] if side in ("E", "W") else p.center[1]
        dist = abs(coord - die.facet_position(side))
        if dist > facet_tol + tol:
            issues.append(FloorplanIssue("facet", f"{leaf_name}.{facet_port} is {dist:.4g} um from the {side} facet "
                                         f"(tolerance {facet_tol:g} um)", (leaf_name,), where))
    return issues
