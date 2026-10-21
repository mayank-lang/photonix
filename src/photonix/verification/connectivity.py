"""Geometric connectivity checks: ports on geometry, opens, near-opens and optical shorts.

:func:`photonix.layout.extract_netlist` joins two ports when their centres
coincide and is otherwise silent. These checks look at the drawn polygons, so
the layout errors that extraction cannot see become explicit
:class:`LvsIssue` records:

* :func:`check_ports_on_geometry` -- a port must sit on the face of its own
  cell's geometry (a probe just inside the face is covered, a probe just
  outside is not) and the drawn face must be at least as wide as the port.
* :func:`check_optical_connectivity` -- unconnected ports that nearly meet
  another port (``near_open``) or abut another instance's polygon, ports
  declared external by nobody (``open``), coincident ports of different width
  or direction, and same-layer overlaps of different instances away from
  their declared connections (``short``; the waveguide-overlap rule of
  SiEPIC's DevRec check, Chrostowski & Hochberg, *Silicon Photonics Design*,
  CUP 2015, ch. 10).

All checks are geometric (gdstk booleans on a 1e-6 µm grid), work in the top
cell's frame, and are invariant under rotating or mirroring the whole layout.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from photonix.layout.cell import OPTICAL, Cell, Port

from ._geom import (
    MARKER_LAYERS,
    Leaf,
    all_loops,
    bbox_of,
    collect_leaves,
    distance_to_boundary,
    gdstk,
    inside,
    layer_tuple,
    line_chord,
    merge,
    optical_ports,
    sweep_pairs,
)

_FINE = 1e-6   # µm boolean grid: overlap areas exact to ~1e-6 um^2 in any orientation

__all__ = ["LvsIssue", "ISSUE_KINDS", "check_ports_on_geometry", "check_optical_connectivity"]

#: Every :attr:`LvsIssue.kind`, with its meaning.
ISSUE_KINDS: dict[str, str] = {
    "open": "an internal optical port connects to nothing and is not a port of the top cell",
    "near_open": "an unconnected port almost meets another port, or its face abuts another instance's geometry",
    "short": "geometry of two instances overlaps on an optical layer away from their declared connection",
    "port_off_geometry": "a port does not sit on the face of its cell's drawn geometry",
    "port_width_mismatch": "a port's width differs from the drawn face or from the port it connects to",
    "angle_mismatch": "two coincident ports do not face each other",
    "ambiguous_connection": "more than two ports coincide",
    "parameter_mismatch": "a model setting differs from the value re-extracted from the drawn geometry",
    "missing_instance": "a schematic instance has no layout counterpart",
    "extra_instance": "a layout instance has no schematic counterpart",
    "model_mismatch": "matched instances use different models",
    "setting_mismatch": "matched instances have different settings",
    "connection_mismatch": "a connection exists in only one of layout and schematic",
    "external_port_mismatch": "an external port is missing or attached to a different instance port",
}


@dataclass(frozen=True)
class LvsIssue:
    """One connectivity or layout-versus-schematic finding.

    Attributes
    ----------
    kind
        One of :data:`ISSUE_KINDS`.
    location
        Representative point in the top cell's frame (µm).
    objects
        The instance ports or instances involved (``"inst.port"`` / ``"inst"``).
    detail
        Human-readable explanation.
    value
        The measured quantity where one exists: the gap of a near-open, the
        overlap area of a short (µm^2), the offset of a port, the extracted
        parameter of a mismatch.
    marker
        Optional polygon (``(n, 2)`` µm) for marker browsers; not compared.
    """

    kind: str
    location: tuple[float, float]
    objects: tuple[str, ...]
    detail: str
    value: float | None = None
    marker: np.ndarray | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in ISSUE_KINDS:
            raise ValueError(f"unknown issue kind {self.kind!r}; expected one of {sorted(ISSUE_KINDS)}")


def _loc(p: Sequence[float]) -> tuple[float, float]:
    return (float(p[0]), float(p[1]))


def _port_layer_geometry(cell: Cell) -> dict[tuple[int, int], tuple[list, list[np.ndarray]]]:
    by_layer: dict[tuple[int, int], list[np.ndarray]] = {}
    for pts, lay in cell.get_polygons():
        by_layer.setdefault(layer_tuple(lay), []).append(pts)
    out = {}
    for lay, pts in by_layer.items():
        merged = merge(pts, precision=_FINE)
        out[lay] = (merged, all_loops(merged))
    return out


def check_ports_on_geometry(
    cell: Cell,
    *,
    probe: float = 1e-3,
    width_tol: float = 2e-3,
    marker_layers: Iterable[tuple[int, int]] = MARKER_LAYERS,
    exact_width: bool = False,
    port_types: Iterable[str] = (OPTICAL,),
) -> list[LvsIssue]:
    """Check that every port of ``cell`` sits on the face of its drawn geometry.

    For a port at ``c`` with outward normal ``n`` and width ``w`` on layer
    ``L``:

    1. ``c - probe n`` must lie inside the merged ``L`` polygons, else
       ``port_off_geometry`` with the distance from ``c`` to the nearest
       ``L`` geometry as value;
    2. ``c + probe n`` must lie outside them (the port is on the face, not
       inside the waveguide), else ``port_off_geometry``;
    3. the chord of the geometry through ``c - probe n`` perpendicular to
       ``n`` must cover the port span ``[-w/2, w/2]``. A face narrower than
       the port gives ``port_width_mismatch`` unless the geometry reaches
       the port width within one port width of the face (corrugated
       gratings ending on a narrow tooth); a face wider than the port is
       accepted (MMI and star-coupler bodies) unless ``exact_width``.

    Ports on ``marker_layers`` (the fibre marker ``(203, 0)``) and ports of
    other types than ``port_types`` are skipped.

    Parameters
    ----------
    cell
        Any cell; its own ports are checked against its flattened geometry.
    probe
        Probe depth (µm).
    width_tol
        Tolerance on widths and offsets (µm).
    marker_layers
        Layers whose ports mark positions rather than geometry.
    exact_width
        Also flag a drawn face wider than the port.
    port_types
        Port types to check (default optical only).

    Returns
    -------
    list of LvsIssue
        In the cell's own frame, ``objects = (port name,)``.

    Examples
    --------
    >>> from photonix.layout import Cell, components as lc
    >>> check_ports_on_geometry(lc.straight(10.0))
    []
    >>> bad = Cell("bad")
    >>> _ = bad.add_polygon([(0, -0.25), (9.5, -0.25), (9.5, 0.25), (0, 0.25)])
    >>> _ = bad.add_port("o1", (0, 0), 180.0, 0.5)
    >>> _ = bad.add_port("o2", (10, 0), 0.0, 0.5)    # 0.5 um beyond the drawn end
    >>> [(i.kind, i.objects, round(i.value, 6)) for i in check_ports_on_geometry(bad)]
    [('port_off_geometry', ('o2',), 0.5)]
    """
    if not (np.isfinite(probe) and probe > 0 and np.isfinite(width_tol) and width_tol >= 0):
        raise ValueError("probe must be positive and width_tol non-negative")
    markers = {layer_tuple(m) for m in marker_layers}
    types = set(port_types)
    geometry = _port_layer_geometry(cell)
    issues: list[LvsIssue] = []
    for name, port in cell.ports.items():
        lay = layer_tuple(port.layer)
        if port.port_type not in types or lay in markers:
            continue
        merged, loops = geometry.get(lay, ([], []))
        c = np.asarray(port.center, float)
        n = port.direction
        t = np.array([-n[1], n[0]])
        w = float(port.width)
        p_in, p_out = c - probe * n, c + probe * n
        hit_in, hit_out = inside([p_in, p_out], merged)
        if not hit_in:
            dist = distance_to_boundary(c, loops)
            issues.append(LvsIssue(
                "port_off_geometry", _loc(c), (name,),
                f"port {name!r} is not on drawn geometry of layer {lay}: nearest {lay} geometry is "
                f"{dist:.6g} um away" if np.isfinite(dist) else f"port {name!r}: no geometry on layer {lay}",
                float(dist) if np.isfinite(dist) else None))
            continue
        if hit_out:
            issues.append(LvsIssue(
                "port_off_geometry", _loc(c), (name,),
                f"port {name!r} is inside the drawn geometry: layer {lay} continues past its face"))
            continue
        chord = line_chord(p_in, t, loops)
        if chord is None:  # pragma: no cover - p_in is inside, so the line crosses the boundary
            continue
        lo, hi = chord
        face = hi - lo
        if face < w - width_tol:
            reach = max([face] + [
                (lambda ch: ch[1] - ch[0] if ch else 0.0)(line_chord(c - d * n, t, loops))
                for d in np.linspace(probe, max(w, 2 * probe), 9)[1:]
            ])
            if reach < w - width_tol:
                issues.append(LvsIssue(
                    "port_width_mismatch", _loc(c), (name,),
                    f"port {name!r} is {w:g} um wide but the drawn face is {face:.6g} um", float(face)))
                continue
        elif exact_width and face > w + width_tol:
            issues.append(LvsIssue(
                "port_width_mismatch", _loc(c), (name,),
                f"port {name!r} is {w:g} um wide but the drawn face is {face:.6g} um", float(face)))
            continue
        over = max(lo + w / 2, w / 2 - hi, 0.0)
        if face >= w - width_tol and over > width_tol:
            issues.append(LvsIssue(
                "port_off_geometry", _loc(c), (name,),
                f"port {name!r} overhangs the drawn face by {over:.6g} um (centre offset "
                f"{0.5 * (lo + hi):.6g} um)", float(over)))
    return issues


# --------------------------------------------------------------------------- #
# connectivity of a placed layout
# --------------------------------------------------------------------------- #
def _antiparallel(a: Port, b: Port, angle_tol: float) -> bool:
    return abs(((a.orientation - b.orientation) % 360.0) - 180.0) <= angle_tol


def _ports_table(leaves: Sequence[Leaf], marker_layers) -> list[tuple[str, str, Port]]:
    return [(lf.name, pn, p) for lf in leaves for pn, p in sorted(optical_ports(lf, marker_layers).items())]


def _match_ports(terms: list[tuple[str, str, Port]], tol: float) -> tuple[list[tuple[int, int]], list[list[int]]]:
    """Coincident port pairs (different instances) and the clusters of more than two."""
    if not terms:
        return [], []
    from scipy.spatial import cKDTree

    xy = np.array([p.center for _i, _n, p in terms], float)
    tree = cKDTree(xy)
    pairs = sorted((int(a), int(b)) for a, b in tree.query_pairs(tol) if terms[a][0] != terms[b][0])
    deg: dict[int, set[int]] = {}
    for a, b in pairs:
        deg.setdefault(a, set()).add(b)
        deg.setdefault(b, set()).add(a)
    clusters: list[list[int]] = []
    seen: set[int] = set()
    for i, nb in deg.items():
        if len(nb) > 1 and i not in seen:
            group = sorted({i, *nb})
            seen.update(group)
            clusters.append(group)
    bad = set(seen)
    return [(a, b) for a, b in pairs if a not in bad and b not in bad], clusters


def check_optical_connectivity(
    top: Cell,
    *,
    tol: float = 1e-3,
    near: float = 1.0,
    layers: Sequence[tuple[int, int]] | None = None,
    min_short_area: float = 1e-4,
    probe: float = 1e-3,
    angle_tol: float = 1e-3,
    marker_layers: Iterable[tuple[int, int]] = MARKER_LAYERS,
) -> list[LvsIssue]:
    """Opens, near-opens, misjoined ports and optical shorts of a placed layout.

    Leaf instances are found with the same rules (and names) as
    :func:`photonix.layout.extract_netlist`; two of their optical ports are
    connected when their centres lie within ``tol``.

    * Coincident ports with different widths give ``port_width_mismatch``,
      ports that do not face each other ``angle_mismatch``, and three or more
      coincident ports ``ambiguous_connection``.
    * An unconnected port within ``near`` of another unconnected port gives
      one ``near_open`` per pair (value: their distance). An unconnected
      port whose face abuts another instance's polygon (a probe ``probe``
      outside the face lands in it) is also a ``near_open`` -- unless the
      two instances already short.
    * When ``top`` declares optical ports, every other unconnected port that
      is not on a marker layer and does not coincide with a top port is an
      ``open``. Without top ports the external ports are unknown and opens
      are not reported.
    * Same-layer overlaps of two instances larger than ``min_short_area``
      (µm^2) are ``short`` (value: overlap area), except inside a disc of
      radius ``max(width)`` around each of their declared connections.

    Parameters
    ----------
    top
        The layout.
    tol
        Port-coincidence tolerance (µm), as in extraction.
    near
        Search radius for near-opens (µm).
    layers
        Optical layers for the short check; default: the layers of all
        non-marker optical ports.
    min_short_area
        Smallest overlap reported as a short (µm^2).
    probe
        Outward probe distance for abutting faces (µm).
    angle_tol
        Antiparallel tolerance (degrees).
    marker_layers
        Port layers that mark positions (fibre ports); never opens.

    Returns
    -------
    list of LvsIssue

    Examples
    --------
    >>> from photonix.layout import Cell, components as lc
    >>> top = Cell("top")
    >>> _ = top.add_ref(lc.straight(10.0), name="a")
    >>> _ = top.add_ref(lc.straight(10.0), origin=(10.005, 0), name="b")   # 5 nm gap
    >>> [(i.kind, i.objects, round(i.value, 9)) for i in check_optical_connectivity(top)]
    [('near_open', ('a.o2', 'b.o1'), 0.005)]
    """
    for nm, v in (("tol", tol), ("near", near), ("min_short_area", min_short_area), ("probe", probe)):
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"{nm} must be non-negative and finite")
    markers = tuple(layer_tuple(m) for m in marker_layers)
    leaves = collect_leaves(top)
    terms = _ports_table(leaves, markers)
    pairs, clusters = _match_ports(terms, tol)
    issues: list[LvsIssue] = []
    tag = [f"{i}.{n}" for i, n, _p in terms]

    connected: set[int] = set()
    conn_by_pair: dict[frozenset, list[tuple[Port, Port]]] = {}
    for a, b in pairs:
        pa, pb = terms[a][2], terms[b][2]
        connected.update((a, b))
        conn_by_pair.setdefault(frozenset((terms[a][0], terms[b][0])), []).append((pa, pb))
        if abs(pa.width - pb.width) > 1e-6:
            issues.append(LvsIssue("port_width_mismatch", _loc(pa.center), (tag[a], tag[b]),
                                   f"connected ports have widths {pa.width:g} and {pb.width:g} um",
                                   float(abs(pa.width - pb.width))))
        if not _antiparallel(pa, pb, angle_tol):
            issues.append(LvsIssue("angle_mismatch", _loc(pa.center), (tag[a], tag[b]),
                                   f"coincident ports face {pa.orientation:g} and {pb.orientation:g} deg",
                                   float(abs(((pa.orientation - pb.orientation) % 360.0) - 180.0))))
    for group in clusters:
        connected.update(group)
        issues.append(LvsIssue("ambiguous_connection", _loc(terms[group[0]][2].center),
                               tuple(tag[i] for i in group), f"{len(group)} ports coincide"))

    top_ports = [np.asarray(p.center, float) for p in top.ports.values()
                 if p.port_type == OPTICAL and layer_tuple(p.layer) not in markers]
    external = {i for i in range(len(terms)) if i not in connected
                and any(np.linalg.norm(c - np.asarray(terms[i][2].center)) <= tol for c in top_ports)}
    loose = [i for i in range(len(terms)) if i not in connected and i not in external]

    # -- shorts -------------------------------------------------------------- #
    opt_layers = ({layer_tuple(x) for x in layers} if layers is not None
                  else {layer_tuple(p.layer) for _i, _n, p in terms})
    shorted: set[frozenset] = set()
    geom: dict[tuple[str, tuple[int, int]], list] = {}
    for lf in leaves:
        by_layer: dict[tuple[int, int], list[np.ndarray]] = {}
        for pts, lay in lf.polygons(opt_layers):
            by_layer.setdefault(lay, []).append(pts)
        for lay, pts in by_layer.items():
            merged = merge(pts, precision=_FINE)
            if merged:
                geom[(lf.name, lay)] = merged
    geom_box = {k: bbox_of(np.concatenate([np.asarray(p.points) for p in v])) for k, v in geom.items()}
    gd = gdstk()
    for lay in sorted(opt_layers):
        keys = [k for k in geom if k[1] == lay]
        boxes = [geom_box[k] for k in keys]
        for i, j in sweep_pairs(boxes):
            ka, kb = keys[i], keys[j]
            overlap = gd.boolean(geom[ka], geom[kb], "and", precision=_FINE)
            if not overlap:
                continue
            pair = frozenset((ka[0], kb[0]))
            discs = [gd.ellipse(tuple(pa.center), max(pa.width, pb.width), tolerance=1e-3)
                     for pa, pb in conn_by_pair.get(pair, [])]
            if discs:
                overlap = gd.boolean(overlap, discs, "not", precision=_FINE)
            area = float(sum(p.area() for p in overlap))
            if area <= min_short_area:
                continue
            big = max(overlap, key=lambda p: p.area())
            pts = np.asarray(big.points)
            shorted.add(pair)
            issues.append(LvsIssue("short", _loc(pts.mean(axis=0)), tuple(sorted((ka[0], kb[0]))),
                                   f"instances overlap by {area:.6g} um^2 on layer {lay}", area, marker=pts))

    # -- near-opens and opens ---------------------------------------------- #
    used: set[int] = set()
    if len(loose) > 1 and near > 0:
        from scipy.spatial import cKDTree

        xy = np.array([terms[i][2].center for i in loose], float)
        cand = []
        for a, b in cKDTree(xy).query_pairs(near):
            ia, ib = loose[a], loose[b]
            if terms[ia][0] != terms[ib][0]:
                cand.append((float(np.linalg.norm(xy[a] - xy[b])), min(ia, ib), max(ia, ib)))
        for dist, ia, ib in sorted(cand):
            if ia in used or ib in used:
                continue
            used.update((ia, ib))
            pa, pb = terms[ia][2], terms[ib][2]
            facing = "facing" if _antiparallel(pa, pb, max(angle_tol, 1.0)) else "not facing"
            issues.append(LvsIssue("near_open", _loc(0.5 * (np.asarray(pa.center) + np.asarray(pb.center))),
                                   (tag[ia], tag[ib]), f"ports are {dist:.6g} um apart ({facing}) but not connected",
                                   dist))
    for i in loose:
        if i in used:
            continue
        inst, _pn, port = terms[i]
        p_out = np.asarray(port.center, float) + probe * port.direction
        lay = layer_tuple(port.layer)
        abut = None
        for (other, olay), merged in geom.items():
            if olay != lay or other == inst:
                continue
            bx0, by0, bx1, by1 = geom_box[(other, olay)]
            if not (bx0 <= p_out[0] <= bx1 and by0 <= p_out[1] <= by1):
                continue
            if inside([p_out], merged)[0]:
                abut = other
                break
        if abut is not None:
            if frozenset((inst, abut)) not in shorted:
                issues.append(LvsIssue("near_open", _loc(port.center), (tag[i], abut),
                                       f"port face abuts instance {abut!r} where it has no port"))
            continue
        if top_ports:
            issues.append(LvsIssue("open", _loc(port.center), (tag[i],),
                                   "port is unconnected and is not a port of the top cell"))
    return issues

