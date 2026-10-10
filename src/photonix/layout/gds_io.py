"""Lossless GDSII/OASIS round trips: units, hierarchy, arrays, ports and a metadata sidecar.

:func:`~photonix.layout.read_gds` flattens the top cell and keeps only label
positions; :func:`read_layout` instead rebuilds the full :class:`Cell` tree
and the ports, and :func:`write_layout` writes what it needs to do so.

What a round trip preserves
---------------------------
* **Units.** Files are read with ``unit=1e-6`` whatever their user unit, so a
  layout drawn in nanometre user units comes back in µm. The database grid
  (``precision``) sets the coordinate resolution; a grid coarser than 1 nm is
  reported with a warning.
* **Hierarchy.** One :class:`~photonix.layout.labels.LabeledCell` per GDS
  structure, shared by every reference to it. References keep origin,
  rotation and x-reflection (Photonix and GDSII both apply *mirror about x ->
  rotate -> translate*). A magnified reference (Photonix references cannot
  scale) is flattened locally into a scaled, geometry-only copy with a
  warning, or rejected with ``magnification="error"``. GDS ``AREF`` arrays
  and OASIS repetitions expand into one reference per element, named
  ``"<base>_<column>_<row>"`` (``"<base>_<k>"`` for explicit offsets).
* **Instance names**, which GDSII cannot store natively, are written as the
  ``SREF`` property :data:`INSTANCE_NAME_ATTR` and read back from it.
* **Ports**, by priority:

  1. the **JSON sidecar** written next to the layout (exact centre,
     orientation, width, layer and type, plus ``model``/``model_settings``
     and the original cell name). Each structure's entry holds a hash of its
     drawn content (polygons, paths and placements on the integer database
     grid, order-independent); an entry whose structure was edited after
     writing is ignored with a :class:`StaleSidecarWarning`, so stale
     simulation semantics are never applied to changed geometry;
  2. **PinRec markers** (SiEPIC-Tools convention, Chrostowski & Hochberg,
     *Silicon Photonics Design*, CUP 2015): a two-point ``PATH`` on the pin
     layer, :data:`PIN_LENGTH` long, centred on the port and drawn from the
     inside to the outside, whose width is the port width, plus a ``TEXT``
     with the port name at its centre. Photonix also writes the port's layer
     and exact orientation as path properties; foreign pins without them get
     the layer of the smallest shape under the port and the orientation of
     the path;
  3. **bare labels**: a ``TEXT`` whose origin lies within ``tol`` of an edge
     of the merged geometry on the port layer. The port faces along that
     edge's outward normal (sign from the polygon's shoelace orientation),
     its width is the length of the straight boundary run (collinear edges
     merged) through the label, and its centre is the label projected onto
     that run.

* **Circuit semantics** (``model``, ``model_settings``) only through the
  sidecar. Without it a leaf structure is marked
  :data:`~photonix.layout.GEOMETRY_ONLY` and extraction refuses to simulate
  it rather than guess.

Exactness and its limits
------------------------
:func:`write_layout` snaps polygon vertices, reference origins, port centres
and widths, and label positions to the database grid exactly as
:func:`snap_to_grid` does (and drops consecutive duplicate vertices and
zero-area polygons it creates), and reports the largest displacement in
:attr:`LayoutArtifact.snap_error`. ``read_layout(write_layout(c))`` then
reproduces ``snap_to_grid(c)`` exactly: the same
:func:`photonix.layout.gds._fingerprint` for every structure, and the same
extracted netlist. Reference rotations are stored as GDS 8-byte reals and come
back to 1e-9 degree.

Marker-only reads are exact for Manhattan ports. A non-Manhattan *label* port
is limited by its snapped face: orientation to about
``atan(sqrt(2) grid / width)`` (0.16 degree for a 0.5 µm face on a 1 nm grid).
Ports inside rotated *references* are inferred in the child's own frame and
are exact. The label convention measures the drawn face, so it cannot know a
port that is narrower than its face (an MMI port on the body edge reads the
body width), a port inside a shape (a pad centre), at a corner (ambiguous,
skipped with a warning) or on a layer without geometry (a fibre port).
OASIS ``TEXT`` records carry no rotation or magnification, so labels lose them
in ``.oas`` files.

The default ``timestamp`` makes GDSII output byte-reproducible, so
:attr:`LayoutArtifact.sha256` identifies a layout.
"""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import math
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from .cell import ELECTRICAL, GEOMETRY_ONLY, OPTICAL, Cell, Port
from .gds import _fingerprint, _gdstk
from .labels import Label, LabeledCell

__all__ = [
    "read_layout", "write_layout", "LayoutArtifact", "list_top_cells", "layout_units", "infer_ports",
    "snap_to_grid", "sidecar_path_for", "StaleSidecarWarning",
    "PINREC_LAYER", "PINREC_ELECTRICAL_LAYER", "PIN_LENGTH", "INSTANCE_NAME_ATTR", "PORT_LAYER_ATTR",
    "PORT_ORIENTATION_ATTR", "SIDECAR_FORMAT", "DEFAULT_TIMESTAMP",
]

#: Optical pin layer (SiEPIC EBeam PDK ``PinRec``).
PINREC_LAYER = (1, 10)
#: Electrical pin layer (SiEPIC EBeam PDK ``PinRecM``; check your PDK's layer map).
PINREC_ELECTRICAL_LAYER = (1, 11)
#: Length of a written pin path in µm (SiEPIC-Tools ``PIN_LENGTH`` = 100 nm).
PIN_LENGTH = 0.1
#: GDS property attribute holding a reference's instance name.
INSTANCE_NAME_ATTR = 1
#: GDS property attribute on a pin path holding the port layer as ``"layer/datatype"``.
PORT_LAYER_ATTR = 2
#: GDS property attribute on a pin path holding the exact port orientation in degrees.
PORT_ORIENTATION_ATTR = 3
#: ``"format"`` field of the JSON sidecar.
SIDECAR_FORMAT = "photonix.layout-sidecar"
_SIDECAR_VERSION = 1
#: Fixed GDSII timestamp used by default so that identical layouts give identical bytes.
DEFAULT_TIMESTAMP = _dt.datetime(2000, 1, 1)
_UNIT = 1e-6  # Photonix lengths are in µm
_MAX_PROPERTY_BYTES = 126  # GDSII PROPVALUE limit


class StaleSidecarWarning(UserWarning):
    """A sidecar entry no longer matches the structure it describes and was ignored."""


@dataclass(frozen=True)
class LayoutArtifact:
    """What :func:`write_layout` wrote.

    Attributes
    ----------
    layout_path
        The GDSII/OASIS file.
    sidecar_path
        The JSON sidecar, or ``None``.
    sha256
        Hex digest of the layout file's bytes.
    structures
        GDS structure name -> Photonix fingerprint of the (grid-snapped) cell
        written there.
    top
        GDS name of the top structure.
    snap_error
        Largest distance (µm) any vertex, origin, port centre or label moved
        when snapped to the database grid; 0 for an on-grid layout.
    """

    layout_path: str
    sidecar_path: str | None
    sha256: str
    structures: dict[str, str]
    top: str
    snap_error: float


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _layer(layer) -> tuple[int, int]:
    return (int(layer[0]), int(layer[1]))


_Format = Literal["gds", "oas"]


def _fmt(path: str, fmt: str | None) -> _Format:
    if fmt == "gds":
        return "gds"
    if fmt == "oas":
        return "oas"
    if fmt is not None:
        raise ValueError(f"fmt must be 'gds', 'oas' or None, got {fmt!r}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".gds", ".gdsii", ".gds2", ".gdsii2"):
        return "gds"
    if ext in (".oas", ".oasis"):
        return "oas"
    raise ValueError(f"cannot infer the layout format from {path!r}; pass fmt='gds' or fmt='oas'")


def sidecar_path_for(path: str | os.PathLike[str]) -> str:
    """Default sidecar location for a layout file: ``<path>.photonix.json``.

    Examples
    --------
    >>> sidecar_path_for("chip.gds")
    'chip.gds.photonix.json'
    """
    return os.fspath(path) + ".photonix.json"


def _clean_angle(deg: float) -> float:
    """Normalise to [0, 360) and remove the float noise of the radian/degree round trip."""
    r = float(deg) % 360.0
    rr = round(r, 9)
    if abs(rr - r) < 1e-9:
        r = rr
    return 0.0 if r >= 360.0 else r


def _norm(values: np.ndarray, grid: float) -> np.ndarray:
    """Snap values that are on the grid up to float noise; leave genuinely off-grid values alone."""
    arr = np.asarray(values, dtype=float)
    snapped = np.rint(arr / grid) * grid + 0.0
    return np.where(np.abs(snapped - arr) <= 1e-6 * grid, snapped, arr + 0.0)


def _ring(points: np.ndarray) -> np.ndarray:
    """Drop consecutive duplicate vertices (including the closing one)."""
    if len(points) == 0:
        return points
    keep = np.concatenate([[True], np.any(points[1:] != points[:-1], axis=1)])
    out = points[keep]
    while len(out) > 1 and np.array_equal(out[-1], out[0]):
        out = out[:-1]
    return out


def _int_area2(ints: np.ndarray) -> int:
    x = ints[:, 0].astype(object)
    y = ints[:, 1].astype(object)
    return int(sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# --------------------------------------------------------------------------- #
# grid snapping
# --------------------------------------------------------------------------- #
class _Snapper:
    def __init__(self, grid: float):
        if not math.isfinite(grid) or grid <= 0:
            raise ValueError(f"grid must be positive and finite, got {grid!r}")
        self.grid = float(grid)
        self.memo: dict[int, Cell] = {}
        self.error = 0.0
        self.dropped = 0

    def pts(self, pts) -> np.ndarray:
        arr = np.asarray(pts, dtype=float)
        snapped = np.rint(arr / self.grid) * self.grid + 0.0  # + 0.0 turns -0.0 into 0.0
        if arr.size:
            self.error = max(self.error, float(np.max(np.linalg.norm(snapped - arr, axis=-1))))
        return snapped

    def cell(self, cell: Cell) -> Cell:
        key = id(cell)
        if key in self.memo:
            return self.memo[key]
        new = copy.copy(cell)  # keeps the concrete class and any extra dataclass fields
        polys = []
        for pts, layer in cell.polygons:
            ring = _ring(self.pts(pts))
            ints = np.rint(ring / self.grid).astype(np.int64) if len(ring) else ring
            if len(ring) < 3 or _int_area2(ints) == 0:
                self.dropped += 1
                continue
            polys.append((ring, _layer(layer)))
        new.polygons = polys
        new.references = []
        for ref in cell.references:
            child = self.cell(ref.cell)
            origin = self.pts(ref.origin)
            new.references.append(type(ref)(child, (float(origin[0]), float(origin[1])), float(ref.rotation),
                                            bool(ref.mirror), ref.name))
        ports = {}
        for name, prt in cell.ports.items():
            ctr = self.pts(prt.center)
            width = max(float(np.rint(prt.width / self.grid)) * self.grid, self.grid)
            ports[name] = Port(prt.name, (float(ctr[0]), float(ctr[1])), float(prt.orientation), width,
                               _layer(prt.layer), prt.port_type)
        new.ports = ports
        new.model_settings = dict(cell.model_settings)
        if isinstance(cell, LabeledCell) and isinstance(new, LabeledCell):
            labels = []
            for lbl in cell.labels:
                pos = self.pts(lbl.position)
                labels.append(Label(lbl.text, (float(pos[0]), float(pos[1])), lbl.layer, lbl.rotation,
                                    lbl.magnification, lbl.anchor))
            new.labels = labels
        self.memo[key] = new
        return new


def snap_to_grid(cell: Cell, grid: float = 1e-3) -> Cell:
    """A copy of the cell tree snapped to a ``grid`` (µm) exactly as :func:`write_layout` does.

    Polygon vertices, reference origins, port centres and widths (at least one
    grid step) and label positions are rounded to the nearest grid point;
    consecutive duplicate vertices and polygons left with zero area are
    removed. Rotations, mirroring, names and model data are unchanged, and
    shared sub-cells stay shared.

    Examples
    --------
    >>> from photonix.layout import Cell
    >>> c = Cell("c").add_polygon([(0, 0), (1.0004, 0), (1.0004, 0.5), (0, 0.5)])
    >>> snap_to_grid(c).polygons[0][0][1].tolist()
    [1.0, 0.0]
    """
    return _Snapper(grid).cell(cell)


# --------------------------------------------------------------------------- #
# JSON encoding of model settings
# --------------------------------------------------------------------------- #
def _encode(value: Any, where: str) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.generic):
        return _encode(value.item(), where)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"__float__": repr(value)}
    if isinstance(value, complex):
        return {"__complex__": [_encode(value.real, where), _encode(value.imag, where)]}
    if isinstance(value, np.ndarray):
        return {"__ndarray__": _encode(value.tolist(), where), "dtype": str(value.dtype),
                "shape": list(value.shape)}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(v, where) for v in value]}
    if isinstance(value, list):
        return [_encode(v, where) for v in value]
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"{where}: model_settings keys must be strings, got {k!r}")
            out[k] = _encode(v, f"{where}[{k!r}]")
        return out
    raise TypeError(
        f"{where}: a {type(value).__name__} cannot be stored in the JSON sidecar; model_settings must hold "
        "numbers, strings, booleans, None, complex numbers, arrays, lists, tuples or dicts of these"
    )


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if isinstance(value, dict):
        if set(value) == {"__float__"}:
            return float(value["__float__"])
        if set(value) == {"__complex__"}:
            re_, im_ = value["__complex__"]
            return complex(_decode(re_), _decode(im_))
        if set(value) == {"__tuple__"}:
            return tuple(_decode(v) for v in value["__tuple__"])
        if set(value) == {"__ndarray__", "dtype", "shape"}:
            return np.asarray(_decode(value["__ndarray__"]), dtype=value["dtype"]).reshape(value["shape"])
        return {k: _decode(v) for k, v in value.items()}
    return value


def _settings_key(cell: Cell) -> str:
    return json.dumps(_encode(cell.model_settings, f"cell {cell.name!r}"), sort_keys=True)


# --------------------------------------------------------------------------- #
# content hash (staleness detection), computed on the gdstk objects as read
# --------------------------------------------------------------------------- #
def _offsets(element) -> np.ndarray:
    rep = getattr(element, "repetition", None)
    offs = np.asarray(rep.get_offsets(), dtype=float).reshape(-1, 2) if rep is not None else np.zeros((0, 2))
    return offs if len(offs) else np.zeros((1, 2))


def _canonical_polygon(points: np.ndarray, grid: float) -> np.ndarray:
    ints = _ring(np.rint(np.asarray(points, dtype=float) / grid).astype(np.int64))
    if len(ints) >= 3 and _int_area2(ints) < 0:
        ints = ints[::-1]
    if len(ints):
        start = int(np.lexsort((ints[:, 1], ints[:, 0]))[0])
        ints = np.roll(ints, -start, axis=0)
    return ints


def _content_hash(gcell, grid: float) -> str:
    """Order-independent hash of a structure's drawn content on the integer grid (TEXT excluded)."""
    shapes: list[str] = []

    def add(points, layer: int, datatype: int) -> None:
        canon = _canonical_polygon(points, grid)
        shapes.append(hashlib.sha256(f"{layer}/{datatype}:".encode() + canon.tobytes()).hexdigest())

    for poly in gcell.polygons:
        for off in _offsets(poly):
            add(np.asarray(poly.points) + off, poly.layer, poly.datatype)
    for path in gcell.paths:
        offs = _offsets(path)
        for poly in path.to_polygons():
            for off in offs:
                add(np.asarray(poly.points) + off, poly.layer, poly.datatype)
    refs: list[str] = []
    for ref in gcell.references:
        child = ref.cell if isinstance(ref.cell, str) else ref.cell.name
        origin = np.asarray(ref.origin, dtype=float)
        offs = sorted(tuple(int(v) for v in np.rint((origin + off) / grid)) for off in _offsets(ref))
        refs.append(repr((child, offs, _clean_angle(math.degrees(ref.rotation)), bool(ref.x_reflection),
                          round(float(ref.magnification), 12))))
    payload = json.dumps([sorted(shapes), sorted(refs)]).encode()
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
class _Writer:
    def __init__(self, lib, *, port_markers: str, pin_layer, pin_layer_e, pin_length: float,
                 instance_names: bool, grid: float):
        self.gdstk = _gdstk()
        self.lib = lib
        self.port_markers = port_markers
        self.pin_layer = _layer(pin_layer)
        self.pin_layer_e = _layer(pin_layer_e)
        self.pin_length = float(pin_length)
        self.instance_names = instance_names
        self.grid = grid
        self.fp_memo: dict[int, str] = {}
        self.key_memo: dict[int, str] = {}
        self.built: dict[str, Any] = {}
        self.counts: dict[str, int] = {}
        self.used: set[str] = set()
        self.cells: dict[str, Cell] = {}  # gds name -> snapped cell

    def key(self, cell: Cell) -> str:
        """Everything the writer emits for ``cell``: identical keys are written once."""
        k = id(cell)
        if k not in self.key_memo:
            labels = [(lb.text, lb.position, lb.layer, lb.rotation, lb.magnification, lb.anchor)
                      for lb in (cell.labels if isinstance(cell, LabeledCell) else [])]
            payload = repr((_fingerprint(cell, self.fp_memo), cell.model, _settings_key(cell), labels,
                            [(self.key(r.cell), r.name) for r in cell.references]))
            self.key_memo[k] = hashlib.sha256(payload.encode()).hexdigest()
        return self.key_memo[k]

    def add(self, cell: Cell):
        if not cell.name:
            raise ValueError("GDS cell names must be non-empty")
        key = self.key(cell)
        if key in self.built:
            return self.built[key]
        count = self.counts.get(cell.name, 0)
        gname = cell.name if count == 0 else f"{cell.name}${count}"
        while gname in self.used:
            count += 1
            gname = f"{cell.name}${count}"
        self.counts[cell.name] = count + 1
        self.used.add(gname)
        gdstk = self.gdstk
        gcell = self.lib.new_cell(gname)
        self.built[key] = gcell
        self.cells[gname] = cell
        for pts, (layer, dt) in cell.polygons:
            gcell.add(gdstk.Polygon(np.asarray(pts), layer=layer, datatype=dt))
        names: set[str] = set()
        for ref in cell.references:
            child = self.add(ref.cell)
            gref = gdstk.Reference(child, origin=tuple(ref.origin), rotation=math.radians(ref.rotation),
                                   x_reflection=bool(ref.mirror))
            if self.instance_names and ref.name is not None:
                if len(ref.name.encode()) > _MAX_PROPERTY_BYTES:
                    raise ValueError(f"instance name {ref.name!r} exceeds the {_MAX_PROPERTY_BYTES}-byte "
                                     "GDSII property limit")
                if ref.name in names:
                    raise ValueError(f"duplicate instance name {ref.name!r} in cell {cell.name!r}")
                names.add(ref.name)
                gref.set_gds_property(INSTANCE_NAME_ATTR, ref.name)
            gcell.add(gref)
        for prt in cell.ports.values():
            self._marker(gcell, prt)
        if isinstance(cell, LabeledCell):
            for lbl in cell.labels:
                gcell.add(gdstk.Label(lbl.text, lbl.position, anchor=lbl.anchor, rotation=math.radians(lbl.rotation),
                                      magnification=lbl.magnification, layer=lbl.layer[0], texttype=lbl.layer[1]))
        return gcell

    def _marker(self, gcell, prt: Port) -> None:
        gdstk = self.gdstk
        if self.port_markers == "none":
            return
        if self.port_markers == "label":
            gcell.add(gdstk.Label(prt.name, prt.center, layer=prt.layer[0], texttype=prt.layer[1]))
            return
        layer = self.pin_layer_e if prt.port_type == ELECTRICAL else self.pin_layer
        th = math.radians(prt.orientation)
        half = np.rint(0.5 * self.pin_length * np.array([math.cos(th), math.sin(th)]) / self.grid) * self.grid
        if not np.any(half):
            raise ValueError(f"pin_length {self.pin_length} is too short for the {self.grid} um grid")
        ctr = np.asarray(prt.center, dtype=float)
        path = gdstk.FlexPath([tuple(ctr - half), tuple(ctr + half)], prt.width, ends="flush", simple_path=True,
                              layer=layer[0], datatype=layer[1])
        path.set_gds_property(PORT_LAYER_ATTR, f"{prt.layer[0]}/{prt.layer[1]}")
        path.set_gds_property(PORT_ORIENTATION_ATTR, repr(float(prt.orientation)))
        gcell.add(path)
        gcell.add(gdstk.Label(prt.name, tuple(ctr), layer=layer[0], texttype=layer[1]))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_lib(path: str, fmt: str):
    gdstk = _gdstk()
    return gdstk.read_gds(path, unit=_UNIT) if fmt == "gds" else gdstk.read_oas(path, unit=_UNIT)


def write_layout(
    cell: Cell,
    path: str | os.PathLike[str],
    *,
    sidecar: bool | str | os.PathLike[str] = True,
    port_markers: Literal["pinrec", "label", "none"] = "pinrec",
    port_layer: Sequence[int] = PINREC_LAYER,
    electrical_port_layer: Sequence[int] = PINREC_ELECTRICAL_LAYER,
    pin_length: float = PIN_LENGTH,
    instance_names: bool = True,
    timestamp: _dt.datetime = DEFAULT_TIMESTAMP,
    fmt: Literal["gds", "oas"] | None = None,
    precision: float = 1e-9,
    max_points: int = 8190,
) -> LayoutArtifact:
    """Write ``cell``'s tree to GDSII or OASIS so that :func:`read_layout` restores it.

    Parameters
    ----------
    cell, path
        The top cell and the output file (format from the extension unless
        ``fmt`` is given).
    sidecar
        ``True`` writes ``<path>.photonix.json`` (see
        :func:`sidecar_path_for`), a path writes it there, ``False`` writes
        none (and deletes a default sidecar an earlier write left for this
        path, which would describe the overwritten file). The sidecar carries
        ports, ``model``, ``model_settings`` and the original cell names, plus
        a content hash per structure.
    port_markers
        In-file port markers: ``"pinrec"`` (SiEPIC pin paths and names on
        ``port_layer``/``electrical_port_layer``), ``"label"`` (a ``TEXT`` with
        the port name at the port centre on the port's own layer) or
        ``"none"``.
    pin_length
        Length of a pin path in µm.
    instance_names
        Store reference names as GDS property :data:`INSTANCE_NAME_ATTR`.
    timestamp
        GDSII creation/modification time; the fixed default keeps the bytes
        (and :attr:`LayoutArtifact.sha256`) reproducible. OASIS stores none.
    precision
        Database unit in metres (the user unit is always 1 µm).
    max_points
        GDSII polygons with more vertices are fractured by gdstk (8190 is the
        format's record limit).

    Returns
    -------
    LayoutArtifact

    Examples
    --------
    >>> import os, tempfile
    >>> from photonix.layout import components as lc, read_layout, extract_netlist
    >>> path = os.path.join(tempfile.mkdtemp(), "mzi.gds")
    >>> art = write_layout(lc.mzi(delta_length=10.0), path)
    >>> os.path.basename(art.sidecar_path)
    'mzi.gds.photonix.json'
    >>> back = read_layout(path)
    >>> extract_netlist(back).settings["top_s1"]          # drawn length survives the round trip
    {'length': 5.0}
    """
    if port_markers not in ("pinrec", "label", "none"):
        raise ValueError(f"port_markers must be 'pinrec', 'label' or 'none', got {port_markers!r}")
    if not math.isfinite(precision) or not 0 < precision <= _UNIT:
        raise ValueError(f"precision must be in (0, 1e-6] metres, got {precision!r}")
    if not math.isfinite(pin_length) or pin_length <= 0:
        raise ValueError("pin_length must be positive")
    gdstk = _gdstk()
    target = os.fspath(path)
    kind = _fmt(target, fmt)
    grid = precision / _UNIT
    snapper = _Snapper(grid)
    snapped = snapper.cell(cell)
    if snapper.dropped:
        warnings.warn(f"{snapper.dropped} polygon(s) collapsed to zero area on the {grid:g} um grid and were "
                      "not written", stacklevel=2)
    lib = gdstk.Library(unit=_UNIT, precision=precision)
    writer = _Writer(lib, port_markers=port_markers, pin_layer=port_layer, pin_layer_e=electrical_port_layer,
                     pin_length=pin_length, instance_names=instance_names, grid=grid)
    top = writer.add(snapped)
    if kind == "gds":
        lib.write_gds(target, max_points=int(max_points), timestamp=timestamp)
    else:
        # Rectangle/trapezoid compression would restart or reorder vertices; keep polygons verbatim.
        lib.write_oas(target, detect_rectangles=False, detect_trapezoids=False)
    digest = _sha256(target)
    structures = {name: _fingerprint(c, writer.fp_memo) for name, c in writer.cells.items()}

    side: str | None = None
    if sidecar is False:
        # A default sidecar left by an earlier write would describe the file just overwritten.
        stale = sidecar_path_for(target)
        if os.path.exists(stale):
            os.remove(stale)
    else:
        side = sidecar_path_for(target) if sidecar is True else os.fspath(sidecar)
        reread = {c.name: c for c in _read_lib(target, kind).cells}
        entries = {}
        for name, c in writer.cells.items():
            entries[name] = {
                "name": c.name,
                "fingerprint": structures[name],
                "content_hash": _content_hash(reread[name], grid),
                "model": c.model,
                "model_settings": _encode(c.model_settings, f"cell {c.name!r}"),
                "ports": [{"name": p.name, "center": list(p.center), "orientation": p.orientation,
                           "width": p.width, "layer": list(p.layer), "port_type": p.port_type}
                          for p in c.ports.values()],
            }
        from photonix import __version__

        doc = {
            "format": SIDECAR_FORMAT, "version": _SIDECAR_VERSION, "generator": f"photonix {__version__}",
            "layout": os.path.basename(target), "layout_sha256": digest, "layout_format": kind,
            "unit": _UNIT, "precision": precision, "top": top.name, "port_markers": port_markers,
            "pin_layers": {"optical": list(_layer(port_layer)), "electrical": list(_layer(electrical_port_layer))},
            "structures": entries,
        }
        with open(side, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
            fh.write("\n")
    return LayoutArtifact(target, side, digest, structures, top.name, snapper.error)


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def layout_units(path: str | os.PathLike[str], *, fmt: Literal["gds", "oas"] | None = None) -> tuple[float, float]:
    """``(user unit, database precision)`` of a layout file, in metres.

    OASIS stores only the database grid; its user unit is reported as 1e-6.

    Examples
    --------
    >>> import os, tempfile
    >>> from photonix.layout import components as lc
    >>> p = os.path.join(tempfile.mkdtemp(), "wg.gds")
    >>> _ = write_layout(lc.straight(), p, sidecar=False)
    >>> layout_units(p)
    (1e-06, 1e-09)
    """
    gdstk = _gdstk()
    target = os.fspath(path)
    if _fmt(target, fmt) == "gds":
        unit, prec = gdstk.gds_units(target)
        return float(unit), float(prec)
    return _UNIT, float(gdstk.oas_precision(target))


def list_top_cells(path: str | os.PathLike[str], *, fmt: Literal["gds", "oas"] | None = None) -> list[str]:
    """Sorted names of the structures no other structure references.

    Examples
    --------
    >>> import os, tempfile
    >>> from photonix.layout import components as lc
    >>> p = os.path.join(tempfile.mkdtemp(), "mzi.gds")
    >>> _ = write_layout(lc.mzi(), p, sidecar=False)
    >>> list_top_cells(p)
    ['mzi']
    """
    gdstk = _gdstk()
    target = os.fspath(path)
    if _fmt(target, fmt) == "gds":
        raw = gdstk.read_rawcells(target)
        deps = {d.name for rc in raw.values() for d in rc.dependencies(False)}
        return sorted(name for name in raw if name not in deps)
    return sorted(c.name for c in gdstk.read_oas(target, unit=_UNIT).top_level())


def _segments_distance(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    t = np.clip(np.einsum("ij,ij->i", pt - a, ab) / np.where(denom > 0, denom, 1.0), 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(pt - proj, axis=1), t


def _merged(polys: list[np.ndarray]) -> list[np.ndarray]:
    if not polys:
        return []
    gdstk = _gdstk()
    return [np.asarray(p.points, dtype=float)
            for p in gdstk.boolean([gdstk.Polygon(p) for p in polys], [], "or", precision=1e-9)]


def _face(poly: np.ndarray, edge: int, pt: np.ndarray, eps: float = 1e-6):
    """Outward normal (deg), width and centre of the straight boundary run through ``edge``.

    The run is the union of every boundary edge that lies on the edge's line
    with the same direction, so a face split by a hole's bridge (gdstk returns
    polygons with holes as keyholes) is still measured whole.
    """
    area2 = float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))
    a, b = poly[edge], poly[(edge + 1) % len(poly)]
    u = (b - a) / np.linalg.norm(b - a)
    nrm = np.array([-u[1], u[0]])
    q = np.roll(poly, -1, axis=0)
    on = (np.abs((poly - a) @ nrm) <= eps) & (np.abs((q - a) @ nrm) <= eps) & ((q - poly) @ u > 0)
    runs = sorted(zip(((poly[on] - a) @ u).tolist(), ((q[on] - a) @ u).tolist(), strict=True))
    merged: list[list[float]] = []
    for s0, s1 in runs:
        if merged and s0 <= merged[-1][1] + eps:
            merged[-1][1] = max(merged[-1][1], s1)
        else:
            merged.append([s0, s1])
    sp = float((pt - a) @ u)
    lo, hi = next(((r0, r1) for r0, r1 in merged if r0 - eps <= sp <= r1 + eps), (0.0, float(np.linalg.norm(b - a))))
    centre = a + float(np.clip(sp, lo, hi)) * u
    normal = np.array([u[1], -u[0]]) if area2 > 0 else -np.array([u[1], -u[0]])
    return _clean_angle(math.degrees(math.atan2(normal[1], normal[0]))), float(hi - lo), centre


def _infer(polygons: list[tuple[np.ndarray, tuple[int, int]]], labels: Sequence[Label], *,
           label_layers, port_layers, tol: float, electrical_layers, where: str, grid: float | None = None):
    """Ports from labels on polygon edges; returns (ports, consumed label indices)."""
    by_layer: dict[tuple[int, int], list[np.ndarray]] = {}
    for pts, lay in polygons:
        by_layer.setdefault(_layer(lay), []).append(np.asarray(pts, dtype=float))
    merged_cache: dict[tuple[int, int], list[np.ndarray]] = {}
    elec = {_layer(v) for v in electrical_layers}
    wanted = None if label_layers is None else {_layer(v) for v in label_layers}
    ports: dict[str, Port] = {}
    consumed: list[int] = []
    for idx, lbl in enumerate(labels):
        if wanted is not None and lbl.layer not in wanted:
            continue
        layers = [lbl.layer] if port_layers is None else [_layer(v) for v in port_layers]
        pt = np.asarray(lbl.position, dtype=float)
        hits = []
        for lay in layers:
            if lay not in merged_cache:
                merged_cache[lay] = _merged(by_layer.get(lay, []))
            for poly in merged_cache[lay]:
                if len(poly) < 3:
                    continue
                lo, hi = poly.min(axis=0) - tol, poly.max(axis=0) + tol
                if np.any(pt < lo) or np.any(pt > hi):
                    continue
                dist, _t = _segments_distance(pt, poly, np.roll(poly, -1, axis=0))
                for e in np.nonzero(dist <= tol)[0]:
                    hits.append((float(dist[e]), lay, poly, int(e)))
        if not hits:
            continue
        hits.sort(key=lambda h: h[0])
        dirs = []
        for _d, _lay, poly, e in hits:
            d = poly[(e + 1) % len(poly)] - poly[e]
            dirs.append(d / np.linalg.norm(d))
        if any(abs(dirs[0][0] * d[1] - dirs[0][1] * d[0]) > 1e-5 for d in dirs[1:]):
            warnings.warn(f"{where}: label {lbl.text!r} at {lbl.position} touches non-collinear edges (a corner); "
                          "its port orientation is ambiguous and it is kept as a label", stacklevel=4)
            continue
        if lbl.text in ports:
            warnings.warn(f"{where}: duplicate port label {lbl.text!r}; keeping the first", stacklevel=4)
            continue
        _d, lay, poly, e = hits[0]
        orientation, width, centre = _face(poly, e, pt)
        if float(np.linalg.norm(centre - pt)) <= (1.5 * grid if grid is not None else 1e-9):
            centre = pt  # the label is on the face (up to grid snapping): keep its exact position
        if grid is not None:
            centre = _norm(centre, grid)
            width = float(_norm(np.array([width]), grid)[0])
        ports[lbl.text] = Port(lbl.text, (float(centre[0]), float(centre[1])), orientation, width, lay,
                               ELECTRICAL if lay in elec else OPTICAL)
        consumed.append(idx)
    return ports, consumed


def infer_ports(
    cell: Cell,
    *,
    labels: Iterable[Label] | None = None,
    label_layers: Sequence[Sequence[int]] | None = None,
    port_layers: Sequence[Sequence[int]] | None = None,
    tol: float = 0.01,
    electrical_layers: Sequence[Sequence[int]] = (),
) -> dict[str, Port]:
    """Ports from text labels that sit on an edge of the cell's merged geometry.

    Parameters
    ----------
    cell
        Geometry to search (flattened, all references included).
    labels
        Labels to consider; defaults to ``cell.get_labels()`` for a
        :class:`~photonix.layout.labels.LabeledCell`.
    label_layers
        Only labels on these layers; ``None`` uses all.
    port_layers
        Layers whose edges may hold ports; ``None`` means each label's own layer.
    tol
        Maximum label-to-edge distance in µm.
    electrical_layers
        Ports found on these layers are electrical.

    Notes
    -----
    Orientation is the outward normal of the edge, width the length of the
    straight boundary run through the label, centre the label projected onto
    it. See the module documentation for what this convention cannot recover.

    Examples
    --------
    >>> from photonix.layout import components as lc
    >>> from photonix.layout.labels import LabeledCell
    >>> c = LabeledCell.from_cell(lc.straight(10.0))
    >>> _ = c.add_label("o2", (10.0, 0.0), (1, 0))
    >>> p = infer_ports(c)["o2"]
    >>> p.center, p.orientation, p.width
    ((10.0, 0.0), 0.0, 0.5)
    """
    if not math.isfinite(tol) or tol < 0:
        raise ValueError("tol must be non-negative and finite")
    if labels is None:
        labels = cell.get_labels() if isinstance(cell, LabeledCell) else []
    ports, _ = _infer(cell.get_polygons(), list(labels), label_layers=label_layers, port_layers=port_layers,
                      tol=tol, electrical_layers=electrical_layers, where=f"cell {cell.name!r}")
    return ports


def _gds_property(element, attr: int) -> str | None:
    value = element.get_gds_property(attr)
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return str(value).rstrip("\x00")


class _Reader:
    def __init__(self, lib, *, grid: float, side: dict | None, port_convention: str, pin_layers: dict,
                 port_layers, label_layers, tol: float, electrical_layers, magnification: str):
        self.lib = lib
        self.grid = grid
        self.side = side or {}
        self.port_convention = port_convention
        self.pin_layers = pin_layers
        self.port_layers = port_layers
        self.label_layers = label_layers
        self.tol = tol
        self.electrical_layers = electrical_layers
        self.magnification = magnification
        self.memo: dict[str, LabeledCell] = {}
        self.visiting: set[str] = set()
        self.mag_memo: dict[tuple[str, float], LabeledCell] = {}
        self.stale: list[str] = []

    # -- elements -------------------------------------------------------- #
    def _pin(self, path):
        layers = list(getattr(path, "layers", ()))
        if len(layers) != 1:
            return None
        layer = (int(layers[0]), int(path.datatypes[0]))
        if layer not in self.pin_layers.values():
            return None
        if len(_offsets(path)) != 1:
            return None
        spine = np.asarray(path.spine(), dtype=float)
        if spine.shape != (2, 2) or np.array_equal(spine[0], spine[1]):
            return None
        widths = np.asarray(path.widths(), dtype=float).reshape(-1)
        if widths.size == 0 or not np.allclose(widths, widths[0]) or widths[0] <= 0:
            return None
        spine = _norm(spine, self.grid)
        return {"p1": spine[0], "p2": spine[1], "width": float(_norm(np.array([widths[0]]), self.grid)[0]),
                "layer": layer, "port_layer": _gds_property(path, PORT_LAYER_ATTR),
                "orientation": _gds_property(path, PORT_ORIENTATION_ATTR)}

    def _pin_ports(self, cell: LabeledCell, pins: list[dict], where: str) -> dict[str, Port]:
        ports: dict[str, Port] = {}
        used: set[int] = set()
        electrical = self.pin_layers["electrical"]
        flat = None
        for k, pin in enumerate(pins):
            mid = _norm(0.5 * (pin["p1"] + pin["p2"]), self.grid)
            vec = pin["p2"] - pin["p1"]
            length = float(np.linalg.norm(vec))
            orient = _clean_angle(math.degrees(math.atan2(vec[1], vec[0])))
            if pin["orientation"] is not None:
                try:
                    exact = float(pin["orientation"]) % 360.0
                except ValueError:
                    exact = None
                if exact is not None:
                    bound = math.degrees(math.asin(min(1.0, 2.0 * math.sqrt(2.0) * self.grid / length))) + 1e-9
                    diff = abs((exact - orient + 180.0) % 360.0 - 180.0)
                    if diff <= bound:
                        orient = _clean_angle(exact)
            # name: the nearest unused TEXT on the pin layer within reach of the pin centre
            best, best_d = None, max(self.tol, 0.5 * length)
            for i, lbl in enumerate(cell.labels):
                if i in used or lbl.layer != pin["layer"]:
                    continue
                d = float(np.linalg.norm(np.asarray(lbl.position) - mid))
                if d <= best_d:
                    best, best_d = i, d
            if best is None:
                name = f"pin{k + 1}"
                warnings.warn(f"{where}: pin at {tuple(mid)} has no name label; calling it {name!r}", stacklevel=4)
            else:
                used.add(best)
                name = cell.labels[best].text
            if name in ports:
                warnings.warn(f"{where}: duplicate pin name {name!r}; keeping the first", stacklevel=4)
                continue
            layer = None
            if pin["port_layer"]:
                try:
                    lay_s, dt_s = pin["port_layer"].split("/")
                    layer = (int(lay_s), int(dt_s))
                except ValueError:
                    layer = None
            if layer is None:
                if flat is None:
                    flat = cell.get_polygons()
                layer = self._layer_under(flat, mid - 1e-3 * vec / length)
            ports[name] = Port(name, (float(mid[0]), float(mid[1])), orient, pin["width"], layer,
                               ELECTRICAL if pin["layer"] == electrical else OPTICAL)
        cell.labels = [lbl for i, lbl in enumerate(cell.labels) if i not in used]
        return ports

    def _layer_under(self, polygons, pt: np.ndarray) -> tuple[int, int]:
        gdstk = _gdstk()
        allowed = None if self.port_layers is None else [_layer(v) for v in self.port_layers]
        best: tuple[float, tuple[int, int]] | None = None
        pins = set(self.pin_layers.values())
        for pts, lay in polygons:
            lay = _layer(lay)
            if lay in pins or (allowed is not None and lay not in allowed):
                continue
            if not gdstk.inside([tuple(pt)], [gdstk.Polygon(pts)])[0]:
                continue
            rank = float(allowed.index(lay)) if allowed is not None else abs(float(
                0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])))
            if best is None or rank < best[0]:
                best = (rank, lay)
        return best[1] if best is not None else (1, 0)

    def _magnified(self, child: LabeledCell, mag: float) -> LabeledCell:
        key = (child.name, mag)
        if key not in self.mag_memo:
            flat = child.flatten()
            scaled = LabeledCell(f"{child.name}_mag{mag:g}", model=GEOMETRY_ONLY)
            scaled.polygons = [(pts * mag, lay) for pts, lay in flat.polygons]
            for prt in flat.ports.values():
                scaled.add_port(prt.name, (prt.center[0] * mag, prt.center[1] * mag), prt.orientation,
                                prt.width * mag, prt.layer, prt.port_type)
            scaled.labels = [Label(lb.text, (lb.position[0] * mag, lb.position[1] * mag), lb.layer, lb.rotation,
                                   lb.magnification * mag, lb.anchor) for lb in flat.labels]
            self.mag_memo[key] = scaled
        return self.mag_memo[key]

    # -- structures ------------------------------------------------------ #
    def convert(self, gcell) -> LabeledCell:
        gname = gcell.name
        if gname in self.memo:
            return self.memo[gname]
        if gname in self.visiting:
            raise ValueError(f"structure {gname!r} references itself (cyclic hierarchy)")
        self.visiting.add(gname)
        where = f"structure {gname!r}"
        entry = self.side.get(gname)
        if entry is not None and entry.get("content_hash") != _content_hash(gcell, self.grid):
            warnings.warn(f"{where} changed after its sidecar was written; its sidecar ports and model are ignored",
                          StaleSidecarWarning, stacklevel=3)
            self.stale.append(gname)
            entry = None
        cell = LabeledCell(entry["name"] if entry is not None else gname)
        g = self.grid
        for poly in gcell.polygons:
            for off in _offsets(poly):
                cell.add_polygon(_norm(np.asarray(poly.points, dtype=float) + off, g), (poly.layer, poly.datatype))
        pins = []
        for path in gcell.paths:
            pin = self._pin(path) if self.port_convention != "none" else None
            if pin is not None:
                pins.append(pin)
                continue
            for off in _offsets(path):
                for poly in path.to_polygons():
                    cell.add_polygon(_norm(np.asarray(poly.points, dtype=float) + off, g), (poly.layer, poly.datatype))
        for lbl in gcell.labels:
            for off in _offsets(lbl):
                pos = _norm(np.asarray(lbl.origin, dtype=float) + off, g)
                cell.labels.append(Label(str(lbl.text), (float(pos[0]), float(pos[1])), (lbl.layer, lbl.texttype),
                                         math.degrees(lbl.rotation), float(lbl.magnification), str(lbl.anchor)))
        used_names: set[str] = set()
        for gref in gcell.references:
            self._reference(cell, gref, used_names, where)

        conv = self.port_convention
        if entry is not None and conv in ("auto", "sidecar"):
            if pins:  # the markers duplicate the sidecar ports: consume them and their name labels
                self._pin_ports(cell, pins, where)
            for p in entry.get("ports", []):
                cell.add_port(p["name"], tuple(p["center"]), p["orientation"], p["width"], tuple(p["layer"]),
                              p["port_type"])
        elif pins:
            ports = self._pin_ports(cell, pins, where)
            if conv in ("auto", "pinrec"):
                cell.ports = ports
        if conv == "label" or (conv == "auto" and entry is None and not pins and cell.labels):
            found, consumed = _infer(cell.get_polygons(), cell.labels, label_layers=self.label_layers,
                                     port_layers=self.port_layers, tol=self.tol,
                                     electrical_layers=self.electrical_layers, where=where, grid=g)
            if found:
                cell.ports = found
                cell.labels = [lb for i, lb in enumerate(cell.labels) if i not in set(consumed)]

        if entry is not None:
            cell.model = entry.get("model")
            cell.model_settings = _decode(entry.get("model_settings") or {})
        else:
            cell.model = None if cell.references else GEOMETRY_ONLY
        self.visiting.discard(gname)
        self.memo[gname] = cell
        return cell

    def _reference(self, cell: LabeledCell, gref, used_names: set[str], where: str) -> None:
        target = gref.cell
        if isinstance(target, str) or not hasattr(target, "polygons"):
            name = target if isinstance(target, str) else getattr(target, "name", "?")
            raise ValueError(f"{where} references structure {name!r}, which is not defined in the file")
        child = self.convert(target)
        rotation = _clean_angle(math.degrees(gref.rotation))
        mirror = bool(gref.x_reflection)
        mag = float(gref.magnification)
        if abs(mag - 1.0) > 1e-12:
            if self.magnification == "error":
                raise ValueError(f"{where} places {target.name!r} with magnification {mag:g}; Photonix references "
                                 "cannot scale (pass magnification='flatten' to flatten it locally)")
            warnings.warn(f"{where} places {target.name!r} with magnification {mag:g}; it is flattened into a scaled, "
                          "geometry-only copy", stacklevel=4)
            child = self._magnified(child, mag)
        base = _gds_property(gref, INSTANCE_NAME_ATTR)
        rep = gref.repetition
        offsets = _offsets(gref)
        origin = np.asarray(gref.origin, dtype=float)
        if len(offsets) == 1:
            names: list[str | None] = [base]
        else:
            stem = base or target.name
            cols, rows = getattr(rep, "columns", None), getattr(rep, "rows", None)
            if cols and rows and cols * rows == len(offsets):
                names = [f"{stem}_{k // rows}_{k % rows}" for k in range(len(offsets))]
            else:
                names = [f"{stem}_{k}" for k in range(len(offsets))]
        for off, name in zip(offsets, names, strict=True):
            if name is not None:
                unique = name
                n = 1
                while unique in used_names:
                    unique = f"{name}${n}"
                    n += 1
                if unique != name:
                    warnings.warn(f"{where}: duplicate instance name {name!r} renamed {unique!r}", stacklevel=4)
                used_names.add(unique)
                name = unique
            pos = _norm(origin + off, self.grid)
            cell.add_ref(child, origin=(float(pos[0]), float(pos[1])), rotation=rotation, mirror=mirror, name=name)


def _load_sidecar(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or doc.get("format") != SIDECAR_FORMAT:
        raise ValueError(f"{path!r} is not a Photonix layout sidecar")
    if int(doc.get("version", 0)) > _SIDECAR_VERSION:
        raise ValueError(f"sidecar {path!r} has version {doc.get('version')}; this Photonix reads up to "
                         f"{_SIDECAR_VERSION}")
    return doc


def read_layout(
    path: str | os.PathLike[str],
    *,
    top: str | None = None,
    flatten: bool = False,
    sidecar: bool | str | os.PathLike[str] = "auto",
    port_convention: Literal["auto", "sidecar", "pinrec", "label", "none"] = "auto",
    pinrec_layer: Sequence[int] = PINREC_LAYER,
    electrical_pinrec_layer: Sequence[int] = PINREC_ELECTRICAL_LAYER,
    port_layers: Sequence[Sequence[int]] | None = None,
    label_layers: Sequence[Sequence[int]] | None = None,
    electrical_layers: Sequence[Sequence[int]] = (),
    tol: float = 0.01,
    magnification: Literal["flatten", "error"] = "flatten",
    fmt: Literal["gds", "oas"] | None = None,
) -> LabeledCell:
    """Read a GDSII/OASIS file into a hierarchical :class:`~photonix.layout.labels.LabeledCell` tree.

    Parameters
    ----------
    path
        Layout file (format from the extension unless ``fmt`` is given).
    top
        Name of the structure to return. ``None`` requires exactly one top
        structure and raises ``ValueError`` listing them otherwise.
    flatten
        Return the top cell flattened (labels kept).
    sidecar
        ``"auto"`` uses ``<path>.photonix.json`` when it exists, ``True``
        requires it, ``False`` ignores it, a path names it explicitly.
    port_convention
        Where ports come from: ``"auto"`` (per structure: a valid sidecar
        entry, else PinRec pins, else labels on edges), or only
        ``"sidecar"``, ``"pinrec"``, ``"label"``, or ``"none"`` (no ports;
        pin paths stay geometry). Recognised pin paths and their name labels
        are consumed as markers in every mode but ``"none"``.
    pinrec_layer, electrical_pinrec_layer
        Pin layers for optical and electrical ports.
    port_layers
        Layers on which ports are sought (label edges, pin layer inference);
        ``None`` means each label's own layer, or any layer under a pin.
    label_layers
        Only labels on these layers are considered as port labels.
    electrical_layers
        Label-inferred ports on these layers are electrical.
    tol
        Label-to-edge (and label-to-pin) distance tolerance in µm.
    magnification
        ``"flatten"`` (warn, flatten the magnified child into a scaled
        geometry-only copy) or ``"error"``.

    Returns
    -------
    LabeledCell
        Every structure becomes one shared ``LabeledCell``; labels that were
        not consumed as port markers stay in :attr:`LabeledCell.labels`.

    Examples
    --------
    >>> import os, tempfile, gdstk
    >>> p = os.path.join(tempfile.mkdtemp(), "nm.gds")
    >>> lib = gdstk.Library(unit=1e-9, precision=1e-9)         # drawn in nanometres
    >>> _ = lib.new_cell("wg").add(gdstk.rectangle((0, 0), (10000, 500)))
    >>> lib.write_gds(p)
    >>> read_layout(p).bbox()                                  # read in micrometres
    ((0.0, 0.0), (10.0, 0.5))
    """
    if port_convention not in ("auto", "sidecar", "pinrec", "label", "none"):
        raise ValueError(f"unknown port_convention {port_convention!r}")
    if magnification not in ("flatten", "error"):
        raise ValueError("magnification must be 'flatten' or 'error'")
    if not math.isfinite(tol) or tol < 0:
        raise ValueError("tol must be non-negative and finite")
    target = os.fspath(path)
    kind = _fmt(target, fmt)
    _unit, precision = layout_units(target, fmt=kind)
    if precision > 1e-9 * (1 + 1e-9):
        warnings.warn(f"{target!r} uses a {precision * 1e9:g} nm database grid, coarser than 1 nm", stacklevel=2)
    grid = precision / _UNIT

    side_doc: dict | None = None
    if sidecar is not False:
        if sidecar is True or sidecar == "auto":
            candidate = sidecar_path_for(target)
        else:
            candidate = os.fspath(cast("str | os.PathLike[str]", sidecar))
        if os.path.exists(candidate):
            side_doc = _load_sidecar(candidate)
        elif sidecar != "auto":
            raise FileNotFoundError(f"sidecar {candidate!r} not found")

    pin_layers = {"optical": _layer(pinrec_layer), "electrical": _layer(electrical_pinrec_layer)}
    if side_doc is not None and "pin_layers" in side_doc:
        recorded = side_doc["pin_layers"]
        if port_convention in ("auto", "sidecar"):
            pin_layers = {"optical": _layer(recorded["optical"]), "electrical": _layer(recorded["electrical"])}

    lib = _read_lib(target, kind)
    cells = {c.name: c for c in lib.cells if hasattr(c, "polygons")}
    if not cells:
        raise ValueError(f"{target!r} contains no structures")
    if top is None:
        tops = sorted(c.name for c in lib.top_level() if hasattr(c, "polygons"))
        if len(tops) != 1:
            raise ValueError(f"{target!r} has {len(tops)} top structures {tops}; choose one with top=...")
        top = tops[0]
    if top not in cells:
        raise ValueError(f"no structure {top!r} in {target!r}; structures: {sorted(cells)}")
    reader = _Reader(lib, grid=grid, side=(side_doc or {}).get("structures"), port_convention=port_convention,
                     pin_layers=pin_layers, port_layers=port_layers, label_layers=label_layers, tol=tol,
                     electrical_layers=electrical_layers, magnification=magnification)
    out = reader.convert(cells[top])
    return out.flatten() if flatten else out
