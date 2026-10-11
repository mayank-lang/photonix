"""Packaging and test cells: fibre-array grating rows, pad arrays, die frames and test labels.

Fibre array
-----------
:func:`fiber_array` places ``n`` grating couplers with their fibre ports
exactly ``pitch`` apart (127 µm is the standard V-groove fibre-array pitch)
and their waveguide ports facing the device side. With ``loopback=True`` the
two outer couplers are joined by a waveguide, the alignment loopback used to
peak the coupling of a fibre array actively before measuring the devices on
the inner channels. The loop runs *around* the coupler row (up, outward,
down past the gratings, under the whole row and back) so it crosses none of
the device waveguides, with six filleted 90-degree corners; its centre-line
length is closed-form::

    L = sum(Manhattan segments) - 12 d + 6 L_bend

with ``d`` the bend's corner offset (the radius for a circular bend) and
``L_bend`` its arc length (``pi r / 2``). The route records this length, so
extraction simulates the drawn loop.

Automated-test labels
---------------------
:func:`add_test_labels` writes the label used by open-access automated
probe stations (SiEPIC / edX "Silicon Photonics Design, Fabrication and Data
Analysis"; L. Chrostowski and M. Hochberg, *Silicon Photonics Design*,
Cambridge University Press, 2015): a GDS ``TEXT`` reading
``opt_in_<TE|TM>_<wavelength nm>_device_<user>_<device>`` on the text layer
(10/0) at the fibre port of the input grating coupler. The station finds the
other couplers of the device by the array pitch, which the function checks.
The exact rules (which coupler is the input, the allowed characters, the
layer) belong to the test facility; check them against its current
documentation before tape-out.

Die frame
---------
:func:`die_frame` returns a :class:`~photonix.layout.placement.DieCell`: the
dicing lane as four rectangles whose area is exactly ``W H - (W - 2d)(H - 2d)``,
a keep-out band of marker geometry inside it, optional L-shaped corner marks,
an optional die ID drawn in polygon text, and optional facet-trench markers on
the edge-coupled sides. The marker layers default to unused numbers; map them
to the foundry's layers (deep trench, dicing, keep-out) before tape-out.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np

from .. import components as lc
from ..cell import ELECTRICAL, Cell, Port, Reference
from ..labels import TEXT_LAYER, Label, LabeledCell, _layer, text
from ..placement import DieCell, _side, reference_bbox
from ..routing import route_manhattan

__all__ = ["fiber_array", "loopback_length", "pad_array", "die_frame", "add_test_labels", "opt_in_label",
           "FIBER_ARRAY_PITCH", "TEST_LABEL_LAYER"]

#: Standard fibre-array (V-groove) pitch in µm.
FIBER_ARRAY_PITCH = 127.0
#: SiEPIC / EBeam text layer that automated-test labels are written on.
TEST_LABEL_LAYER = (10, 0)


# --------------------------------------------------------------------------- #
# fibre array
# --------------------------------------------------------------------------- #
def _rotate_placements(cell: Cell, angle: float) -> None:
    """Rotate every reference, polygon and port of ``cell`` about the origin by ``angle`` degrees."""
    if angle % 360.0 == 0.0:
        return
    th = math.radians(angle)
    c, s = math.cos(th), math.sin(th)

    def rot(p):
        p = np.asarray(p, dtype=float)
        return np.stack([c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1]], axis=-1)

    for ref in cell.references:
        o = rot(ref.origin)
        ref.origin = (float(o[0]), float(o[1]))
        ref.rotation = (ref.rotation + angle) % 360.0
    cell.polygons = [(rot(p), lay) for p, lay in cell.polygons]
    cell.ports = {k: p.moved(0.0, 0.0, angle) for k, p in cell.ports.items()}


_ROUTE_NPTS = 48  # route_manhattan's default bend sampling; the Euler corner offset depends on it


def _bend_geometry(radius: float, bend: str, p: float) -> tuple[float, float]:
    """(corner offset d, arc length) of the 90-degree bend ``route_manhattan`` draws."""
    if bend == "circular":
        return radius, 0.5 * math.pi * radius
    if bend == "euler":
        from .._geometry import corner_offset, euler_curve

        return corner_offset(euler_curve(radius, 90.0, p, _ROUTE_NPTS)), 0.5 * math.pi * radius * (1.0 + p)
    raise ValueError("bend must be 'circular' or 'euler'")


def loopback_length(corners: Sequence[Sequence[float]], start: Sequence[float], end: Sequence[float], *,
                    radius: float = 10.0, bend: str = "circular", p: float = 0.5) -> float:
    """Closed-form centre-line length of a filleted Manhattan path through ``corners``.

    ``sum(|segments|) - 2 d n + n L_bend`` for ``n`` 90-degree corners.

    Examples
    --------
    >>> import math
    >>> L = loopback_length([(0, 10), (-20, 10)], (0, 0), (-20, -5), radius=10.0)
    >>> abs(L - (10 + 20 + 15 - 40 + math.pi * 10)) < 1e-12
    True
    """
    pts = np.asarray([start, *corners, end], dtype=float)
    d, arc = _bend_geometry(radius, bend, p)
    n = len(corners)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)) - 2.0 * d * n + n * arc)


def fiber_array(
    n: int = 4,
    *,
    pitch: float = FIBER_ARRAY_PITCH,
    gc: Cell | None = None,
    loopback: bool = True,
    radius: float = 10.0,
    bend: str = "circular",
    orientation: float = 90.0,
    fiber_port: str = "o2",
    wg_port: str = "o1",
    clearance: float = 5.0,
    width: float | None = None,
    name: str = "fiber_array",
) -> Cell:
    """A row of ``n`` grating couplers on the fibre-array pitch, optionally with an alignment loopback.

    In the canonical frame (``orientation=90``) channel ``k``'s fibre port is
    at exactly ``(k * pitch, 0)`` and every waveguide port faces +y. The cell
    is then rotated about the origin so the waveguide ports face
    ``orientation``.

    Parameters
    ----------
    n
        Number of couplers (fibres). With ``loopback`` channels 0 and
        ``n - 1`` form the loop, so ``n >= 2``.
    gc
        Grating-coupler cell (default :func:`photonix.layout.components.grating_coupler`)
        with a waveguide port ``wg_port`` and a fibre port ``fiber_port``.
    radius, bend
        Loopback bend radius and type (``"circular"`` or ``"euler"``).
    clearance
        Minimum gap between the loopback waveguide and the coupler bodies.
    width
        Loopback waveguide width (default: the coupler's waveguide port width).

    Returns
    -------
    Cell
        References ``gc0`` ... ``gc{n-1}`` (and ``loopback``), and optical
        ports ``o1``, ``o2``, ... on the waveguide ports of the device
        channels (channels ``1 .. n-2`` with a loopback, all otherwise), left
        to right in the canonical frame. Fibre port ``k`` is
        ``cell.get_ports()[f"gc{k}/{fiber_port}"]``.

    Examples
    --------
    >>> fa = fiber_array(4)
    >>> [round(fa.get_ports()[f"gc{k}/o2"].center[0], 9) for k in range(4)]
    [0.0, 127.0, 254.0, 381.0]
    >>> sorted(fa.ports), fa.ports["o1"].orientation
    (['o1', 'o2'], 90.0)
    """
    n = int(n)
    if n < (2 if loopback else 1):
        raise ValueError(f"a fibre array needs n >= {2 if loopback else 1} couplers")
    if not math.isfinite(pitch) or pitch <= 0:
        raise ValueError("pitch must be positive and finite")
    gc = lc.grating_coupler() if gc is None else gc
    for port in (wg_port, fiber_port):
        if port not in gc.ports:
            raise ValueError(f"grating coupler {gc.name!r} has no port {port!r}; ports: {sorted(gc.ports)}")
    rot = (90.0 - gc.ports[wg_port].orientation) % 360.0
    fib = gc.ports[fiber_port].moved(0.0, 0.0, rot)
    cell = Cell(name)
    refs: list[Reference] = []
    for k in range(n):
        origin = (k * pitch - fib.center[0], -fib.center[1])
        refs.append(cell.add_ref(gc, origin=origin, rotation=rot, name=f"gc{k}"))
    for k in range(n):
        p = refs[k].port(fiber_port)  # land the fibre port exactly on the pitch (no trigonometric residue)
        refs[k].origin = (refs[k].origin[0] + (k * pitch - p.center[0]), refs[k].origin[1] - p.center[1])
    channels = range(1, n - 1) if loopback else range(n)
    for i, k in enumerate(channels):
        cell.expose_port(refs[k], wg_port, f"o{i + 1}")

    if loopback:
        p0, pn = refs[0].port(wg_port), refs[-1].port(wg_port)
        w = p0.width if width is None else float(width)
        d, _arc = _bend_geometry(radius, bend, 0.5)
        boxes = [reference_bbox(r) for r in refs]
        y_top = max(p0.center[1], pn.center[1]) + d
        y_bot = min(b[0][1] for b in boxes) - clearance - w / 2.0
        if y_top - y_bot < 2.0 * d:
            y_bot = y_top - 2.0 * d
        x_left = min(p0.center[0] - 2.0 * d, boxes[0][0][0] - clearance - w / 2.0 - d)
        x_right = max(pn.center[0] + 2.0 * d, boxes[-1][1][0] + clearance + w / 2.0 + d)
        corners = [(p0.center[0], y_top), (x_left, y_top), (x_left, y_bot), (x_right, y_bot),
                   (x_right, y_top), (pn.center[0], y_top)]
        loop = route_manhattan(p0, pn, radius=radius, bend=bend, width=w, waypoints=corners, npts=_ROUTE_NPTS)
        loop.name = "loopback"
        cell.add_ref(loop, name="loopback")

    _rotate_placements(cell, orientation - 90.0)
    return cell


# --------------------------------------------------------------------------- #
# pads
# --------------------------------------------------------------------------- #
def pad_array(n: int = 8, *, pitch: float = 100.0, size: float = 80.0, layer: Sequence[int] = (11, 0),
              orientation: float = 90.0, name: str = "pad_array") -> Cell:
    """``n`` square bond pads on a row ``pitch`` apart, with electrical ports ``e1 .. en``.

    Pad ``k`` is centred at ``(k * pitch, 0)``; its port sits at the centre,
    faces ``orientation`` and is ``size`` wide (the probe/bond landing).

    Examples
    --------
    >>> pa = pad_array(3, pitch=150.0)
    >>> [p.center for p in pa.ports.values()], pa.ports["e1"].port_type
    ([(0.0, 0.0), (150.0, 0.0), (300.0, 0.0)], 'electrical')
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    if not math.isfinite(pitch) or not math.isfinite(size) or size <= 0 or pitch < size:
        raise ValueError("need 0 < size <= pitch")
    pad = lc.pad(size=size, layer=_layer(layer))
    cell = Cell(name)
    for k in range(n):
        cell.add_ref(pad, origin=(k * pitch, 0.0), name=f"pad{k}")
        cell.add_port(f"e{k + 1}", (k * pitch, 0.0), orientation, size, _layer(layer), ELECTRICAL)
    return cell


# --------------------------------------------------------------------------- #
# die frame
# --------------------------------------------------------------------------- #
def _rect(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def die_frame(
    size: tuple[float, float] = (5000.0, 5000.0),
    *,
    dicing_lane: float = 100.0,
    keepout: float = 50.0,
    frame_layer: Sequence[int] = (99, 0),
    keepout_layer: Sequence[int] | None = (98, 0),
    die_id: str | None = None,
    text_layer: Sequence[int] = TEXT_LAYER,
    text_size: float | None = None,
    corner_marks: bool = True,
    corner_mark_layer: Sequence[int] = (99, 1),
    corner_mark_size: tuple[float, float] = (100.0, 10.0),
    facet_sides: Sequence[str] = (),
    facet_trench_layer: Sequence[int] | None = None,
    facet_trench_width: float = 30.0,
    name: str = "die",
) -> DieCell:
    """A die outline with dicing lane, keep-out band, corner marks, die ID and facet trenches.

    Parameters
    ----------
    size
        Die outline ``(W, H)`` in µm, lower-left corner at the origin.
    dicing_lane
        Lane width ``d`` inside the outline (four rectangles on ``frame_layer``).
    keepout
        Band inside the lane that devices must avoid (marker rectangles on
        ``keepout_layer``; ``None`` draws none). Edge couplers on
        ``facet_sides`` are allowed through it.
    die_id
        Drawn (polygon) text in the lower-left corner of the keep-out band,
        ``text_size`` high (default ``0.4 * keepout``).
    corner_marks
        L-shaped marks (arm length, arm width ``corner_mark_size``) in the
        inner corners of the lane on ``corner_mark_layer``.
    facet_sides
        Edge-coupled sides (``"N"``, ``"E"``, ``"S"``, ``"W"``).
    facet_trench_layer, facet_trench_width
        When a layer is given, a trench marker ``facet_trench_width`` wide
        along each facet side, from the outline inward; the optical facet is
        then its inner wall (see :meth:`DieCell.facet_position`).

    Examples
    --------
    >>> die = die_frame((2000.0, 1000.0), dicing_lane=100.0)
    >>> lane = sum(np.ptp(p[:, 0]) * np.ptp(p[:, 1]) for p, lay in die.polygons if lay == (99, 0))
    >>> float(lane) == 2000 * 1000 - 1800 * 800
    True
    """
    w, h = float(size[0]), float(size[1])
    d, k = float(dicing_lane), float(keepout)
    if not all(math.isfinite(v) for v in (w, h, d, k)) or w <= 0 or h <= 0 or d < 0 or k < 0:
        raise ValueError("size must be positive, dicing_lane and keepout non-negative")
    if 2.0 * (d + k) >= min(w, h):
        raise ValueError("dicing lane and keep-out leave no usable area")
    sides = tuple(dict.fromkeys(_side(s) for s in facet_sides))
    trench = float(facet_trench_width) if facet_trench_layer is not None and sides else 0.0
    if trench < 0 or trench > d + k:
        raise ValueError("facet_trench_width must lie within the dicing lane plus keep-out band")
    die = DieCell(name, die_size=(w, h), dicing_lane=d, keepout=k, facet_sides=sides, facet_trench_width=trench)
    fl = _layer(frame_layer)
    if d > 0:
        for r in (_rect(0, 0, w, d), _rect(0, h - d, w, h), _rect(0, d, d, h - d), _rect(w - d, d, w, h - d)):
            die.add_polygon(r, fl)
    if keepout_layer is not None and k > 0:
        kl = _layer(keepout_layer)
        a, b = d, d + k
        for r in (_rect(a, a, w - a, b), _rect(a, h - b, w - a, h - a), _rect(a, b, b, h - b),
                  _rect(w - b, b, w - a, h - b)):
            die.add_polygon(r, kl)
    if corner_marks:
        arm, t = float(corner_mark_size[0]), float(corner_mark_size[1])
        if not (0 < t < arm) or arm > 0.5 * min(w, h) - d:
            raise ValueError("corner_mark_size must be (arm, width) with 0 < width < arm and arms that fit")
        base = np.array([(0, 0), (arm, 0), (arm, t), (t, t), (t, arm), (0, arm)], dtype=float)
        for sx, sy, cx, cy in ((1, 1, d, d), (-1, 1, w - d, d), (1, -1, d, h - d), (-1, -1, w - d, h - d)):
            pts = base * np.array([sx, sy]) + np.array([cx, cy])
            die.add_polygon(pts if sx * sy > 0 else pts[::-1], _layer(corner_mark_layer))
    if die_id is not None:
        size_ = 0.4 * k if text_size is None else float(text_size)
        if size_ <= 0:
            raise ValueError("die_id needs a positive text_size when keepout is 0")
        gap = (float(corner_mark_size[1]) if corner_marks else 0.0) + 0.1 * max(k, size_)
        label = text(die_id, size=size_, layer=text_layer, position=(d + gap, d + gap), anchor="sw")
        die.polygons.extend(label.polygons)
    if trench > 0 and facet_trench_layer is not None:
        tl = _layer(facet_trench_layer)
        spans = {"E": _rect(w - trench, 0, w, h), "W": _rect(0, 0, trench, h),
                 "N": _rect(0, h - trench, w, h), "S": _rect(0, 0, w, trench)}
        for side in sides:
            die.add_polygon(spans[side], tl)
    return die


# --------------------------------------------------------------------------- #
# automated-test labels
# --------------------------------------------------------------------------- #
_USER = re.compile(r"^[A-Za-z0-9]+$")
_DEVICE = re.compile(r"^[A-Za-z0-9_\-]+$")


def opt_in_label(device: str, user: str, *, pol: str = "TE", wl_nm: int = 1550) -> str:
    """The automated-test label string ``opt_in_<pol>_<wl_nm>_device_<user>_<device>``.

    ``user`` must be alphanumeric (the station splits the label on ``_``),
    ``device`` alphanumeric with ``_`` or ``-``.

    Examples
    --------
    >>> opt_in_label("MZI1", "LukasC")
    'opt_in_TE_1550_device_LukasC_MZI1'
    """
    if pol not in ("TE", "TM"):
        raise ValueError("pol must be 'TE' or 'TM'")
    if isinstance(wl_nm, bool) or int(wl_nm) != wl_nm or not 100 <= int(wl_nm) <= 10000:
        raise ValueError("wl_nm must be an integer wavelength in nm, e.g. 1550 or 1310")
    if not _USER.match(user or ""):
        raise ValueError(f"user {user!r} must be non-empty and alphanumeric (no '_', spaces or punctuation)")
    if not _DEVICE.match(device or ""):
        raise ValueError(f"device {device!r} must be non-empty and use only letters, digits, '_' or '-'")
    return f"opt_in_{pol}_{int(wl_nm)}_device_{user}_{device}"



def add_test_labels(
    top: Cell,
    *,
    gc_refs: Sequence[Reference | Port],
    device: str,
    user: str,
    pol: str = "TE",
    wl_nm: int = 1550,
    layer: Sequence[int] = TEST_LABEL_LAYER,
    fiber_port: str = "o2",
    pitch: float | None = FIBER_ARRAY_PITCH,
    tol: float = 1e-3,
) -> LabeledCell:
    """Add the automated-test ``opt_in`` label at the input grating coupler's fibre port.

    Parameters
    ----------
    top
        Cell to label. A :class:`LabeledCell` gets the label in place and is
        returned; any other cell is wrapped in a new ``LabeledCell`` that
        shares its geometry and references.
    gc_refs
        The device's grating couplers, input first: references placed in
        ``top`` (their ``fiber_port`` is used) or ports already in ``top``'s
        coordinates.
    pitch
        When given, every coupler's fibre port must lie on one line with the
        input, at a whole multiple of ``pitch`` from it (within ``tol``), as
        the station expects; ``None`` skips the check.

    Examples
    --------
    >>> from photonix.layout import Cell
    >>> top = Cell("chip")
    >>> _ = top.add_ref(fiber_array(3, loopback=False), origin=(100.0, 50.0), name="fa")
    >>> fibres = [top.get_ports()[f"fa/gc{k}/o2"] for k in range(3)]     # nested: pass ports
    >>> labeled = add_test_labels(top, gc_refs=fibres, device="ring1", user="me")
    >>> labeled.labels[0].text, labeled.labels[0].position
    ('opt_in_TE_1550_device_me_ring1', (100.0, 50.0))
    """
    string = opt_in_label(device, user, pol=pol, wl_nm=wl_nm)
    if not gc_refs:
        raise ValueError("gc_refs needs at least the input grating coupler")
    points = []
    for item in gc_refs:
        if isinstance(item, Port):
            points.append(np.asarray(item.center, dtype=float))
        elif isinstance(item, Reference):
            if not any(item is r for r in top.references):
                raise ValueError("gc_refs references must be placed directly in top (pass ports for nested ones)")
            points.append(np.asarray(item.port(fiber_port).center, dtype=float))
        else:
            raise TypeError("gc_refs entries must be References or Ports")
    if pitch is not None and len(points) > 1:
        base = points[0]
        far = max(points[1:], key=lambda q: float(np.linalg.norm(q - base)))
        axis = (far - base) / max(float(np.linalg.norm(far - base)), 1e-300)
        for q in points[1:]:
            v = q - base
            off_line = abs(float(axis[0] * v[1] - axis[1] * v[0]))
            steps = float(np.linalg.norm(v)) / pitch
            if off_line > tol or abs(steps - round(steps)) * pitch > tol or round(steps) == 0:
                raise ValueError(f"grating coupler at {tuple(q)} is not on the {pitch:g} um fibre-array pitch "
                                 f"from the input at {tuple(base)}")
    out = top if isinstance(top, LabeledCell) else LabeledCell.from_cell(top)
    out.labels.append(Label(string, (float(points[0][0]), float(points[0][1])), _layer(layer)))
    return out
