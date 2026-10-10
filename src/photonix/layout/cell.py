"""Layout primitives: ``Port``, ``Reference`` and ``Cell``.

A :class:`Cell` holds polygons (each tagged with a GDS layer), child references to
other cells (with placement transforms), and named :class:`Port` terminals used
for routing and netlist extraction. Geometry is plain NumPy — layout is not part
of the differentiable path.

Every reference transform is applied in the fixed order *mirror about the x
axis -> rotate counter-clockwise -> translate*, identically for polygons, ports
and GDS export.

Circuit semantics
-----------------
A cell may declare the compact model its geometry represents through
:attr:`Cell.model` and :attr:`Cell.model_settings` (for example a 25 µm straight
records ``model="straight"`` and ``model_settings={"length": 25.0}``).
:func:`photonix.layout.extract_netlist` copies these into the netlist, so the
circuit simulates the *drawn* geometry rather than a model's defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

__all__ = ["Port", "Reference", "Cell", "OPTICAL", "ELECTRICAL"]

OPTICAL = "optical"
ELECTRICAL = "electrical"


@dataclass
class Port:
    """An optical/electrical terminal on a cell.

    Attributes
    ----------
    name : str
    center : tuple[float, float]
        Position in µm.
    orientation : float
        Outward direction in degrees (0 = +x, 90 = +y).
    width : float
        Port width in µm.
    layer : tuple[int, int]
        GDS (layer, datatype).
    port_type : str
        ``"optical"`` (default) or ``"electrical"``. Only optical ports take
        part in optical netlist extraction.
    """

    name: str
    center: tuple[float, float] = (0.0, 0.0)
    orientation: float = 0.0
    width: float = 0.5
    layer: tuple[int, int] = (1, 0)
    port_type: str = OPTICAL

    @property
    def direction(self) -> np.ndarray:
        """Unit vector of the outward orientation."""
        th = np.deg2rad(self.orientation)
        return np.array([np.cos(th), np.sin(th)])

    def moved(self, dx: float, dy: float, rot_deg: float = 0.0, mirror: bool = False) -> Port:
        """Return a copy transformed by mirror -> rotation -> translation."""
        x, y = self.center
        o = self.orientation
        if mirror:
            y, o = -y, -o
        th = np.deg2rad(rot_deg)
        xr = x * np.cos(th) - y * np.sin(th)
        yr = x * np.sin(th) + y * np.cos(th)
        return Port(self.name, (float(xr + dx), float(yr + dy)), (o + rot_deg) % 360.0,
                    self.width, self.layer, self.port_type)

    def flipped(self) -> Port:
        """Copy facing the opposite direction (useful as a routing target)."""
        return Port(self.name, self.center, (self.orientation + 180.0) % 360.0,
                    self.width, self.layer, self.port_type)


@dataclass
class Reference:
    """A placed instance of another :class:`Cell`."""

    cell: Cell
    origin: tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    mirror: bool = False
    name: str | None = None

    @property
    def ports(self) -> dict[str, Port]:
        """The referenced cell's ports in the parent cell's coordinates."""
        return {
            name: prt.moved(self.origin[0], self.origin[1], self.rotation, self.mirror)
            for name, prt in self.cell.ports.items()
        }

    def port(self, name: str) -> Port:
        """One transformed port (see :attr:`ports`)."""
        if name not in self.cell.ports:
            raise KeyError(f"Cell {self.cell.name!r} has no port {name!r}; ports: {sorted(self.cell.ports)}")
        return self.cell.ports[name].moved(self.origin[0], self.origin[1], self.rotation, self.mirror)

    def connect(self, port: str, destination: Port, *, mirror: bool | None = None) -> Reference:
        """Place this reference so ``port`` mates with ``destination``.

        After the call the transformed ``port`` sits on ``destination.center``
        and faces the opposite way (``orientation + 180``), which is how two
        abutting waveguide ports meet. ``mirror`` optionally changes the
        reference's mirroring first.

        Examples
        --------
        >>> from photonix.layout import Cell, components as lc
        >>> top = Cell("top")
        >>> a = top.add_ref(lc.straight(10.0), name="a")
        >>> b = top.add_ref(lc.bend_circular(radius=5.0), name="b").connect("o1", a.port("o2"))
        >>> tuple(round(v, 9) for v in b.port("o1").center)
        (10.0, 0.0)
        """
        if mirror is not None:
            self.mirror = bool(mirror)
        local = self.cell.ports[port] if port in self.cell.ports else None
        if local is None:
            raise KeyError(f"Cell {self.cell.name!r} has no port {port!r}; ports: {sorted(self.cell.ports)}")
        x, y = local.center
        o = local.orientation
        if self.mirror:
            y, o = -y, -o
        rot = (destination.orientation + 180.0 - o) % 360.0
        th = np.deg2rad(rot)
        xr = x * np.cos(th) - y * np.sin(th)
        yr = x * np.sin(th) + y * np.cos(th)
        self.rotation = float(rot)
        self.origin = (float(destination.center[0] - xr), float(destination.center[1] - yr))
        return self


def _transform_points(pts: np.ndarray, ref: Reference) -> np.ndarray:
    th = np.deg2rad(ref.rotation)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    p = pts.copy()
    if ref.mirror:
        p[:, 1] = -p[:, 1]
    p = p @ rot.T
    p[:, 0] += ref.origin[0]
    p[:, 1] += ref.origin[1]
    return p


@dataclass
class Cell:
    """A hierarchical layout cell.

    Attributes
    ----------
    name
        Cell name (also the GDS structure name; distinct cells sharing a name
        are disambiguated on export).
    polygons, references, ports
        Geometry, child placements and terminals.
    model
        Name of the compact model this cell's geometry represents, used by
        :func:`~photonix.layout.extract_netlist`. ``None`` means: use ``name``
        for a leaf cell, and recurse into a composite cell's references.
    model_settings
        Model parameters implied by the geometry (e.g. ``{"length": 25.0}``).

    Examples
    --------
    >>> c = Cell("wg")
    >>> _ = c.add_polygon([(0, -0.25), (10, -0.25), (10, 0.25), (0, 0.25)], layer=(1, 0))
    >>> _ = c.add_port("o1", center=(0, 0), orientation=180)
    >>> len(c.polygons), len(c.ports)
    (1, 1)
    """

    name: str = "cell"
    polygons: list[tuple[np.ndarray, tuple[int, int]]] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    ports: dict[str, Port] = field(default_factory=dict)
    model: str | None = None
    model_settings: dict[str, Any] = field(default_factory=dict)

    # -- builders ----------------------------------------------------------- #
    def add_polygon(self, points, layer: tuple[int, int] = (1, 0)) -> Cell:
        self.polygons.append((np.asarray(points, dtype=float), cast(tuple[int, int], tuple(layer))))
        return self

    def add_ref(self, cell: Cell, origin=(0.0, 0.0), rotation=0.0, mirror=False, name=None) -> Reference:
        if name is not None and any(ref.name == name for ref in self.references):
            raise ValueError(f"Reference name {name!r} already exists in cell {self.name!r}.")
        ref = Reference(cell, tuple(origin), float(rotation), bool(mirror), name)
        self.references.append(ref)
        return ref

    def add_port(self, name, center=(0.0, 0.0), orientation=0.0, width=0.5, layer=(1, 0),
                 port_type: str = OPTICAL) -> Cell:
        if port_type not in (OPTICAL, ELECTRICAL):
            raise ValueError("port_type must be 'optical' or 'electrical'")
        self.ports[name] = Port(name, tuple(center), float(orientation), float(width), tuple(layer), port_type)
        return self

    def expose_port(self, ref: Reference, port: str, name: str | None = None) -> Cell:
        """Promote a reference's (transformed) port to a port of this cell."""
        prt = ref.port(port)
        new = name or port
        if new in self.ports:
            raise ValueError(f"Port {new!r} already exists in cell {self.name!r}.")
        self.ports[new] = Port(new, prt.center, prt.orientation, prt.width, prt.layer, prt.port_type)
        return self

    # -- queries ------------------------------------------------------------ #
    def get_polygons(self) -> list[tuple[np.ndarray, tuple[int, int]]]:
        """Flattened polygons of this cell and all references (with transforms)."""
        out = list(self.polygons)
        for ref in self.references:
            for pts, layer in ref.cell.get_polygons():
                out.append((_transform_points(pts, ref), layer))
        return out

    def get_ports(self) -> dict[str, Port]:
        """All ports including those of references (namespaced ``ref.name/port``)."""
        out = dict(self.ports)
        for i, ref in enumerate(self.references):
            tag = ref.name or f"ref{i}"
            for pn, prt in ref.cell.get_ports().items():
                out[f"{tag}/{pn}"] = prt.moved(ref.origin[0], ref.origin[1], ref.rotation, ref.mirror)
        return out

    @property
    def layers(self) -> set[tuple[int, int]]:
        """Every GDS layer used by this cell's flattened geometry."""
        return {layer for _pts, layer in self.get_polygons()}

    def flatten(self, name: str | None = None) -> Cell:
        """A new reference-free cell with all polygons and this cell's own ports."""
        return Cell(name or self.name, [(p.copy(), lay) for p, lay in self.get_polygons()], [],
                    dict(self.ports), self.model, dict(self.model_settings))

    def bbox(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Axis-aligned bounding box ``((xmin, ymin), (xmax, ymax))``."""
        polys = self.get_polygons()
        if not polys:
            return ((0.0, 0.0), (0.0, 0.0))
        allp = np.concatenate([p for p, _ in polys], axis=0)
        return ((float(allp[:, 0].min()), float(allp[:, 1].min())),
                (float(allp[:, 0].max()), float(allp[:, 1].max())))

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Cell {self.name!r}: {len(self.polygons)} polys, {len(self.references)} refs, {len(self.ports)} ports>"
