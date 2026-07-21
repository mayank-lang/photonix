"""Layout primitives: ``Port``, ``Reference`` and ``Cell``.

A :class:`Cell` holds polygons (each tagged with a GDS layer), child references to
other cells (with placement transforms), and named :class:`Port` terminals used
for routing and netlist extraction. Geometry is plain NumPy — layout is not part
of the differentiable path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Port", "Reference", "Cell"]


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
    """

    name: str
    center: tuple[float, float] = (0.0, 0.0)
    orientation: float = 0.0
    width: float = 0.5
    layer: tuple[int, int] = (1, 0)

    def moved(self, dx: float, dy: float, rot_deg: float = 0.0, mirror: bool = False) -> Port:
        """Return a copy transformed by mirror -> rotation -> translation."""
        x, y = self.center
        o = self.orientation
        if mirror:
            y, o = -y, -o
        th = np.deg2rad(rot_deg)
        xr = x * np.cos(th) - y * np.sin(th)
        yr = x * np.sin(th) + y * np.cos(th)
        return Port(self.name, (xr + dx, yr + dy), (o + rot_deg) % 360.0, self.width, self.layer)


@dataclass
class Reference:
    """A placed instance of another :class:`Cell`."""

    cell: Cell
    origin: tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    mirror: bool = False
    name: str | None = None


@dataclass
class Cell:
    """A hierarchical layout cell.

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

    # -- builders ----------------------------------------------------------- #
    def add_polygon(self, points, layer: tuple[int, int] = (1, 0)) -> Cell:
        self.polygons.append((np.asarray(points, dtype=float), tuple(layer)))
        return self

    def add_ref(self, cell: Cell, origin=(0.0, 0.0), rotation=0.0, mirror=False, name=None) -> Reference:
        ref = Reference(cell, tuple(origin), float(rotation), bool(mirror), name)
        self.references.append(ref)
        return ref

    def add_port(self, name, center=(0.0, 0.0), orientation=0.0, width=0.5, layer=(1, 0)) -> Cell:
        self.ports[name] = Port(name, tuple(center), float(orientation), float(width), tuple(layer))
        return self

    # -- queries ------------------------------------------------------------ #
    def get_polygons(self) -> list[tuple[np.ndarray, tuple[int, int]]]:
        """Flattened polygons of this cell and all references (with transforms)."""
        out = list(self.polygons)
        for ref in self.references:
            th = np.deg2rad(ref.rotation)
            ox, oy = ref.origin
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            for pts, layer in ref.cell.get_polygons():
                p = pts.copy()
                if ref.mirror:
                    p[:, 1] = -p[:, 1]
                p = p @ R.T
                p[:, 0] += ox
                p[:, 1] += oy
                out.append((p, layer))
        return out

    def get_ports(self) -> dict[str, Port]:
        """All ports including those of references (namespaced ``ref.name/port``)."""
        out = dict(self.ports)
        for i, ref in enumerate(self.references):
            tag = ref.name or f"ref{i}"
            for pn, prt in ref.cell.ports.items():
                out[f"{tag}/{pn}"] = prt.moved(ref.origin[0], ref.origin[1], ref.rotation, ref.mirror)
        return out

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
