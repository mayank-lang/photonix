"""Schematic-driven layout: placements + connections + routes <-> layout cells.

A :class:`SchematicSpec` is a :class:`~photonix.circuit.Netlist` of *layout
components* (instance -> component name and layout settings, abutting
connections, exposed ports) plus :class:`Placement` s and :class:`RouteSpec`
s. It reads and writes gdsfactory's "pic" YAML schema::

    name: two_mzis
    instances:
      m0: {component: mzi, settings: {delta_length: 20.0}}
      m1: {component: mzi, settings: {delta_length: 40.0}}
    placements:
      m0: {x: 0.0, y: 0.0}
      m1: {x: 200.0, y: 50.0, rotation: 0.0}
    routes:
      r0: {links: {"m0,o2": "m1,o1"}, settings: {radius: 10.0}, routing_strategy: route_auto}
    ports: {o1: "m0,o1", o2: "m1,o2"}

(``connections`` and the newer ``nets`` list form are accepted too).

:func:`build_layout` realizes a spec: (1) instantiate every component through
the PDK (``pdk.get_layout``) or :mod:`photonix.layout.components`; (2) apply
absolute placements -- with ``port``, the named port is put on ``(x, y)`` --
then ``dx``/``dy``; (3) place the remaining instances by
:meth:`~photonix.layout.Reference.connect` along ``connections`` in
breadth-first order from placed ones (an instance reachable from none is put
at the origin, as gdsfactory does), and reject any connection whose ports end
up more than ``tol`` apart or not facing each other (a conflicting cycle);
(4) route the ``routes`` links with the named router; (5) expose ``ports``.
:func:`schematic_from_layout` goes the other way for a flat top cell.

Transform convention: photonix references mirror about the x axis, then
rotate, then translate (gdsfactory's ``mirror`` flips about the y axis; a
gdsfactory ``mirror: true`` therefore corresponds to ``mirror: true`` with
``rotation + 180``). PyYAML is optional and imported only by
:meth:`SchematicSpec.to_yaml` / :meth:`SchematicSpec.from_yaml`.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .cell import OPTICAL, Cell, Port, Reference

__all__ = ["Placement", "RouteSpec", "SchematicSpec", "build_layout", "schematic_from_layout",
           "ROUTING_STRATEGIES"]

ROUTING_STRATEGIES = ("route_manhattan", "route_bundle", "route_auto", "route_astar")
Strategy = Literal["route_manhattan", "route_bundle", "route_auto", "route_astar"]


def _yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("SchematicSpec YAML I/O needs PyYAML: pip install pyyaml") from exc
    return yaml


@dataclass
class Placement:
    """Absolute placement of one instance (µm, degrees).

    ``port`` names the instance port anchored at ``(x, y)``; without it the
    instance origin goes there. ``dx``/``dy`` are added afterwards.
    """

    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    mirror: bool = False
    port: str | None = None
    dx: float = 0.0
    dy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Non-default fields only (gdsfactory style)."""
        out: dict[str, Any] = {"x": float(self.x), "y": float(self.y)}
        if self.rotation:
            out["rotation"] = float(self.rotation)
        if self.mirror:
            out["mirror"] = True
        if self.port is not None:
            out["port"] = self.port
        if self.dx:
            out["dx"] = float(self.dx)
        if self.dy:
            out["dy"] = float(self.dy)
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Placement:
        """Inverse of :meth:`to_dict` (unknown keys raise ``ValueError``)."""
        unknown = set(d) - {"x", "y", "rotation", "mirror", "port", "dx", "dy"}
        if unknown:
            raise ValueError(f"unsupported placement keys {sorted(unknown)}")
        return cls(float(d.get("x", 0.0)), float(d.get("y", 0.0)), float(d.get("rotation", 0.0)),
                   bool(d.get("mirror", False)), d.get("port"), float(d.get("dx", 0.0)), float(d.get("dy", 0.0)))


@dataclass
class RouteSpec:
    """A group of routed links ``{"a,o2": "b,o1"}`` with router settings."""

    links: dict[str, str] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    strategy: Strategy = "route_bundle"

    def __post_init__(self) -> None:
        from photonix.circuit.netlist_io import parse_instance_port

        if self.strategy not in ROUTING_STRATEGIES:
            raise ValueError(f"unknown routing strategy {self.strategy!r}; choose one of {ROUTING_STRATEGIES}")
        for a, b in self.links.items():
            parse_instance_port(a)
            parse_instance_port(b)
        self.links = dict(self.links)
        self.settings = dict(self.settings)

    def to_dict(self) -> dict[str, Any]:
        """gdsfactory route-group dictionary."""
        out: dict[str, Any] = {"links": dict(self.links)}
        if self.settings:
            out["settings"] = dict(self.settings)
        out["routing_strategy"] = self.strategy
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RouteSpec:
        """Inverse of :meth:`to_dict`."""
        return cls(dict(d.get("links", {})), dict(d.get("settings", {}) or {}),
                   d.get("routing_strategy", d.get("strategy", "route_bundle")))


@dataclass
class SchematicSpec:
    """Schematic intent of a layout: components, placements, connections, routes, ports.

    Examples
    --------
    >>> from photonix.circuit import Netlist
    >>> nl = Netlist(name="pair")
    >>> _ = nl.add("a", "straight", length=10.0).add("b", "straight", length=5.0)
    >>> _ = nl.connect(("a", "o2"), ("b", "o1"))
    >>> _ = nl.expose("in", ("a", "o1")).expose("out", ("b", "o2"))
    >>> spec = SchematicSpec(nl, placements={"a": Placement(x=3.0)})
    >>> cell = build_layout(spec)
    >>> cell.ports["out"].center
    (18.0, 0.0)
    >>> SchematicSpec.from_dict(spec.to_dict()) == spec
    True
    """

    netlist: Any  # photonix.circuit.Netlist (component names and layout settings)
    placements: dict[str, Placement] = field(default_factory=dict)
    routes: dict[str, RouteSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from photonix.circuit import Netlist

        if not isinstance(self.netlist, Netlist):
            raise TypeError("SchematicSpec.netlist must be a photonix.circuit.Netlist")
        self.placements = {k: v if isinstance(v, Placement) else Placement.from_dict(v)
                           for k, v in self.placements.items()}
        self.routes = {k: v if isinstance(v, RouteSpec) else RouteSpec.from_dict(v) for k, v in self.routes.items()}
        self.validate()

    @property
    def name(self) -> str | None:
        """The schematic (and built cell) name."""
        return self.netlist.name

    def route_links(self) -> list[tuple[str, tuple[str, str], tuple[str, str]]]:
        """``(route group, terminal_a, terminal_b)`` for every routed link."""
        from photonix.circuit.netlist_io import parse_instance_port

        return [(g, parse_instance_port(a), parse_instance_port(b))
                for g, r in self.routes.items() for a, b in r.links.items()]

    def validate(self) -> None:
        """Raise ``ValueError`` for unknown instances or terminals used twice."""
        nl = self.netlist
        nl.validate()
        unknown = sorted(set(self.placements) - set(nl.instances))
        if unknown:
            raise ValueError(f"placements reference unknown instances {unknown}")
        used = {t for pair in nl.connections.items() for t in pair} | set(nl.ports.values())
        for g, a, b in self.route_links():
            for t in (a, b):
                if t[0] not in nl.instances:
                    raise ValueError(f"route {g!r} references unknown instance {t[0]!r}")
                if t in used:
                    raise ValueError(f"route {g!r}: terminal {t} is already connected, exposed or routed")
                used.add(t)

    # -- serialization ------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """gdsfactory pic-YAML compatible dictionary."""
        from photonix.circuit.netlist_io import to_sax_netlist

        sax = to_sax_netlist(self.netlist)
        out: dict[str, Any] = {}
        if self.name is not None:
            out["name"] = self.name
        out["instances"] = sax["instances"]
        if self.placements:
            out["placements"] = {k: v.to_dict() for k, v in self.placements.items()}
        if sax["connections"]:
            out["connections"] = sax["connections"]
        if self.routes:
            out["routes"] = {k: v.to_dict() for k, v in self.routes.items()}
        out["ports"] = sax["ports"]
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SchematicSpec:
        """Inverse of :meth:`to_dict` (also reads gdsfactory's ``nets`` list)."""
        from photonix.circuit.netlist_io import from_sax_netlist

        body = {k: d[k] for k in ("instances", "connections", "nets", "ports") if k in d}
        nl = from_sax_netlist({**body, "instances": body.get("instances", {}), "info": {"name": d.get("name")}})
        return cls(nl, {k: Placement.from_dict(v) for k, v in dict(d.get("placements", {}) or {}).items()},
                   {k: RouteSpec.from_dict(v) for k, v in dict(d.get("routes", {}) or {}).items()})

    def to_yaml(self, path: str | os.PathLike[str] | None = None) -> str:
        """Write gdsfactory pic-YAML text (and the file ``path`` if given). Needs PyYAML."""
        text = _yaml().safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    @classmethod
    def from_yaml(cls, source: str | os.PathLike[str]) -> SchematicSpec:
        """Read pic-YAML text or a ``.yml``/``.yaml`` file. Needs PyYAML."""
        text = os.fspath(source)
        if "\n" not in text and os.path.exists(text):
            with open(text, encoding="utf-8") as fh:
                text = fh.read()
        data = _yaml().safe_load(text)
        if not isinstance(data, Mapping):
            raise ValueError("pic-YAML must contain a mapping")
        return cls.from_dict(data)


# --------------------------------------------------------------------------- #
# Schematic -> layout
# --------------------------------------------------------------------------- #
def _instantiate(component: str, settings: Mapping[str, Any], pdk: Any) -> Cell:
    if pdk is not None and component in getattr(pdk, "components", {}):
        return pdk.get_layout(component, **settings)
    from . import components as lc

    gen = getattr(lc, component, None)
    if gen is None or not callable(gen) or component.startswith("_"):
        where = f"PDK {pdk.name!r} or " if pdk is not None else ""
        raise KeyError(f"component {component!r} not found in {where}photonix.layout.components")
    return gen(**settings)


def _residual(a: Port, b: Port) -> tuple[float, float]:
    dist = float(np.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]))
    ang = abs(((a.orientation - b.orientation) % 360.0) - 180.0)
    return dist, ang


def _route_style(spec_settings: Mapping[str, Any], pdk: Any) -> dict[str, Any]:
    settings = dict(spec_settings)
    xs_name = settings.pop("cross_section", None)
    if pdk is not None and getattr(pdk, "cross_sections", None):
        xs = pdk.cross_section(xs_name)
        settings = {"radius": xs.radius, "width": xs.width, "layer": xs.layer, **settings}
    elif xs_name is not None:
        raise ValueError("a route 'cross_section' needs a PDK with cross-sections")
    return settings


def build_layout(spec: SchematicSpec, pdk: Any = None, *, name: str | None = None, tol: float = 1e-3) -> Cell:
    """Lay out a :class:`SchematicSpec` (module docs for the order of operations).

    Parameters
    ----------
    spec
        The schematic.
    pdk
        Optional :class:`~photonix.pdk.Pdk`: components are instantiated with
        ``pdk.get_layout`` and routes default to its cross-section; names not
        in the PDK fall back to :mod:`photonix.layout.components`.
    name
        Cell name (default ``spec.name`` or ``"schematic"``).
    tol
        Maximum mismatch (µm) of a connection after placement.

    Returns
    -------
    Cell
        Instances as references named like the schematic; route cells are
        references named ``"<route group>_<k>"``.
    """
    from . import routing, routing_astar, routing_auto, routing_bundle

    nl = spec.netlist
    top = Cell(name or spec.name or "schematic")
    refs: dict[str, Reference] = {}
    for inst, component in nl.instances.items():
        refs[inst] = top.add_ref(_instantiate(component, nl.settings.get(inst, {}), pdk), name=inst)

    placed: set[str] = set()
    for inst, pl in spec.placements.items():
        ref = refs[inst]
        ref.rotation, ref.mirror = float(pl.rotation) % 360.0, bool(pl.mirror)
        ox, oy = pl.x, pl.y
        if pl.port is not None:
            local = Reference(ref.cell, (0.0, 0.0), ref.rotation, ref.mirror).port(pl.port)
            ox, oy = pl.x - local.center[0], pl.y - local.center[1]
        ref.origin = (float(ox + pl.dx), float(oy + pl.dy))
        placed.add(inst)

    adjacency: dict[str, list[tuple[str, str, str]]] = {k: [] for k in nl.instances}
    for (ia, pa), (ib, pb) in nl.connections.items():
        adjacency[ia].append((pa, ib, pb))
        adjacency[ib].append((pb, ia, pa))
    order = list(nl.instances)
    while len(placed) < len(order):
        frontier = [i for i in order if i in placed]
        if not frontier or all(all(n in placed for _p, n, _q in adjacency[i]) for i in frontier):
            seed = next(i for i in order if i not in placed)   # disconnected: origin, like gdsfactory
            placed.add(seed)
            frontier = [seed]
        queue = list(frontier)
        while queue:
            cur = queue.pop(0)
            for p_cur, nxt, p_nxt in adjacency[cur]:
                if nxt in placed:
                    continue
                refs[nxt].connect(p_nxt, refs[cur].port(p_cur))
                placed.add(nxt)
                queue.append(nxt)
    for (ia, pa), (ib, pb) in nl.connections.items():
        dist, ang = _residual(refs[ia].port(pa), refs[ib].port(pb))
        if dist > tol or ang > 1e-6:
            raise ValueError(f"connection {ia},{pa} <-> {ib},{pb} cannot be satisfied: ports are {dist:.6g} um "
                             f"apart and {ang:.6g} deg from facing each other (conflicting placements or a "
                             "connection cycle that does not close)")

    for group, rs in spec.routes.items():
        links = [(a, b) for g, a, b in spec.route_links() if g == group]
        p1 = [refs[a[0]].port(a[1]) for a, _b in links]
        p2 = [refs[b[0]].port(b[1]) for _a, b in links]
        settings = _route_style(rs.settings, pdk)
        if rs.strategy == "route_bundle":
            cells = routing_bundle.route_bundle(p1, p2, **{"sort": False, **settings})
        elif rs.strategy == "route_manhattan":
            cells = [routing.route_manhattan(x, y, **settings) for x, y in zip(p1, p2, strict=True)]
        elif rs.strategy == "route_auto":
            cells = [routing_auto.route_auto(x, y, **settings) for x, y in zip(p1, p2, strict=True)]
        else:
            cells = [routing_astar.route_astar(x, y, **{"obstacles": top, **settings})
                     for x, y in zip(p1, p2, strict=True)]
        for k, c in enumerate(cells):
            top.add_ref(c, name=f"{group}_{k}")

    for ext, (inst, port) in nl.ports.items():
        top.expose_port(refs[inst], port, ext)
    return top


# --------------------------------------------------------------------------- #
# Layout -> schematic
# --------------------------------------------------------------------------- #
def schematic_from_layout(top: Cell, *, tol: float = 1e-3) -> SchematicSpec:
    """Recover a :class:`SchematicSpec` from the references placed in ``top``.

    Every direct reference becomes an instance with an absolute
    :class:`Placement` (its origin, rotation and mirror). The component name
    and settings come from :func:`~photonix.layout.pcell_info` when the cell
    was made by a :func:`~photonix.layout.pcell`, and are ``cell.name`` with
    no settings otherwise. Coincident, facing optical ports of two instances
    become connections; a reference of a ``model="route"`` cell joining two
    instance ports becomes a one-link route group named after it (the router
    that drew it is not recoverable, so ``route_manhattan`` is recorded);
    ``top``'s own optical ports become the exposed ports.

    Examples
    --------
    >>> from photonix.circuit import Netlist
    >>> nl = Netlist(name="pair")
    >>> _ = nl.add("a", "straight", length=10.0).add("b", "straight", length=5.0)
    >>> _ = nl.expose("in", ("a", "o1")).expose("out", ("b", "o2"))
    >>> spec = SchematicSpec(nl, {"a": Placement(), "b": Placement(x=40.0, y=20.0)},
    ...                      {"r": RouteSpec({"a,o2": "b,o1"}, {"radius": 5.0}, "route_manhattan")})
    >>> back = schematic_from_layout(build_layout(spec))
    >>> back.placements["b"].x, back.routes["r_0"].links
    (40.0, {'a,o2': 'b,o1'})
    """
    from photonix.circuit import Netlist

    from .pcell import pcell_info

    instances: dict[str, Any] = {}
    placements: dict[str, Placement] = {}
    route_refs: list[tuple[str, Reference]] = []
    terms: list[tuple[str, str, Port]] = []
    for i, ref in enumerate(top.references):
        tag = ref.name or f"ref{i}"
        if ref.cell.model == "route" and not ref.cell.references:
            route_refs.append((tag, ref))
            continue
        info = pcell_info(ref.cell)
        if info is not None:
            instances[tag] = {"model": info["component"], "settings": dict(info["settings"])}
        else:
            instances[tag] = {"model": ref.cell.name}
        placements[tag] = Placement(float(ref.origin[0]), float(ref.origin[1]), float(ref.rotation) % 360.0,
                                    bool(ref.mirror))
        terms.extend((tag, pn, p) for pn, p in ref.ports.items() if p.port_type == OPTICAL)

    def find(port: Port, exclude: str | None = None) -> tuple[str, str] | None:
        hits = [(t, pn) for t, pn, p in terms if t != exclude and _residual(p, port)[0] <= tol]
        if len(hits) > 1:
            raise ValueError(f"more than two ports coincide at {port.center}: {hits}")
        return hits[0] if hits else None

    nl = Netlist(instances=instances, name=top.name)
    by_terminal = {(t, pn): p for t, pn, p in terms}
    done: set[tuple[str, str]] = set()
    for tag, pn, p in terms:
        if (tag, pn) in done:
            continue
        other = find(p, exclude=tag)
        if other is not None and other not in done:
            if _residual(p, by_terminal[other])[1] > 1e-6:
                raise ValueError(f"ports {tag},{pn} and {other[0]},{other[1]} coincide but do not face each other")
            nl.connect((tag, pn), other)
            done.update({(tag, pn), other})
    routes: dict[str, RouteSpec] = {}
    for tag, ref in route_refs:
        ends = [find(ref.port(pn).flipped()) for pn in ("o1", "o2")]
        if ends[0] is None or ends[1] is None:
            raise ValueError(f"route {tag!r} does not join two instance ports")
        routes[tag] = RouteSpec({f"{ends[0][0]},{ends[0][1]}": f"{ends[1][0]},{ends[1][1]}"}, {}, "route_manhattan")
        done.update({ends[0], ends[1]})
    for ext, p in top.ports.items():
        if p.port_type != OPTICAL:
            continue
        hit = find(p)
        if hit is None:
            raise ValueError(f"top port {ext!r} is not on an instance port")
        nl.expose(ext, hit)
    return SchematicSpec(nl, placements, routes)
