"""Exchange netlists with SAX and gdsfactory (``"inst,port"`` dictionaries).

SAX (Laporte et al., github.com/flaport/sax) and gdsfactory describe a
circuit as plain dictionaries::

    {"instances": {"a": {"component": "straight", "settings": {"length": 5.0}}},
     "connections": {"a,o2": "b,o1"},
     "ports": {"in0": "a,o1"}}

A *recursive* SAX netlist maps several such netlists by name, the top one
being ``"top_level"``; it corresponds to a
:class:`~photonix.circuit.HierarchicalNetlist`, and an instance whose
``component`` names another netlist of the mapping is a subcircuit instance.

:func:`to_sax_netlist` and :func:`from_sax_netlist` convert in both directions;
``Netlist -> SAX -> Netlist`` and ``HierarchicalNetlist -> SAX -> HierarchicalNetlist``
round-trip to equality. gdsfactory's ``nets`` list form
(``[{"p1": "a,o2", "p2": "b,o1"}]``) and its ``placements``/``routes`` sections
are accepted on input (the latter two are layout information and are ignored
here; see :class:`~photonix.layout.SchematicSpec`). Subcircuit parameter
declarations and expression settings have no SAX equivalent: they are stored
under the ``"info"`` key of each netlist, which SAX and gdsfactory ignore.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .hierarchy import HierarchicalNetlist, Subcircuit
from .netlist import InstancePort, Netlist, _json_value

__all__ = ["to_sax_netlist", "from_sax_netlist", "parse_instance_port", "format_instance_port", "TOP_LEVEL"]

#: Name of the top netlist in a recursive SAX netlist.
TOP_LEVEL = "top_level"


def format_instance_port(terminal: InstancePort) -> str:
    """``("a", "o2")`` -> ``"a,o2"``.

    Examples
    --------
    >>> format_instance_port(("mzi1", "o2"))
    'mzi1,o2'
    """
    inst, port = terminal
    if "," in inst or "," in port:
        raise ValueError(f"instance and port names must not contain ',' for SAX netlists: {terminal!r}")
    return f"{inst},{port}"


def parse_instance_port(text: str) -> InstancePort:
    """``"a,o2"`` -> ``("a", "o2")`` (whitespace around the comma is ignored).

    Examples
    --------
    >>> parse_instance_port("mzi1, o2")
    ('mzi1', 'o2')
    """
    if not isinstance(text, str) or text.count(",") != 1:
        raise ValueError(f"expected an 'instance,port' string, got {text!r}")
    inst, port = (t.strip() for t in text.split(","))
    if not inst or not port:
        raise ValueError(f"expected an 'instance,port' string, got {text!r}")
    return inst, port


def _one_to_sax(nl: Netlist, info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    nl.validate()
    instances: dict[str, Any] = {}
    for inst, model in nl.instances.items():
        settings = {k: _json_value(v, inst, k) for k, v in nl.settings.get(inst, {}).items()}
        instances[inst] = {"component": model, "settings": settings}
    out: dict[str, Any] = {
        "instances": instances,
        "connections": {format_instance_port(a): format_instance_port(b) for a, b in nl.connections.items()},
        "ports": {ext: format_instance_port(ip) for ext, ip in nl.ports.items()},
    }
    meta = {k: v for k, v in (info or {}).items() if v not in (None, {}, "")}
    if nl.name is not None:
        meta["name"] = nl.name
    if meta:
        out["info"] = meta
    return out


def to_sax_netlist(nl: Netlist | HierarchicalNetlist) -> dict[str, Any]:
    """SAX/gdsfactory netlist dictionary (recursive for a hierarchy).

    Examples
    --------
    >>> nl = Netlist(name="pair")
    >>> _ = nl.add("a", "straight", length=5.0).add("b", "straight")
    >>> _ = nl.connect(("a", "o2"), ("b", "o1"))
    >>> _ = nl.expose("in0", ("a", "o1")).expose("out0", ("b", "o2"))
    >>> d = to_sax_netlist(nl)
    >>> d["connections"], d["ports"]
    ({'a,o2': 'b,o1'}, {'in0': 'a,o1', 'out0': 'b,o2'})
    >>> from_sax_netlist(d) == nl
    True
    """
    if isinstance(nl, Netlist):
        return _one_to_sax(nl)
    if not isinstance(nl, HierarchicalNetlist):
        raise TypeError(f"expected a Netlist or HierarchicalNetlist, got {type(nl).__name__}")
    nl.validate()
    if TOP_LEVEL in nl.subcircuits:
        raise ValueError(f"a subcircuit may not be called {TOP_LEVEL!r} in a SAX netlist")
    top_info: dict[str, Any] = {"parameters": dict(nl.parameters)}
    if nl.name != nl.top.name:
        top_info["hierarchy_name"] = nl.name
    out = {TOP_LEVEL: _one_to_sax(nl.top, top_info)}
    for name, sub in nl.subcircuits.items():
        out[name] = _one_to_sax(sub.netlist, {"parameters": dict(sub.parameters), "doc": sub.doc})
    return out


def _is_single(d: Mapping[str, Any]) -> bool:
    return any(k in d for k in ("instances", "connections", "ports", "nets"))


def _one_from_sax(d: Mapping[str, Any]) -> tuple[Netlist, dict[str, Any]]:
    instances: dict[str, Any] = {}
    for inst, spec in dict(d.get("instances", {})).items():
        if isinstance(spec, str):
            instances[inst] = {"model": spec}
            continue
        if not isinstance(spec, Mapping):
            raise ValueError(f"instance {inst!r} must be a component name or a mapping, got {spec!r}")
        component = spec.get("component", spec.get("model"))
        if not isinstance(component, str):
            raise ValueError(f"instance {inst!r} has no 'component' name")
        settings = dict(spec.get("settings") or {})
        instances[inst] = {"model": component, "settings": settings} if settings else {"model": component}
    connections: dict[InstancePort, InstancePort] = {}
    for a, b in dict(d.get("connections", {})).items():
        connections[parse_instance_port(a)] = parse_instance_port(b)
    for net in d.get("nets", []) or []:
        if not isinstance(net, Mapping) or "p1" not in net or "p2" not in net:
            raise ValueError(f"nets entries must be {{'p1': 'inst,port', 'p2': 'inst,port'}}, got {net!r}")
        connections[parse_instance_port(net["p1"])] = parse_instance_port(net["p2"])
    ports = {ext: parse_instance_port(ip) for ext, ip in dict(d.get("ports", {})).items()}
    info = dict(d.get("info", {}) or {})
    name = info.get("name", d.get("name"))
    nl = Netlist(instances=instances, connections=connections, ports=ports, name=name)
    nl.validate()
    return nl, info


def from_sax_netlist(d: Mapping[str, Any]) -> Netlist | HierarchicalNetlist:
    """Inverse of :func:`to_sax_netlist`.

    A single netlist dictionary gives a :class:`~photonix.circuit.Netlist`; a
    recursive one (``{name: netlist, ...}`` with a ``"top_level"`` entry, or a
    single entry) gives a :class:`~photonix.circuit.HierarchicalNetlist`.

    Examples
    --------
    >>> rec = {
    ...     "top_level": {"instances": {"x": {"component": "blk", "settings": {}}},
    ...                   "connections": {}, "ports": {"o1": "x,o1"}},
    ...     "blk": {"instances": {"w": {"component": "straight", "settings": {"length": 2.0}}},
    ...             "connections": {}, "ports": {"o1": "w,o1"}},
    ... }
    >>> hnl = from_sax_netlist(rec)
    >>> type(hnl).__name__, sorted(hnl.subcircuits), hnl.flatten().instances
    ('HierarchicalNetlist', ['blk'], {'x.w': 'straight'})
    """
    if not isinstance(d, Mapping):
        raise TypeError("a SAX netlist must be a mapping")
    if _is_single(d):
        return _one_from_sax(d)[0]
    if not d:
        raise ValueError("empty SAX netlist")
    if TOP_LEVEL in d:
        top_name = TOP_LEVEL
    elif len(d) == 1:
        top_name = next(iter(d))
    else:
        raise ValueError(f"a recursive SAX netlist needs a {TOP_LEVEL!r} entry; got {sorted(d)}")
    top, top_info = _one_from_sax(d[top_name])
    subs: dict[str, Subcircuit] = {}
    for name, body in d.items():
        if name == top_name:
            continue
        if not isinstance(body, Mapping):
            raise ValueError(f"recursive SAX netlist entry {name!r} must be a netlist mapping")
        nl, info = _one_from_sax(body)
        subs[name] = Subcircuit(nl, dict(info.get("parameters", {}) or {}), str(info.get("doc", "") or ""))
    hnl = HierarchicalNetlist(top, subs, dict(top_info.get("parameters", {}) or {}),
                              top_info.get("hierarchy_name", top.name))
    hnl.validate()
    return hnl
