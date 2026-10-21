"""Extract a hierarchical netlist (subcircuits per composite cell) from a layout.

:func:`photonix.layout.extract_netlist` flattens every composite cell into
dotted instance names. :func:`extract_hierarchical_netlist` keeps the
hierarchy instead: each distinct composite :class:`~photonix.layout.Cell`
(references and no ``model``) becomes one
:class:`~photonix.circuit.Subcircuit`, and every reference to it becomes an
instance of that subcircuit, so a cell placed 120 times is described -- and,
by :func:`photonix.circuit.compile_hierarchical`, solved -- once per distinct
setting.

Rules
-----
* Leaf rules, connection tolerance and warnings are those of
  :func:`~photonix.layout.extract_netlist` (each composite is extracted with
  it in its own coordinates; geometric coincidence is invariant under the
  rigid reference transforms, so every placement of a cell has the same
  internal connections).
* A subcircuit's external ports are the optical terminals of its leaves that
  are not connected inside the cell. A terminal on one of the cell's declared
  optical ports takes that port's name; any other open terminal is named
  ``"<instance>_<port>"`` relative to the cell (it still has to be a port,
  because the parent may connect to it or expose it).
* Subcircuits are named after their cells. Distinct cell objects with the same
  name share one subcircuit when their subcircuits are equal and get a
  numbered suffix otherwise; a name that clashes with a leaf model name gets
  the suffix ``"_cell"``.

The result satisfies ``extract_hierarchical_netlist(top).flatten() ==
extract_netlist(top)`` (instances, settings, connections and port names).
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from .cell import OPTICAL, Cell
from .extract import _collect, _is_leaf, extract_netlist

if TYPE_CHECKING:
    from photonix.circuit import HierarchicalNetlist, Netlist, Subcircuit

__all__ = ["extract_hierarchical_netlist"]


def _has_optical_leaves(cell: Cell) -> bool:
    out: list = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _collect(cell, [], "", out, set())
    return bool(out)


def extract_hierarchical_netlist(top: Cell, *, tol: float = 1e-3, warn: bool = True) -> HierarchicalNetlist:
    """Build a :class:`~photonix.circuit.HierarchicalNetlist` from ``top``.

    Parameters
    ----------
    top
        The cell to extract.
    tol
        Coincidence tolerance for port centres, in µm.
    warn
        Emit the warnings of :func:`~photonix.layout.extract_netlist`
        (coincident ports that are not antiparallel or differ in width,
        leaves that would simulate with model defaults), once.

    Returns
    -------
    photonix.circuit.HierarchicalNetlist
        One subcircuit per distinct composite cell.

    Examples
    --------
    >>> from photonix.layout import Cell, components, extract_netlist
    >>> pair = Cell("pair")
    >>> a = pair.add_ref(components.straight(10.0), name="a")
    >>> b = pair.add_ref(components.straight(5.0), origin=(10, 0), name="b")
    >>> _ = pair.expose_port(a, "o1").expose_port(b, "o2")
    >>> top = Cell("top")
    >>> p0 = top.add_ref(pair, name="p0")
    >>> _ = top.add_ref(pair, origin=(15, 0), name="p1")
    >>> hnl = extract_hierarchical_netlist(top)
    >>> sorted(hnl.subcircuits), hnl.top.instances
    (['pair'], {'p0': 'pair', 'p1': 'pair'})
    >>> hnl.top.connections
    {('p0', 'o2'): ('p1', 'o1')}
    >>> hnl.flatten() == extract_netlist(top)
    True
    """
    from photonix.circuit import HierarchicalNetlist, Netlist, Subcircuit

    flat_top = extract_netlist(top, tol=tol, warn=warn)
    if not top.references:
        return HierarchicalNetlist(flat_top)

    leaf_models = set(flat_top.instances.values())
    # id(cell) -> (Subcircuit, {(relative leaf, port): subcircuit port name})
    built: dict[int, tuple[Subcircuit, dict[tuple[str, str], str]]] = {}
    names: dict[int, str] = {}
    by_name: dict[str, Subcircuit] = {}
    keep_alive: list[Cell] = []

    def level(cell: Cell, flat: Netlist) -> tuple[Netlist, dict[tuple[str, str], tuple[str, str]]]:
        """``cell``'s own netlist (direct children only) and the leaf -> child-terminal map."""
        nl = Netlist(name=cell.name)
        leaf_to_child: dict[tuple[str, str], tuple[str, str]] = {}
        for i, ref in enumerate(cell.references):
            tag = ref.name or f"ref{i}"
            sub = ref.cell
            if _is_leaf(sub):
                if not any(p.port_type == OPTICAL for p in sub.ports.values()):
                    continue
                nl.add(tag, sub.model or sub.name, **dict(sub.model_settings))
                for pn, p in sub.ports.items():
                    if p.port_type == OPTICAL:
                        leaf_to_child[(tag, pn)] = (tag, pn)
                continue
            if not _has_optical_leaves(sub):
                continue
            _child, child_ports = build(sub)
            nl.add(tag, names[id(sub)])
            for (rel, pn), sub_port in child_ports.items():
                leaf_to_child[(f"{tag}.{rel}", pn)] = (tag, sub_port)
        for a, b in flat.connections.items():
            if a not in leaf_to_child and b not in leaf_to_child:
                continue  # both ends inside one composite child: part of that child's subcircuit
            if a not in leaf_to_child or b not in leaf_to_child:  # pragma: no cover - extract_netlist forbids it
                raise RuntimeError(f"connection {a} <-> {b} crosses a cell boundary through an internal terminal")
            ta, tb = leaf_to_child[a], leaf_to_child[b]
            nl.connect(ta, tb)
        for ext, ip in flat.ports.items():
            nl.expose(ext, leaf_to_child[ip])
        return nl, leaf_to_child

    def build(cell: Cell) -> tuple[Subcircuit, dict[tuple[str, str], str]]:
        key = id(cell)
        if key in built:
            return built[key]
        keep_alive.append(cell)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flat = extract_netlist(cell, tol=tol, warn=False)
        nl, _ = level(cell, flat)
        sub = Subcircuit(nl, doc=f"extracted from layout cell {cell.name!r}")
        port_of = {ip: ext for ext, ip in flat.ports.items()}
        base = cell.name if cell.name not in leaf_models else f"{cell.name}_cell"
        name, k = base, 1
        while name in by_name and by_name[name] != sub:
            name, k = f"{base}_{k}", k + 1
        by_name[name] = sub
        names[key] = name
        built[key] = (sub, port_of)
        return built[key]

    top_nl, _ = level(top, flat_top)
    subcircuits = {names[k]: built[k][0] for k in built}
    hnl = HierarchicalNetlist(top_nl, subcircuits, name=top.name)
    hnl.validate()
    return hnl
