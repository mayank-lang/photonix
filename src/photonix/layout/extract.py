"""Extract a circuit netlist from a placed layout.

Two instance ports are considered *connected* when their centers coincide within
a tolerance. The result is always a :class:`photonix.circuit.Netlist`, which
:func:`photonix.circuit.circuit_from_netlist` simulates directly -- closing the
layout -> netlist -> simulation loop described in ``docs/ARCHITECTURE.md``.
"""
from __future__ import annotations

import numpy as np

from .cell import Cell

__all__ = ["extract_netlist"]


def extract_netlist(top: Cell, *, tol: float = 1e-3):
    """Build a netlist from the references placed in ``top``.

    Each reference becomes an instance (named ``ref.name`` or ``refN``) whose
    model name is the referenced cell's ``name``. Coincident reference ports are
    connected; unconnected ports become exposed circuit ports.

    Returns
    -------
    photonix.circuit.Netlist
        Ready to hand to :func:`photonix.circuit.circuit_from_netlist`.

    Examples
    --------
    >>> from photonix.layout import Cell, components, extract_netlist
    >>> top = Cell("top")
    >>> _ = top.add_ref(components.straight(10.0), origin=(0, 0), name="a")
    >>> _ = top.add_ref(components.straight(10.0), origin=(10, 0), name="b")
    >>> nl = extract_netlist(top)
    >>> "a" in nl.instances
    True
    >>> nl.connections                      # the two straights abut
    {('a', 'o2'): ('b', 'o1')}
    """
    if not np.isfinite(tol) or tol < 0:
        raise ValueError(f"tol must be non-negative and finite, got {tol!r}.")

    # Collect each reference's transformed ports.
    inst_ports: dict[str, dict[str, np.ndarray]] = {}
    instances: dict[str, str] = {}
    for i, ref in enumerate(top.references):
        tag = ref.name or f"ref{i}"
        if tag in instances:
            raise ValueError(f"Duplicate reference instance name {tag!r}.")
        instances[tag] = ref.cell.name
        moved = {}
        for pn, prt in ref.cell.ports.items():
            mp = prt.moved(ref.origin[0], ref.origin[1], ref.rotation, ref.mirror)
            moved[pn] = np.asarray(mp.center)
        inst_ports[tag] = moved

    # Pairwise coincidence -> connections.
    terms = [(inst, pn, ctr) for inst, ps in inst_ports.items() for pn, ctr in ps.items()]
    candidates: list[tuple[int, int]] = []
    incidence: dict[int, int] = {}
    for a in range(len(terms)):
        ia, _pa, ca = terms[a]
        for b in range(a + 1, len(terms)):
            ib, _pb, cb = terms[b]
            if ia != ib and np.linalg.norm(ca - cb) <= tol:
                candidates.append((a, b))
                incidence[a] = incidence.get(a, 0) + 1
                incidence[b] = incidence.get(b, 0) + 1
    ambiguous = [terms[i][:2] for i, count in incidence.items() if count > 1]
    if ambiguous:
        raise ValueError(
            "More than two instance ports coincide within tol; multi-way optical "
            f"connections are ambiguous: {ambiguous!r}."
        )

    connections: dict[tuple[str, str], tuple[str, str]] = {}
    used: set[int] = set()
    for a, b in candidates:
        ia, pa, ca = terms[a]
        ib, pb, _cb = terms[b]
        connections[(ia, pa)] = (ib, pb)
        used.add(a)
        used.add(b)

    # Unconnected terminals -> exposed ports.
    ports: dict[str, tuple[str, str]] = {}
    for idx, (inst, pn, _c) in enumerate(terms):
        if idx not in used:
            ports[f"{inst}_{pn}"] = (inst, pn)

    # Imported lazily (not at module scope) only to keep the layout -> circuit
    # dependency one-directional; photonix.circuit is pure-Python core and is
    # always importable, so there is no fallback path to take here.
    from photonix.circuit import Netlist

    return Netlist(instances=instances, connections=connections, ports=ports, name=top.name)
