"""Extract a circuit netlist from a placed layout.

Two instance ports are considered *connected* when their centers coincide within
a tolerance. The result is a :class:`photonix.circuit.Netlist` when the circuit
package is importable, otherwise an equivalent plain-dict structure.
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
    photonix.circuit.Netlist | dict
        A real ``Netlist`` if :mod:`photonix.circuit` is available, else a dict
        with keys ``instances``/``connections``/``ports``.

    Examples
    --------
    >>> from photonix.layout import Cell, components, extract_netlist
    >>> top = Cell("top")
    >>> _ = top.add_ref(components.straight(10.0), origin=(0, 0), name="a")
    >>> _ = top.add_ref(components.straight(10.0), origin=(10, 0), name="b")
    >>> nl = extract_netlist(top)
    >>> ("a" in getattr(nl, "instances", nl["instances"]))
    True
    """
    # Collect each reference's transformed ports.
    inst_ports: dict[str, dict[str, np.ndarray]] = {}
    instances: dict[str, str] = {}
    for i, ref in enumerate(top.references):
        tag = ref.name or f"ref{i}"
        instances[tag] = ref.cell.name
        moved = {}
        for pn, prt in ref.cell.ports.items():
            mp = prt.moved(ref.origin[0], ref.origin[1], ref.rotation, ref.mirror)
            moved[pn] = np.asarray(mp.center)
        inst_ports[tag] = moved

    # Pairwise coincidence -> connections.
    terms = [(inst, pn, ctr) for inst, ps in inst_ports.items() for pn, ctr in ps.items()]
    connections: dict[tuple[str, str], tuple[str, str]] = {}
    used: set[int] = set()
    for a in range(len(terms)):
        if a in used:
            continue
        ia, pa, ca = terms[a]
        for b in range(a + 1, len(terms)):
            if b in used:
                continue
            ib, pb, cb = terms[b]
            if ia == ib:
                continue
            if np.linalg.norm(ca - cb) <= tol:
                connections[(ia, pa)] = (ib, pb)
                used.add(a)
                used.add(b)
                break

    # Unconnected terminals -> exposed ports.
    ports: dict[str, tuple[str, str]] = {}
    for idx, (inst, pn, _c) in enumerate(terms):
        if idx not in used:
            ports[f"{inst}_{pn}"] = (inst, pn)

    try:
        from photonix.circuit import Netlist

        return Netlist(instances=instances, connections=connections, ports=ports, name=top.name)
    except Exception:  # noqa: BLE001
        return {"instances": instances, "connections": connections, "ports": ports}
