"""Extract a circuit netlist from a placed layout.

Two optical instance ports are *connected* when their centres coincide within
a tolerance. The result is a :class:`photonix.circuit.Netlist`, which
:func:`photonix.circuit.circuit_from_netlist` simulates directly -- closing the
layout -> netlist -> simulation loop described in ``docs/ARCHITECTURE.md``.

Rules
-----
* A referenced cell is a *leaf instance* if it declares a ``model`` or has no
  references of its own; its model name is ``cell.model or cell.name`` and its
  ``model_settings`` become the instance settings, so the circuit simulates the
  drawn geometry.
* A composite cell (references, no ``model``) is flattened recursively; its
  instances are named ``"<ref>.<subref>"``.
* Cells without optical ports (pads, logos, closed rings) are omitted.
* Unconnected optical ports become external circuit ports. If one coincides
  with a port of ``top`` itself, that port's name is used; otherwise the name
  is ``"<instance>_<port>"``.
* Coincident ports that do not face each other, or whose widths differ, are
  reported with a warning: they usually mean a mis-placed device.
"""
from __future__ import annotations

import warnings

import numpy as np

from .cell import OPTICAL, Cell, Port, Reference

__all__ = ["extract_netlist"]


def _is_leaf(cell: Cell) -> bool:
    return cell.model is not None or not cell.references


def _compose(port: Port, chain: list[Reference]) -> Port:
    for ref in reversed(chain):
        port = port.moved(ref.origin[0], ref.origin[1], ref.rotation, ref.mirror)
    return port


def _collect(cell: Cell, chain: list[Reference], prefix: str, out: list, seen: set[str]) -> None:
    for i, ref in enumerate(cell.references):
        tag = ref.name or f"ref{i}"
        name = f"{prefix}{tag}"
        sub = ref.cell
        if not _is_leaf(sub):
            _collect(sub, [*chain, ref], f"{name}.", out, seen)
            continue
        optical = {pn: p for pn, p in sub.ports.items() if p.port_type == OPTICAL}
        if not optical:
            continue
        if name in seen:
            raise ValueError(f"Duplicate reference instance name {name!r}.")
        seen.add(name)
        ports = {pn: _compose(p, [*chain, ref]) for pn, p in optical.items()}
        out.append((name, sub, ports))


def extract_netlist(top: Cell, *, tol: float = 1e-3, warn: bool = True):
    """Build a netlist from the references placed (recursively) in ``top``.

    Parameters
    ----------
    top
        The cell to extract.
    tol
        Coincidence tolerance for port centres, in µm.
    warn
        Emit a :class:`UserWarning` for coincident ports that are not
        antiparallel or have different widths.

    Returns
    -------
    photonix.circuit.Netlist
        Ready to hand to :func:`photonix.circuit.circuit_from_netlist`.

    Examples
    --------
    >>> from photonix.layout import Cell, components, extract_netlist
    >>> top = Cell("top")
    >>> _ = top.add_ref(components.straight(10.0), origin=(0, 0), name="a")
    >>> _ = top.add_ref(components.straight(25.0), origin=(10, 0), name="b")
    >>> nl = extract_netlist(top)
    >>> nl.connections                      # the two straights abut
    {('a', 'o2'): ('b', 'o1')}
    >>> nl.settings["b"]                    # the drawn length reaches the model
    {'length': 25.0}
    """
    if not np.isfinite(tol) or tol < 0:
        raise ValueError(f"tol must be non-negative and finite, got {tol!r}.")

    leaves: list[tuple[str, Cell, dict[str, Port]]] = []
    _collect(top, [], "", leaves, set())

    instances: dict[str, str] = {}
    settings: dict[str, dict] = {}
    terms: list[tuple[str, str, Port]] = []
    for name, cell, leaf_ports in leaves:
        instances[name] = cell.model or cell.name
        if cell.model_settings:
            settings[name] = dict(cell.model_settings)
        terms.extend((name, pn, p) for pn, p in leaf_ports.items())

    centres = np.array([p.center for _i, _p, p in terms], dtype=float).reshape(-1, 2)
    candidates: list[tuple[int, int]] = []
    incidence: dict[int, int] = {}
    for a in range(len(terms)):
        if not len(centres):
            break
        close = np.nonzero(np.linalg.norm(centres[a + 1:] - centres[a], axis=1) <= tol)[0] + a + 1
        for b in close:
            if terms[a][0] == terms[b][0]:
                continue
            candidates.append((a, int(b)))
            incidence[a] = incidence.get(a, 0) + 1
            incidence[int(b)] = incidence.get(int(b), 0) + 1
    ambiguous = [terms[i][:2] for i, count in incidence.items() if count > 1]
    if ambiguous:
        raise ValueError(
            "More than two instance ports coincide within tol; multi-way optical "
            f"connections are ambiguous: {ambiguous!r}."
        )

    connections: dict[tuple[str, str], tuple[str, str]] = {}
    used: set[int] = set()
    for a, b in candidates:
        ia, pa, port_a = terms[a]
        ib, pb, port_b = terms[b]
        if warn:
            mismatch = abs(((port_a.orientation - port_b.orientation) % 360.0) - 180.0)
            if mismatch > 1e-3:
                warnings.warn(
                    f"Ports {ia}.{pa} and {ib}.{pb} coincide but are not antiparallel "
                    f"({port_a.orientation:g} vs {port_b.orientation:g} deg).",
                    stacklevel=2,
                )
            if abs(port_a.width - port_b.width) > 1e-6:
                warnings.warn(
                    f"Ports {ia}.{pa} and {ib}.{pb} coincide but have different widths "
                    f"({port_a.width:g} vs {port_b.width:g} um).",
                    stacklevel=2,
                )
        connections[(ia, pa)] = (ib, pb)
        used.update((a, b))

    named = {pn: np.asarray(p.center, dtype=float) for pn, p in top.ports.items() if p.port_type == OPTICAL}
    ports: dict[str, tuple[str, str]] = {}
    for idx, (inst, pn, prt) in enumerate(terms):
        if idx in used:
            continue
        external = f"{inst}_{pn}"
        for top_name, ctr in named.items():
            if top_name not in ports and np.linalg.norm(ctr - np.asarray(prt.center)) <= tol:
                external = top_name
                break
        if external in ports:
            raise ValueError(f"External port name {external!r} is not unique.")
        ports[external] = (inst, pn)

    # Imported lazily (not at module scope) only to keep the layout -> circuit
    # dependency one-directional; photonix.circuit is pure-Python core and is
    # always importable, so there is no fallback path to take here.
    from photonix.circuit import Netlist

    return Netlist(instances=instances, connections=connections, ports=ports, settings=settings,
                   name=top.name)
