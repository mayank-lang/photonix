"""LVS-lite: layout connectivity checks plus layout-versus-schematic netlist comparison.

:func:`run_lvs_lite` runs, on one top cell,

1. :func:`~photonix.verification.check_ports_on_geometry` on every distinct
   leaf cell (issues are reported once per placed instance, in the top frame);
2. :func:`~photonix.verification.check_optical_connectivity` (opens,
   near-opens, misjoined ports, shorts);
3. :func:`~photonix.verification.check_parameters` (settings versus drawn
   lengths, gaps and radii);
4. with a ``schematic``, :func:`compare_netlists` between
   :func:`photonix.layout.extract_netlist` of the layout and the schematic.

Netlist comparison
------------------
Both netlists become port-level graphs: an instance node per instance
(labelled by its model), one node per used instance port, ``conn`` edges for
connections and ``ext:<name>`` nodes for external ports. Instances whose
model is *transparent* (routes, by default) are first collapsed into direct
connections, so a routed layout compares equal to a schematic that joins the
devices directly.

Instances are then matched deterministically, most specific evidence first:
identical instance names with the same model, the instances behind
identically named external ports, propagation along connections from matched
instances (port names and models must agree), and finally Weisfeiler-Lehman
colour refinement (Shervashidze et al., JMLR 12, 2539, 2011) on the union of
both graphs, assigning the remaining instances greedily within equal colour
classes (each assignment must agree with every already matched neighbour). A deterministic matcher is
used instead of a plain isomorphism test because it localises a difference
(a swapped connection is reported at the two connections, not as "not
isomorphic"). Matched instances are compared for model and settings
(numbers with ``rtol``/``atol``).
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from photonix.circuit.netlist import Netlist
from photonix.layout.cell import Cell

from ._geom import MARKER_LAYERS, collect_leaves, layer_tuple, to_top
from .connectivity import ISSUE_KINDS, LvsIssue, check_optical_connectivity, check_ports_on_geometry
from .extract_params import check_parameters

__all__ = ["LvsReport", "compare_netlists", "run_lvs_lite"]


@dataclass
class LvsReport:
    """Result of :func:`run_lvs_lite` or :func:`compare_netlists`.

    Attributes
    ----------
    issues
        Every finding.
    mapping
        Layout instance -> schematic instance, for matched instances.
    """

    issues: list[LvsIssue] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        """True when no issue was found."""
        return not self.issues

    def summary(self) -> dict[str, int]:
        """Issue counts by kind."""
        out: dict[str, int] = {}
        for i in self.issues:
            out[i.kind] = out.get(i.kind, 0) + 1
        return dict(sorted(out.items()))

    def of_kind(self, kind: str) -> list[LvsIssue]:
        """The issues of one kind."""
        return [i for i in self.issues if i.kind == kind]

    def __str__(self) -> str:
        if self.clean:
            return "LVS-lite: clean"
        lines = [f"LVS-lite: {len(self.issues)} issue(s)"]
        for kind, n in self.summary().items():
            lines.append(f"  {kind:<22s} {n}")
        for i in self.issues[:50]:
            lines.append(f"  - {i.kind}: {', '.join(i.objects)} at ({i.location[0]:.4f}, {i.location[1]:.4f}): "
                         f"{i.detail}")
        if len(self.issues) > 50:
            lines.append(f"  ... {len(self.issues) - 50} more")
        return "\n".join(lines)

    def to_lyrdb(self, path: str | os.PathLike[str], *, top: str = "TOP", marker_size: float = 0.2) -> str:
        """Write the issues as a KLayout report database (one category per kind).

        Issues without a marker polygon get a square of ``marker_size`` µm
        around their location.
        """
        from .rdb import _write_items

        items = []
        for i in self.issues:
            poly = i.marker if i.marker is not None else _square(i.location, 0.5 * marker_size)
            items.append((i.kind, poly))
        return _write_items(items, path, top=top, descriptions=dict(ISSUE_KINDS),
                                 generator="photonix.verification.run_lvs_lite")


def _square(c: Sequence[float], h: float) -> np.ndarray:
    x, y = float(c[0]), float(c[1])
    return np.array([(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)])


# --------------------------------------------------------------------------- #
# netlist normalisation
# --------------------------------------------------------------------------- #
IP = tuple[str, str]


@dataclass
class _Net:
    models: dict[str, str]
    settings: dict[str, dict[str, Any]]
    adj: dict[IP, IP]             # symmetric instance-port adjacency
    ext: dict[str, IP]            # external port -> instance port
    index: dict[str, set[str]] = field(default_factory=dict)

    def reindex(self) -> _Net:
        self.index = {}
        for i, p in [*self.adj, *self.ext.values()]:
            self.index.setdefault(i, set()).add(p)
        return self

    def ports_of(self, inst: str) -> set[str]:
        return self.index.get(inst, set())


def _normalise(nl: Netlist, transparent: set[str]) -> _Net:
    models = dict(nl.instances)
    settings = {k: dict(v) for k, v in nl.settings.items()}
    adj: dict[IP, IP] = {}
    for a, b in nl.connections.items():
        a, b = (str(a[0]), str(a[1])), (str(b[0]), str(b[1]))
        adj[a] = b
        adj[b] = a
    ext = {str(k): (str(v[0]), str(v[1])) for k, v in nl.ports.items()}
    # Collapse transparent two-port instances (routes) into direct connections.
    for inst in sorted(i for i, m in models.items() if m in transparent):
        ends = sorted(ip for ip in set(adj) | set(ext.values()) if ip[0] == inst)
        if len(ends) > 2:
            continue
        partners: list[IP | None] = [adj.pop(ip, None) for ip in ends]
        for p in partners:
            if p is not None:
                adj.pop(p, None)
        ext_names = [n for n, ip in ext.items() if ip[0] == inst]
        if len(ends) == 2:
            pa, pb = partners
            if pa is not None and pb is not None:
                adj[pa] = pb
                adj[pb] = pa
            for n in ext_names:
                ip = ext[n]
                other = partners[1] if ip == ends[0] else partners[0]
                if other is not None:
                    ext[n] = other
                else:
                    del ext[n]
        else:
            for n in ext_names:
                del ext[n]
        models.pop(inst)
        settings.pop(inst, None)
    return _Net(models, settings, adj, ext).reindex()


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
def _wl_colours(net: _Net, rounds: int = 6) -> dict[str, str]:
    """Weisfeiler-Lehman colours of instances on the port-labelled graph."""
    ext_of: dict[IP, str] = {ip: n for n, ip in net.ext.items()}
    col = {i: f"{m}" for i, m in net.models.items()}
    for step in range(rounds):
        new = {}
        for i in net.models:
            sig = []
            for p in sorted(net.ports_of(i)):
                nb = net.adj.get((i, p))
                if nb is not None and nb[0] in col:
                    sig.append(f"{p}>{nb[1]}@{col[nb[0]]}")
                elif (i, p) in ext_of:
                    sig.append(f"{p}>ext")
                else:
                    sig.append(f"{p}>-")
            new[i] = col[i] + "|" + ";".join(sig)
        # Compress
        table = {c: f"c{k}" for k, c in enumerate(sorted(set(new.values())))}
        compressed = {i: table[c] for i, c in new.items()}
        if step > 0 and len(set(compressed.values())) == len(set(col.values())):
            col = compressed
            break
        col = compressed
    return col


def _consistent(lay: _Net, sch: _Net, mapping: dict[str, str], a: str, s: str) -> bool:
    if lay.models[a] != sch.models[s]:
        return False
    for p in lay.ports_of(a) | sch.ports_of(s):
        na, ns = lay.adj.get((a, p)), sch.adj.get((s, p))
        if (na is None) != (ns is None):
            return False
        if na is not None and ns is not None:
            if na[1] != ns[1]:
                return False
            if na[0] in mapping and mapping[na[0]] != ns[0]:
                return False
    return True


def _match(lay: _Net, sch: _Net) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()

    def bind(a: str, s: str) -> None:
        mapping[a] = s
        used.add(s)

    # 1. identical names and models
    for a in sorted(lay.models):
        if a in sch.models and lay.models[a] == sch.models[a]:
            bind(a, a)
    # 2. external-port anchors
    for name in sorted(set(lay.ext) & set(sch.ext)):
        a, s = lay.ext[name][0], sch.ext[name][0]
        if a not in mapping and s not in used and lay.ext[name][1] == sch.ext[name][1]:
            bind(a, s)
    # 3. propagation along connections
    def propagate() -> None:
        changed = True
        while changed:
            changed = False
            for a, s in list(mapping.items()):
                for p in sorted(lay.ports_of(a)):
                    na, ns = lay.adj.get((a, p)), sch.adj.get((s, p))
                    if na is None or ns is None or na[0] in mapping or ns[0] in used:
                        continue
                    if na[1] == ns[1] and lay.models[na[0]] == sch.models[ns[0]]:
                        bind(na[0], ns[0])
                        changed = True

    propagate()
    # 4. colour refinement on the disjoint union, greedy consistent assignment, re-propagation
    rest_a = sorted(a for a in lay.models if a not in mapping)
    rest_s = sorted(s for s in sch.models if s not in used)
    if rest_a and rest_s:
        union = _Net({**{f"L|{a}": m for a, m in lay.models.items()}, **{f"S|{s}": m for s, m in sch.models.items()}},
                     {}, {**{(f"L|{x[0]}", x[1]): (f"L|{y[0]}", y[1]) for x, y in lay.adj.items()},
                          **{(f"S|{x[0]}", x[1]): (f"S|{y[0]}", y[1]) for x, y in sch.adj.items()}},
                     {**{f"L|{n}": (f"L|{ip[0]}", ip[1]) for n, ip in lay.ext.items()},
                      **{f"S|{n}": (f"S|{ip[0]}", ip[1]) for n, ip in sch.ext.items()}}).reindex()
        colour = _wl_colours(union)
        for a in rest_a:
            if a in mapping:
                continue
            for s in rest_s:
                if s in used or colour[f"L|{a}"] != colour[f"S|{s}"] or not _consistent(lay, sch, mapping, a, s):
                    continue
                bind(a, s)
                propagate()
                break
    return mapping


def _close(a: Any, b: Any, rtol: float, atol: float) -> bool:
    num = (int, float, np.integer, np.floating)
    if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool) and not isinstance(b, bool):
        return bool(np.isclose(float(a), float(b), rtol=rtol, atol=atol))
    try:
        return bool(a == b)
    except Exception:  # pragma: no cover - exotic objects
        return False


def compare_netlists(
    layout: Netlist,
    schematic: Netlist,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-3,
    ignore_settings: Iterable[str] = (),
    transparent_models: Iterable[str] = ("route",),
    require_settings: bool = False,
) -> LvsReport:
    """Compare a layout-extracted netlist with a schematic netlist.

    Parameters
    ----------
    layout, schematic
        The two :class:`photonix.circuit.Netlist` objects.
    rtol, atol
        Tolerances for numeric settings (``atol`` in the setting's units,
        µm for geometry).
    ignore_settings
        Setting names never compared.
    transparent_models
        Models collapsed into direct connections before comparison.
    require_settings
        Also report schematic settings that the layout does not record.

    Returns
    -------
    LvsReport
        With ``missing_instance``, ``extra_instance``, ``model_mismatch``,
        ``setting_mismatch``, ``connection_mismatch`` and
        ``external_port_mismatch`` issues (location ``(0, 0)``: netlists have
        no coordinates) and the instance ``mapping``.

    Examples
    --------
    >>> from photonix.circuit import Netlist
    >>> sch = Netlist(instances={"a": "straight", "b": "straight"},
    ...               connections={("a", "o2"): ("b", "o1")},
    ...               ports={"in": ("a", "o1"), "out": ("b", "o2")},
    ...               settings={"a": {"length": 10.0}, "b": {"length": 20.0}})
    >>> lay = Netlist(instances={"w1": "straight", "w2": "straight"},
    ...               connections={("w1", "o2"): ("w2", "o1")},
    ...               ports={"in": ("w1", "o1"), "out": ("w2", "o2")},
    ...               settings={"w1": {"length": 10.0}, "w2": {"length": 25.0}})
    >>> rep = compare_netlists(lay, sch)
    >>> rep.mapping, [(i.kind, i.objects) for i in rep.issues]
    ({'w1': 'a', 'w2': 'b'}, [('setting_mismatch', ('w2', 'b'))])
    """
    transparent = set(transparent_models)
    lay, sch = _normalise(layout, transparent), _normalise(schematic, transparent)
    mapping = _match(lay, sch)
    inv = {s: a for a, s in mapping.items()}
    issues: list[LvsIssue] = []
    origin = (0.0, 0.0)

    for a in sorted(lay.models):
        if a not in mapping:
            issues.append(LvsIssue("extra_instance", origin, (a,), f"layout instance {a!r} ({lay.models[a]}) "
                                   "has no schematic counterpart"))
    for s in sorted(sch.models):
        if s not in inv:
            issues.append(LvsIssue("missing_instance", origin, (s,), f"schematic instance {s!r} ({sch.models[s]}) "
                                   "is not in the layout"))
    ignore = set(ignore_settings)
    for a, s in sorted(mapping.items()):
        if lay.models[a] != sch.models[s]:
            issues.append(LvsIssue("model_mismatch", origin, (a, s),
                                   f"model {lay.models[a]!r} in layout, {sch.models[s]!r} in schematic"))
        sl, ss = lay.settings.get(a, {}), sch.settings.get(s, {})
        for key in sorted(set(ss) - ignore):
            if key not in sl:
                if require_settings:
                    issues.append(LvsIssue("setting_mismatch", origin, (a, s),
                                           f"setting {key!r} = {ss[key]!r} is not recorded by the layout"))
                continue
            if not _close(sl[key], ss[key], rtol, atol):
                val = float(sl[key]) if isinstance(sl[key], (int, float)) and not isinstance(sl[key], bool) else None
                issues.append(LvsIssue("setting_mismatch", origin, (a, s),
                                       f"setting {key!r}: layout {sl[key]!r}, schematic {ss[key]!r}", val))

    def mapped(ip: IP) -> IP | None:
        return (mapping[ip[0]], ip[1]) if ip[0] in mapping else None

    lay_edges = {frozenset((x, y)) for x, y in lay.adj.items()}
    sch_edges = {frozenset((x, y)) for x, y in sch.adj.items()}
    lay_mapped = set()
    for e in sorted(lay_edges, key=lambda e: sorted(e)):
        x, y = sorted(e)
        mx, my = mapped(x), mapped(y)
        if mx is None or my is None:
            continue
        lay_mapped.add(frozenset((mx, my)))
        if frozenset((mx, my)) not in sch_edges:
            issues.append(LvsIssue("connection_mismatch", origin, (f"{x[0]}.{x[1]}", f"{y[0]}.{y[1]}"),
                                   f"drawn connection {x[0]}.{x[1]} - {y[0]}.{y[1]} is not in the schematic"))
    for e in sorted(sch_edges, key=lambda e: sorted(e)):
        x, y = sorted(e)
        if x[0] not in inv or y[0] not in inv:
            continue
        if e not in lay_mapped:
            issues.append(LvsIssue("connection_mismatch", origin, (f"{x[0]}.{x[1]}", f"{y[0]}.{y[1]}"),
                                   f"schematic connection {x[0]}.{x[1]} - {y[0]}.{y[1]} is not drawn"))
    for name in sorted(set(lay.ext) | set(sch.ext)):
        la, sa = lay.ext.get(name), sch.ext.get(name)
        if la is None:
            issues.append(LvsIssue("external_port_mismatch", origin, (name,),
                                   f"schematic port {name!r} -> {sa[0]}.{sa[1]} is not a layout port"
                                   if sa else f"port {name!r}"))
        elif sa is None:
            issues.append(LvsIssue("external_port_mismatch", origin, (name,),
                                   f"layout port {name!r} -> {la[0]}.{la[1]} is not a schematic port"))
        elif mapped(la) != sa and la[0] in mapping:
            issues.append(LvsIssue("external_port_mismatch", origin, (name,),
                                   f"port {name!r} is {la[0]}.{la[1]} in the layout, {sa[0]}.{sa[1]} in the schematic"))
    return LvsReport(issues, dict(sorted(mapping.items())))


# --------------------------------------------------------------------------- #
# the full run
# --------------------------------------------------------------------------- #
def run_lvs_lite(
    top: Cell,
    schematic: Netlist | None = None,
    *,
    tol: float = 1e-3,
    near: float = 1.0,
    probe: float = 1e-3,
    width_tol: float = 2e-3,
    layers: Sequence[tuple[int, int]] | None = None,
    min_short_area: float = 1e-4,
    marker_layers: Iterable[tuple[int, int]] = MARKER_LAYERS,
    parameters: bool = True,
    rtol: float = 2e-3,
    atol: float = 2e-3,
    compare: Mapping[str, Any] | None = None,
) -> LvsReport:
    """Run every LVS-lite check on ``top`` (and compare with ``schematic``).

    Parameters
    ----------
    top
        The layout.
    schematic
        Optional reference netlist; compared with
        :func:`photonix.layout.extract_netlist` ``(top, tol=tol)`` by
        :func:`compare_netlists`.
    tol, near, probe, width_tol, layers, min_short_area, marker_layers
        Passed to :func:`~photonix.verification.check_ports_on_geometry` and
        :func:`~photonix.verification.check_optical_connectivity`.
    parameters
        Run :func:`~photonix.verification.check_parameters` with
        ``rtol``/``atol``.
    rtol, atol
        Tolerances of the parameter check.
    compare
        Keyword arguments for :func:`compare_netlists`.

    Examples
    --------
    >>> from photonix.layout import components as lc
    >>> rep = run_lvs_lite(lc.mzi(delta_length=20.0))
    >>> rep.clean
    True
    """
    markers = tuple(layer_tuple(m) for m in marker_layers)
    issues: list[LvsIssue] = []
    cache: dict[int, list[LvsIssue]] = {}
    for lf in collect_leaves(top):
        if not lf.cell.ports:
            continue
        key = id(lf.cell)
        if key not in cache:
            cache[key] = check_ports_on_geometry(lf.cell, probe=probe, width_tol=width_tol, marker_layers=markers)
        for i in cache[key]:
            loc = to_top(np.asarray([i.location]), lf.chain)[0]
            issues.append(LvsIssue(i.kind, (float(loc[0]), float(loc[1])),
                                   tuple(f"{lf.name}.{o}" for o in i.objects), f"{lf.name}: {i.detail}", i.value))
    issues.extend(check_optical_connectivity(top, tol=tol, near=near, layers=layers, min_short_area=min_short_area,
                                             probe=probe, marker_layers=markers))
    if parameters:
        issues.extend(check_parameters(top, rtol=rtol, atol=atol, marker_layers=markers))
    mapping: dict[str, str] = {}
    if schematic is not None:
        from photonix.layout.extract import extract_netlist

        rep = compare_netlists(extract_netlist(top, tol=tol, warn=False), schematic, **dict(compare or {}))
        issues.extend(rep.issues)
        mapping = rep.mapping
    return LvsReport(issues, mapping)
