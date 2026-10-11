"""Scalable circuit engine: compiled topology, vectorized assembly, three solvers.

:func:`photonix.circuit.evaluate_circuit` and
:func:`photonix.circuit.circuit_from_netlist` run on this engine. It solves
the same linear model as before -- with every instance terminal stacked into
one wave vector, ``b = S a`` (block-diagonal ``S`` stored as ``S[out, in]``)
and ``a = Gamma b + u`` (``Gamma`` the symmetric connection permutation, ``u``
the excitation at exposed ports), so ``(I - S Gamma) b = S u`` -- but splits the
work into three stages:

1. :func:`compile_topology` -- port structure only, no values: terminal
   numbering, the ``(row, col)`` index of every ``SDict`` entry, the
   connection permutation and the exposed terminals. Cached per structure.
2. :func:`gather_values` -- one stack of the entry values, shaped
   ``(*batch, nnz)``; no per-entry scatter into an ``n x n`` array.
3. :func:`solve` -- only the ``m`` exposed right-hand sides are needed
   (``u`` is nonzero only at exposed terminals), so it solves
   ``(I - S Gamma) X = S[:, E]`` and reads ``X[E, :]``.

Backends
--------
``"dense"``
    Batched dense LU (``xp.linalg.solve``) of the full system: all JAX
    transformations including forward-mode AD. ``O(n^3)`` per sample and
    ``O(batch * n^2)`` memory.
``"sparse"``
    SuperLU on ``A = I - S Gamma`` (``nnz(A) <= nnz(S) + n``) with a
    hand-written transpose adjoint (:mod:`photonix.circuit.sparse_linalg`).
    ``jit``/``vmap``/``grad`` under JAX via a host callback; no forward-mode
    AD. ``O(batch * (nnz + m^2))`` memory.
``"reduce"``
    Subnetwork growth (:mod:`photonix.circuit.reduction`): pairwise
    ``innerconnect`` on small dense blocks in greedy minimum-size order.
    Pure ``xp`` algebra, so every JAX transformation works and nothing leaves
    the device; cost set by the largest intermediate block.
``"auto"`` (default)
    ``dense`` when ``n <= DENSE_MAX_TERMINALS`` and the batched dense system
    fits in ``DENSE_MAX_BYTES``; otherwise ``sparse``. The sparse callback
    ships only ``O(batch * nnz)`` values to the host and ``O(batch * m^2)``
    back, so it is also preferred on accelerators; choose ``reduce`` or
    ``dense`` explicitly when forward-mode AD is needed.

Subnetworks with no exposed terminal cannot affect the exposed response (the
system is block-diagonal and their right-hand side is zero). They are pruned
at compile time with a :class:`FloatingSubcircuitWarning`, which also removes a
spurious singularity when such a subnetwork is a lossless resonator. A
genuinely singular system (a lossless resonance that no exposed port excites,
e.g. a ring with ``coupling=0`` exactly on resonance) raises
:class:`CircuitSingularError` on concrete inputs; inside a JAX trace the
affected samples come back as NaN because exceptions cannot cross a traced
computation.

Evidence
--------
Level 3 (cross-solver): the three backends are independent formulations
(global dense LU, sparse LU with an analytic adjoint, pairwise network
reduction) and agree with each other, with an explicit ``(I - S Gamma)^{-1} S``
reference and with closed forms (all-pass ring, MZI, Sagnac mirror, Clements
mesh reproducing its target unitary) to ~1e-12; gradients agree with central
finite differences. See ``tests/test_circuit_engine.py`` and
``docs/CIRCUIT_ENGINE.md``.
"""
from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from photonix.core.backend import HAS_JAX, xp
from photonix.core.types import AliasedSDict, Array, Model, PortPair, SDict, ports_of

from .netlist import Netlist
from .reduction import ReductionPlan, execute_plan, plan_reduction
from .sparse_linalg import SingularSampleError, SparseExposedSolver

__all__ = [
    "Backend",
    "BACKENDS",
    "DENSE_MAX_TERMINALS",
    "DENSE_MAX_BYTES",
    "CircuitSingularError",
    "FloatingSubcircuitWarning",
    "CompiledTopology",
    "compile_topology",
    "gather_values",
    "resolve_backend",
    "solve",
    "solve_matrix",
    "compile_circuit",
]

Backend = Literal["auto", "dense", "sparse", "reduce"]
BACKENDS: tuple[str, ...] = ("auto", "dense", "sparse", "reduce")

#: ``auto`` uses the dense backend up to this many active terminals ...
DENSE_MAX_TERMINALS = 300
#: ... and only while the batched dense matrix ``batch * n^2 * 16`` bytes fits here.
DENSE_MAX_BYTES = 256 * 2**20

Terminal = tuple[str, str]


class CircuitSingularError(ValueError):
    """The circuit system ``I - S Gamma`` is singular for some batch sample.

    Physically the circuit then supports a self-sustained field with no
    excitation: a lossless resonance decoupled from every exposed port.
    ``samples`` holds the flat batch indices concerned (when known).
    """

    def __init__(self, message: str, samples: list[int] | None = None):
        super().__init__(message)
        self.samples = samples or []


class FloatingSubcircuitWarning(UserWarning):
    """Part of the circuit has no path to any exposed port and was pruned."""


# --------------------------------------------------------------------------- #
# Compiled topology
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class CompiledTopology:
    """Value-free structure of a circuit, ready for repeated solves.

    Attributes
    ----------
    terminals
        Active ``(instance, port)`` terminals in solver order (instance order,
        then sorted port names, as in the original solver), after pruning
        floating subnetworks.
    entries
        ``(instance, (in_port, out_port))`` of every gathered ``SDict`` entry.
    entry_rows, entry_cols
        Output (row) and input (column) terminal index of each entry.
    partner
        Connection partner of each terminal, ``-1`` if unconnected.
    exposed
        Terminal index of each exposed port, in ``exposed_names`` order.
    exposed_names
        External port names, in the order of the netlist/``ports`` mapping.
    floating
        Instances (or ``instance:port`` groups) pruned because no exposed
        port can reach them.
    """

    terminals: tuple[Terminal, ...]
    entries: tuple[tuple[str, PortPair], ...]
    entry_rows: np.ndarray
    entry_cols: np.ndarray
    partner: np.ndarray
    exposed: np.ndarray
    exposed_names: tuple[str, ...]
    floating: tuple[str, ...] = ()
    total_terminals: int = 0
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        """Number of active terminals (unknowns of the global system)."""
        return len(self.terminals)

    @property
    def m(self) -> int:
        """Number of exposed ports."""
        return len(self.exposed_names)

    @property
    def nnz(self) -> int:
        """Number of stored ``S`` entries."""
        return len(self.entries)

    def sparse_solver(self) -> SparseExposedSolver:
        """The cached :class:`~photonix.circuit.sparse_linalg.SparseExposedSolver`."""
        solver = self._cache.get("sparse")
        if solver is None:
            solver = SparseExposedSolver(self.entry_rows, self.entry_cols, self.partner, self.exposed, self.n)
            self._cache["sparse"] = solver
        return cast(SparseExposedSolver, solver)

    def reduction_plan(self) -> tuple[ReductionPlan, list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]]:
        """Cached elimination plan plus per-block entry scatter maps.

        Each block item is ``(entry_index, local_row, local_col, ports)``.
        Matched-load terminals (neither connected nor exposed) are dropped.
        """
        cached = self._cache.get("reduce")
        if cached is not None:
            return cast(tuple, cached)
        keep = (self.partner >= 0)
        keep[self.exposed] = True
        block_of: dict[str, list[int]] = {}
        for t, (inst, _) in enumerate(self.terminals):
            if keep[t]:
                block_of.setdefault(inst, []).append(t)
        names = list(block_of)
        local: dict[int, tuple[int, int]] = {}
        for b, inst in enumerate(names):
            for i, t in enumerate(block_of[inst]):
                local[t] = (b, i)
        per_block: list[list[tuple[int, int, int]]] = [[] for _ in names]
        for e, (r, c) in enumerate(zip(self.entry_rows.tolist(), self.entry_cols.tolist(), strict=True)):
            if r in local and c in local:
                b, lr = local[r]
                _, lc = local[c]
                per_block[b].append((e, lr, lc))
        blocks = []
        for b, items in enumerate(per_block):
            arr = np.asarray(items, dtype=np.int64).reshape(-1, 3)
            blocks.append((arr[:, 0], arr[:, 1], arr[:, 2], len(block_of[names[b]])))
        conns = [(j, int(self.partner[j])) for j in range(self.n) if self.partner[j] > j]
        plan = plan_reduction(
            [block_of[inst] for inst in names],
            cast(list[tuple[Hashable, Hashable]], conns),
            [int(t) for t in self.exposed],
        )
        self._cache["reduce"] = (plan, blocks)
        return plan, blocks


def _entry_keys(sd: Any) -> list[PortPair]:
    keys = list(sd.keys()) if isinstance(sd, Mapping) else list(sd)
    for key in keys:
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError(f"SDict keys must be (in_port, out_port) pairs, got {key!r}")
    return keys


def compile_topology(
    instances: Mapping[str, SDict | Iterable[PortPair]],
    connections: Mapping[Terminal, Terminal],
    ports: Mapping[str, Terminal],
) -> CompiledTopology:
    """Compile the value-free structure of a circuit.

    Parameters
    ----------
    instances
        ``instance -> SDict`` (only the keys are used) or ``instance ->``
        iterable of ``(in_port, out_port)`` pairs.
    connections
        Internal links ``(inst_a, port_a) -> (inst_b, port_b)``.
    ports
        Exposed ports ``external_name -> (instance, port)``.

    Returns
    -------
    CompiledTopology
        Validated exactly like :func:`photonix.circuit.evaluate_circuit`
        (``KeyError`` for unknown terminals, ``ValueError`` for a terminal
        connected twice, connected and exposed, or exposed twice).

    Examples
    --------
    >>> topo = compile_topology(
    ...     {"a": [("o1", "o2"), ("o2", "o1")], "b": [("o1", "o2"), ("o2", "o1")]},
    ...     {("a", "o2"): ("b", "o1")},
    ...     {"in": ("a", "o1"), "out": ("b", "o2")},
    ... )
    >>> topo.n, topo.nnz, topo.m, topo.partner.tolist()
    (4, 4, 2, [-1, 2, 1, -1])
    """
    keys_of: dict[str, list[PortPair]] = {inst: _entry_keys(sd) for inst, sd in instances.items()}
    terminals: list[Terminal] = []
    for inst, keys in keys_of.items():
        for p in ports_of(dict.fromkeys(keys, 0)):
            terminals.append((inst, p))
    index = {t: i for i, t in enumerate(terminals)}
    n_all = len(terminals)

    connected: set[Terminal] = set()
    for a, b in connections.items():
        if a not in index or b not in index:
            raise KeyError(f"Connection {a} <-> {b} references an unknown instance port.")
        if a == b:
            raise ValueError(f"A terminal cannot be connected to itself: {a!r}.")
        for terminal in (a, b):
            if terminal in connected:
                raise ValueError(
                    f"Terminal {terminal!r} appears in multiple connections. "
                    "Each terminal may be connected at most once."
                )
            connected.add(terminal)
    exposed_set: set[Terminal] = set()
    for external, terminal in ports.items():
        if terminal not in index:
            raise KeyError(f"External port {external!r} references unknown instance port {terminal!r}.")
        if terminal in connected:
            raise ValueError(f"External port {external!r} references internally connected terminal {terminal!r}.")
        if terminal in exposed_set:
            raise ValueError(f"Instance port {terminal!r} is exposed more than once.")
        exposed_set.add(terminal)

    rows_all: list[int] = []
    cols_all: list[int] = []
    entries_all: list[tuple[str, PortPair]] = []
    for inst, keys in keys_of.items():
        for key in keys:
            p_in, p_out = key
            rows_all.append(index[(inst, p_out)])
            cols_all.append(index[(inst, p_in)])
            entries_all.append((inst, key))
    partner_all = np.full(n_all, -1, dtype=np.int64)
    for a, b in connections.items():
        partner_all[index[a]] = index[b]
        partner_all[index[b]] = index[a]
    exposed_all = np.asarray([index[t] for t in ports.values()], dtype=np.int64)

    if n_all == 0 or not ports:
        empty = np.zeros(0, dtype=np.int64)
        return CompiledTopology((), (), empty, empty, empty, empty, (), (), n_all)

    # Connected components (structural S entries + connections); keep those
    # that contain an exposed terminal.
    parent = list(range(n_all))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for r, c in zip(rows_all, cols_all, strict=True):
        union(r, c)
    for j in range(n_all):
        if partner_all[j] > j:
            union(j, int(partner_all[j]))
    live_roots = {find(int(t)) for t in exposed_all}
    active = np.asarray([find(t) in live_roots for t in range(n_all)], dtype=bool)

    floating: list[str] = []
    if not active.all():
        by_inst: dict[str, list[str]] = {}
        for t in np.nonzero(~active)[0].tolist():
            inst, p = terminals[t]
            by_inst.setdefault(inst, []).append(p)
        for inst, plist in by_inst.items():
            n_ports = sum(1 for t in terminals if t[0] == inst)
            floating.append(inst if len(plist) == n_ports else f"{inst}:{','.join(plist)}")
        shown = ", ".join(floating[:8]) + (" ..." if len(floating) > 8 else "")
        warnings.warn(
            f"{len(floating)} instance(s) have no path to any exposed port and cannot affect the "
            f"circuit response; they are excluded from the solve: {shown}. Check for a missing "
            "connection if this is unintended.",
            FloatingSubcircuitWarning,
            stacklevel=2,
        )

    new_index = np.cumsum(active) - 1
    rows_np = np.asarray(rows_all, dtype=np.int64)
    cols_np = np.asarray(cols_all, dtype=np.int64)
    keep_entry = active[rows_np] if rows_np.size else np.zeros(0, dtype=bool)
    partner = partner_all[active]
    partner = np.where(partner >= 0, new_index[np.maximum(partner, 0)], -1)
    return CompiledTopology(
        terminals=tuple(t for t, a in zip(terminals, active.tolist(), strict=True) if a),
        entries=tuple(e for e, k in zip(entries_all, keep_entry.tolist(), strict=True) if k),
        entry_rows=new_index[rows_np[keep_entry]],
        entry_cols=new_index[cols_np[keep_entry]],
        partner=partner.astype(np.int64),
        exposed=new_index[exposed_all],
        exposed_names=tuple(ports),
        floating=tuple(floating),
        total_terminals=n_all,
    )


# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #
def _is_tracer(x: Any) -> bool:
    if not HAS_JAX:
        return False
    import jax

    return isinstance(x, jax.core.Tracer)


def _batch_shape(instances: Mapping[str, Any]) -> tuple[int, ...]:
    batch: tuple[int, ...] = ()
    for sd in instances.values():
        for v in sd.values():
            batch = np.broadcast_shapes(batch, np.shape(v))
    return batch


def gather_values(topology: CompiledTopology, instances: Mapping[str, SDict]) -> Array:
    """Stack the ``SDict`` entry values in topology order.

    Parameters
    ----------
    topology
        From :func:`compile_topology` on the same structure.
    instances
        ``instance -> SDict``.

    Returns
    -------
    Array
        ``(*batch, nnz)`` complex array on the active backend, ``batch`` being
        the broadcast shape of every value of every instance (as in the
        original solver). Values shared between entries (the same array
        object, e.g. a deduplicated instance) are converted once. When no
        value is a JAX tracer the stacking runs in NumPy and is moved to the
        backend once.

    Examples
    --------
    >>> import numpy as np
    >>> sd = {("o1", "o2"): np.array([1j, 2j]), ("o2", "o1"): 0.5}
    >>> topo = compile_topology({"w": sd}, {}, {"a": ("w", "o1"), "b": ("w", "o2")})
    >>> np.asarray(gather_values(topo, {"w": sd})).tolist()
    [[1j, (0.5+0j)], [2j, (0.5+0j)]]
    """
    batch = _batch_shape(instances)
    unique: list[Any] = []
    slot: dict[int, int] = {}
    order: list[int] = []
    for inst, key in topology.entries:
        v = instances[inst][key]
        pos = slot.get(id(v))
        if pos is None:
            pos = slot[id(v)] = len(unique)
            unique.append(v)
        order.append(pos)
    idx = np.asarray(order, dtype=np.int64)
    if any(_is_tracer(v) for v in unique):
        stacked = xp.stack([xp.broadcast_to(xp.asarray(v, dtype=complex), batch) for v in unique], axis=-1)
        return xp.take(stacked, xp.asarray(idx), axis=-1)
    if not unique:
        return xp.zeros((*batch, 0), dtype=complex)
    host = np.stack([np.broadcast_to(np.asarray(v, dtype=complex), batch) for v in unique], axis=-1)
    return xp.asarray(np.ascontiguousarray(host[..., idx]))


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #
def resolve_backend(topology: CompiledTopology, batch_shape: tuple[int, ...], backend: str = "auto") -> str:
    """The concrete backend ``"auto"`` resolves to (others are validated and returned).

    Examples
    --------
    >>> topo = compile_topology({"w": [("o1", "o2"), ("o2", "o1")]}, {}, {"a": ("w", "o1"), "b": ("w", "o2")})
    >>> resolve_backend(topo, (101,)), resolve_backend(topo, (), "reduce")
    ('dense', 'reduce')
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown circuit backend {backend!r}; choose one of {BACKENDS}")
    if backend != "auto":
        return backend
    n = topology.n
    samples = math.prod(batch_shape) if batch_shape else 1
    if n <= DENSE_MAX_TERMINALS and samples * n * n * 16 <= DENSE_MAX_BYTES:
        return "dense"
    return "sparse"


def _singular_message(topology: CompiledTopology, backend: str) -> str:
    msg = (
        f"I - S Gamma is singular ({backend} backend): the circuit supports a self-sustained, "
        "unexcited resonance -- a lossless loop that no exposed port couples to (e.g. a ring with "
        "coupling=0 exactly on resonance, or a zero-phase lossless loop)."
    )
    if topology.floating:
        msg += f" Pruned floating instances: {', '.join(topology.floating[:8])}."
    return msg


def _scatter(target: Array, rows: np.ndarray, cols: np.ndarray, vals: Array) -> Array:
    if hasattr(target, "at"):
        return target.at[..., rows, cols].set(vals)
    target[..., rows, cols] = vals
    return target


def _solve_dense(topology: CompiledTopology, values: Array) -> Array:
    n, m = topology.n, topology.m
    batch = tuple(values.shape[:-1])
    rows, cols, partner, exposed = topology.entry_rows, topology.entry_cols, topology.partner, topology.exposed
    conn = np.nonzero(partner[cols] >= 0)[0]
    k_of = np.full(n, -1, dtype=np.int64)
    k_of[exposed] = np.arange(m)
    rhs = np.nonzero(k_of[cols] >= 0)[0]
    A = xp.zeros((*batch, n, n), dtype=values.dtype)
    A = _scatter(A, rows[conn], partner[cols[conn]], -xp.take(values, xp.asarray(conn), axis=-1))
    A = A + xp.eye(n, dtype=values.dtype)
    B = xp.zeros((*batch, n, m), dtype=values.dtype)
    B = _scatter(B, rows[rhs], k_of[cols[rhs]], xp.take(values, xp.asarray(rhs), axis=-1))
    X = xp.linalg.solve(A, B)
    return xp.take(X, xp.asarray(exposed), axis=-2)


def _solve_reduce(topology: CompiledTopology, values: Array) -> Array:
    plan, blocks = topology.reduction_plan()
    batch = tuple(values.shape[:-1])
    # Concrete inputs run in NumPy (same algebra, no per-shape XLA compiles of
    # the many small block operations); traced inputs stay on the backend.
    traced = _is_tracer(values)
    ns: Any = xp if traced else np
    vals = values if traced else np.asarray(values)
    mats = []
    for entry_idx, lr, lc, p in blocks:
        S = ns.zeros((*batch, p, p), dtype=vals.dtype)
        if entry_idx.size:
            S = _scatter(S, lr, lc, ns.take(vals, ns.asarray(entry_idx), axis=-1))
        mats.append(S)
    with np.errstate(divide="ignore", invalid="ignore"):
        Y = execute_plan(plan, mats)
    return Y if traced else xp.asarray(Y)


def solve_matrix(topology: CompiledTopology, values: Array, *, backend: Backend | str = "auto") -> Array:
    """Exposed scattering matrix ``Y[..., out, in]`` of a compiled circuit.

    Parameters
    ----------
    topology
        From :func:`compile_topology`.
    values
        ``(*batch, nnz)`` entry values from :func:`gather_values`.
    backend
        ``"auto"``, ``"dense"``, ``"sparse"`` or ``"reduce"`` (module docs).

    Returns
    -------
    Array
        ``(*batch, m, m)``; ``Y[..., i, k]`` is the wave leaving exposed port
        ``i`` for unit excitation at exposed port ``k``.

    Raises
    ------
    CircuitSingularError
        If the system is singular for a sample and the inputs are concrete.

    Examples
    --------
    >>> import numpy as np
    >>> sd = {("o1", "o2"): np.exp(-0.5j), ("o2", "o1"): np.exp(-0.5j)}
    >>> topo = compile_topology({"w": sd}, {}, {"a": ("w", "o1"), "b": ("w", "o2")})
    >>> Y = solve_matrix(topo, gather_values(topo, {"w": sd}), backend="sparse")
    >>> bool(np.isclose(complex(Y[1, 0]), np.exp(-0.5j)))
    True
    """
    batch = tuple(values.shape[:-1])
    m = topology.m
    if m == 0:
        return xp.zeros((*batch, 0, 0), dtype=complex)
    if topology.nnz == 0:
        return xp.zeros((*batch, m, m), dtype=complex)
    if not (hasattr(values, "dtype") and np.issubdtype(np.dtype(values.dtype), np.complexfloating)):
        values = xp.asarray(values, dtype=complex)
    chosen = resolve_backend(topology, batch, backend)
    traced = _is_tracer(values)
    if chosen == "sparse":
        solver = topology.sparse_solver()
        if traced:
            return solver(values)
        try:
            return xp.asarray(solver.solve_numpy(np.asarray(values)))
        except SingularSampleError as exc:
            raise CircuitSingularError(_singular_message(topology, "sparse") + f" Samples: {exc.samples[:10]}.",
                                       exc.samples) from None
    try:
        Y = _solve_dense(topology, values) if chosen == "dense" else _solve_reduce(topology, values)
    except np.linalg.LinAlgError as exc:
        raise CircuitSingularError(_singular_message(topology, chosen)) from exc
    if not traced:
        y_host = np.asarray(Y)
        if not np.all(np.isfinite(y_host)) and np.all(np.isfinite(np.asarray(values))):
            bad = np.nonzero(~np.isfinite(y_host.reshape(-1, m * m)).all(axis=-1))[0].tolist()
            raise CircuitSingularError(_singular_message(topology, chosen) + f" Samples: {bad[:10]}.", bad)
    return Y


def solve(topology: CompiledTopology, values: Array, *, backend: Backend | str = "auto") -> SDict:
    """Composite :data:`SDict` ``{(external_in, external_out): value}``.

    Same contract as :func:`photonix.circuit.evaluate_circuit`; see
    :func:`solve_matrix` for the parameters.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.exp(-0.5j)
    >>> sd = {("o1", "o2"): t, ("o2", "o1"): t}
    >>> topo = compile_topology({"a": sd, "b": sd}, {("a", "o2"): ("b", "o1")},
    ...                         {"in": ("a", "o1"), "out": ("b", "o2")})
    >>> s = solve(topo, gather_values(topo, {"a": sd, "b": sd}))
    >>> bool(np.isclose(complex(s[("in", "out")]), t * t))
    True
    """
    Y = solve_matrix(topology, values, backend=backend)
    names = topology.exposed_names
    keys = [(e_in, e_out) for e_in in names for e_out in names]
    m = len(names)
    if HAS_JAX and not isinstance(Y, np.ndarray) and not _is_tracer(Y):
        # One batched host->device transfer instead of m^2 eager slicing ops.
        import jax

        host = np.asarray(Y)
        pieces = jax.device_put([np.array(host[..., i, k]) for k in range(m) for i in range(m)])  # 0-d stays 0-d
        return dict(zip(keys, pieces, strict=True))
    return {key: Y[..., i, k] for key, (k, i) in zip(keys, ((k, i) for k in range(m) for i in range(m)), strict=True)}


# --------------------------------------------------------------------------- #
# Structural caching
# --------------------------------------------------------------------------- #
class _TopologyCache:
    """Small LRU of compiled topologies keyed by the full circuit structure."""

    def __init__(self, maxsize: int = 16):
        self.maxsize = maxsize
        self._data: OrderedDict[Hashable, CompiledTopology] = OrderedDict()

    def get(
        self,
        instances: Mapping[str, SDict],
        connections: Mapping[Terminal, Terminal],
        ports: Mapping[str, Terminal],
    ) -> CompiledTopology:
        seen: dict[int, tuple] = {}
        struct = []
        for inst, sd in instances.items():
            keys = seen.get(id(sd))
            if keys is None:
                keys = seen[id(sd)] = tuple(sd.keys())
            struct.append((inst, keys))
        key = (tuple(struct), tuple(connections.items()), tuple(ports.items()))
        try:
            topo = self._data[key]
        except (KeyError, TypeError):
            topo = compile_topology(instances, connections, ports)
            try:
                self._data[key] = topo
            except TypeError:  # unhashable port names: compile every time
                return topo
            if len(self._data) > self.maxsize:
                self._data.popitem(last=False)
            return topo
        self._data.move_to_end(key)
        return topo

    def clear(self) -> None:
        self._data.clear()


#: Topologies reused by :func:`photonix.circuit.evaluate_circuit`.
TOPOLOGY_CACHE = _TopologyCache()


# --------------------------------------------------------------------------- #
# Netlist compilation with instance deduplication
# --------------------------------------------------------------------------- #
def _freeze(v: Any) -> Hashable:
    """A hashable stand-in for a setting value; equal keys imply equal values."""
    if v is None or isinstance(v, (bool, int, str)):
        return (type(v).__name__, v)
    if isinstance(v, float):
        return ("float", v.hex())
    if isinstance(v, complex):
        return ("complex", v.real.hex(), v.imag.hex())
    if isinstance(v, np.generic):
        return ("np", v.dtype.str, v.tobytes())
    if isinstance(v, np.ndarray) and v.dtype.kind in "biufc" and v.size <= 4096:
        return ("nd", v.dtype.str, v.shape, v.tobytes())
    if isinstance(v, (tuple, list)):
        return (type(v).__name__, tuple(_freeze(x) for x in v))
    if isinstance(v, Mapping):
        return ("map", tuple(sorted(((str(k), _freeze(x)) for k, x in v.items()), key=lambda kv: kv[0])))
    # JAX arrays and tracers, callables, arbitrary objects: identity. The
    # object is alive for the whole call, so equal ids mean the same value.
    return ("id", id(v))


def compile_circuit(
    netlist: Netlist,
    models: Mapping[str, Callable[..., SDict]] | None = None,
    *,
    backend: Backend | str = "auto",
    dedupe: bool = True,
    port_aliases: Mapping[str, str] | None = None,
) -> Model:
    """Compile a :class:`~photonix.circuit.Netlist` into a fast, differentiable model.

    A drop-in for :func:`photonix.circuit.circuit_from_netlist` that also
    selects the solver backend and deduplicates instance evaluations.

    Parameters
    ----------
    netlist
        The circuit topology (validated now; read again at every call).
    models
        ``model_name -> callable``; defaults to
        :data:`photonix.components.MODELS`.
    backend
        Engine backend (module docs). ``"auto"`` by default.
    dedupe
        Evaluate instances sharing one model callable and identical settings
        once per call (e.g. the hundreds of identical 50:50 couplers of an MZI
        mesh). Settings are compared by value for numbers, strings, tuples
        and NumPy arrays, and by identity otherwise (JAX arrays, tracers,
        callables), so gradients still flow to every instance that has its
        own traced setting. Requires pure models; pass ``False`` for a model
        that draws fresh random numbers on every call.
    port_aliases
        Optional legacy port names resolved on lookup only.

    Returns
    -------
    Model
        ``f(*, wl=1.55, **overrides) -> SDict`` with ``overrides`` given as
        ``{instance: {param: value}}``.

    Examples
    --------
    >>> import numpy as np
    >>> from photonix.circuit import Netlist
    >>> nl = Netlist(name="pair")
    >>> _ = nl.add("a", "straight", length=10.0).add("b", "straight", length=10.0)
    >>> _ = nl.connect(("a", "o2"), ("b", "o1"))
    >>> _ = nl.expose("in", ("a", "o1")).expose("out", ("b", "o2"))
    >>> f = compile_circuit(nl, backend="sparse")
    >>> abs(abs(complex(f(wl=1.55)[("in", "out")])) - 1.0) < 1e-12
    True
    """
    netlist.validate()
    if backend not in BACKENDS:
        raise ValueError(f"unknown circuit backend {backend!r}; choose one of {BACKENDS}")
    if models is None:
        from photonix.components import MODELS

        registry = cast(Mapping[str, Callable[..., SDict]], MODELS)
    else:
        registry = models
    cache = _TopologyCache(maxsize=4)

    def model(*, wl=1.55, **overrides) -> SDict:
        inst_sdicts: dict[str, SDict] = {}
        memo: dict[Hashable, SDict] = {}
        for inst, model_name in netlist.instances.items():
            if model_name not in registry:
                raise KeyError(
                    f"Model {model_name!r} for instance {inst!r} not in registry. "
                    f"Known models: {sorted(registry)}"
                )
            fn = registry[model_name]
            settings = netlist.merged_settings(inst, overrides.get(inst))
            if dedupe:
                key = (id(fn), tuple(sorted((k, _freeze(v)) for k, v in settings.items())))
                sd = memo.get(key)
                if sd is None:
                    sd = memo[key] = fn(wl=wl, **settings)
            else:
                sd = fn(wl=wl, **settings)
            inst_sdicts[inst] = sd
        topo = cache.get(inst_sdicts, netlist.connections, netlist.ports)
        out = solve(topo, gather_values(topo, inst_sdicts), backend=backend)
        return AliasedSDict(out, aliases=port_aliases) if port_aliases else out

    model.__name__ = netlist.name or "circuit"
    return model
