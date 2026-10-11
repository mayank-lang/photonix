"""Subnetwork-growth reduction of an S-parameter network (fully traceable).

This module is the third, independent circuit solver of
:mod:`photonix.circuit.engine`. Instead of assembling one global linear system
(:func:`photonix.circuit.evaluate_circuit`), it eliminates internal connections
one at a time on small dense blocks, the *subnetwork growth* method of
Filipsson (1981) and Gupta, Garg & Chadha (1981):

1. every instance starts as its own block ``(S, terminals)`` with
   ``S[..., out, in]`` (the photonix row = output, column = input convention);
2. two blocks joined by a connection are merged block-diagonally;
3. a connection between ports ``k`` and ``l`` of one block (``a_k = b_l``,
   ``a_l = b_k``) is removed with :func:`innerconnect`;
4. connections are processed greedily so the merged block stays small.

Every step is plain array algebra on the active backend (``xp``), so the
result is differentiable in both forward and reverse mode, ``jit``-able, and
needs no host callback -- the reason to prefer it over the sparse backend for
forward-mode AD (``jax.jacfwd``) or accelerator-only programs. Its cost grows
with the size of the largest intermediate block rather than the total number
of terminals, which suits long cascades and meshes but not densely
interconnected networks.

References
----------
* S. Filipsson, "A new general computer algorithm for S-matrix calculation of
  interconnected multiports", Proc. 11th European Microwave Conf. (1981)
  700-704.
* K. C. Gupta, R. Garg, R. Chadha, *Computer-Aided Design of Microwave
  Circuits* (Artech House, 1981), ch. 13.
* scikit-rf, ``skrf.network.innerconnect_s`` (the same closed form).
"""
from __future__ import annotations

import heapq
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from photonix.core.backend import xp
from photonix.core.types import Array

__all__ = ["innerconnect", "block_diagonal", "reduce_network", "ReductionPlan", "plan_reduction", "execute_plan"]

Terminal = tuple[str, str]


def _ns(*arrays: Any) -> Any:
    """NumPy for host arrays (no per-shape compilation), the backend otherwise."""
    return np if all(isinstance(a, np.ndarray) for a in arrays) else xp


def innerconnect(S: Array, k: int, l: int) -> Array:  # noqa: E741 - textbook index names
    """Connect ports ``k`` and ``l`` of one network and remove them.

    With ``b = S a`` and the connection ``a_k = b_l``, ``a_l = b_k`` the
    remaining ports see (Gupta, Garg & Chadha 1981, eq. 13.12)::

        S'_ij = S_ij + [S_kj S_il (1 - S_lk) + S_lj S_ik (1 - S_kl)
                        + S_kj S_ll S_ik + S_lj S_kk S_il]
                       / [(1 - S_kl)(1 - S_lk) - S_kk S_ll]

    Parameters
    ----------
    S
        ``(..., p, p)`` scattering matrix, row = output port, column = input
        port. Leading axes are batch axes (wavelength, parameters).
    k, l
        The two distinct port indices to join.

    Returns
    -------
    Array
        ``(..., p - 2, p - 2)`` matrix over the remaining ports in their
        original order. Pure ``xp`` algebra: differentiable and ``jit``-able.
        The denominator vanishes for a lossless loop at resonance that no
        remaining port can excite; the result is then non-finite.

    Examples
    --------
    Two independent phase delays (ports ``a1, a2, b1, b2``); joining ``a2``
    to ``b1`` leaves their cascade:

    >>> import numpy as np
    >>> ta, tb = np.exp(-0.3j), np.exp(-0.4j)
    >>> S = np.zeros((4, 4), complex)
    >>> S[1, 0] = S[0, 1] = ta
    >>> S[3, 2] = S[2, 3] = tb
    >>> S2 = innerconnect(S, 1, 2)
    >>> S2.shape, bool(np.isclose(S2[1, 0], np.exp(-0.7j)))
    ((2, 2), True)
    """
    ns = _ns(S)
    S = ns.asarray(S)
    p = S.shape[-1]
    if S.shape[-2] != p:
        raise ValueError(f"S must be square in its last two axes, got shape {S.shape}")
    k, l = int(k), int(l)  # noqa: E741
    if not (0 <= k < p and 0 <= l < p) or k == l:
        raise ValueError(f"innerconnect needs two distinct port indices in [0, {p}), got {k} and {l}")
    keep = np.asarray([i for i in range(p) if i != k and i != l], dtype=np.int64)
    if ns is np:
        return _innerconnect_math(S, k, l, keep, np)
    # Backend arrays: one compiled kernel per (batch, p) shape, with the port
    # indices as runtime arguments, so a long elimination reuses a handful of
    # compilations instead of dispatching ~25 eager operations per step.
    return _jitted("innerconnect")(S, xp.asarray(k), xp.asarray(l), xp.asarray(keep))


def _innerconnect_math(S: Any, k: Any, l: Any, keep: Any, ns: Any) -> Any:  # noqa: E741
    Skl = S[..., k, l][..., None, None]
    Slk = S[..., l, k][..., None, None]
    Skk = S[..., k, k][..., None, None]
    Sll = S[..., l, l][..., None, None]
    kept_rows = ns.take(S, keep, axis=-2)
    col_k = kept_rows[..., :, k][..., :, None]          # S_ik, (..., p-2, 1)
    col_l = kept_rows[..., :, l][..., :, None]          # S_il
    kept_cols = ns.take(S, keep, axis=-1)
    row_k = kept_cols[..., k, :][..., None, :]          # S_kj, (..., 1, p-2)
    row_l = kept_cols[..., l, :][..., None, :]          # S_lj
    core = ns.take(kept_rows, keep, axis=-1)
    denom = (1.0 - Skl) * (1.0 - Slk) - Skk * Sll
    num = (
        row_k * col_l * (1.0 - Slk)
        + row_l * col_k * (1.0 - Skl)
        + row_k * Sll * col_k
        + row_l * Skk * col_l
    )
    return core + num / denom


def _block_diagonal_math(A: Any, B: Any, ns: Any) -> Any:
    batch = np.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    pa, pb = A.shape[-1], B.shape[-1]
    dtype = ns.result_type(A.dtype, B.dtype)
    A = ns.broadcast_to(A, (*batch, pa, pa)).astype(dtype)
    B = ns.broadcast_to(B, (*batch, pb, pb)).astype(dtype)
    top = ns.concatenate([A, ns.zeros((*batch, pa, pb), dtype=dtype)], axis=-1)
    bottom = ns.concatenate([ns.zeros((*batch, pb, pa), dtype=dtype), B], axis=-1)
    return ns.concatenate([top, bottom], axis=-2)


_JITTED: dict[str, Any] = {}


def _jitted(name: str) -> Any:
    fn = _JITTED.get(name)
    if fn is None:
        import jax

        if name == "innerconnect":
            fn = jax.jit(lambda S, k, l, keep: _innerconnect_math(S, k, l, keep, xp))  # noqa: E741
        else:
            fn = jax.jit(lambda A, B: _block_diagonal_math(A, B, xp))
        _JITTED[name] = fn
    return fn


def block_diagonal(A: Array, B: Array) -> Array:
    """Block-diagonal merge ``[[A, 0], [0, B]]`` of two batched square matrices.

    The batch axes of ``A`` and ``B`` are broadcast against each other.

    Examples
    --------
    >>> import numpy as np
    >>> block_diagonal(np.eye(1), 2 * np.eye(2)).tolist()
    [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    """
    ns = _ns(A, B)
    A, B = ns.asarray(A), ns.asarray(B)
    if ns is np:
        return _block_diagonal_math(A, B, np)
    return _jitted("block_diagonal")(A, B)


# --------------------------------------------------------------------------- #
# Elimination planning (topology only, no values)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReductionPlan:
    """A precomputed elimination sequence for :func:`reduce_network`.

    Attributes
    ----------
    steps
        ``("merge", a, b)`` replaces blocks ``a`` and ``b`` by their
        block-diagonal merge (stored in slot ``a``, ports of ``a`` first);
        ``("inner", a, k, l)`` joins local ports ``k``/``l`` of block ``a``.
    roots
        Block slots that survive, in output order.
    output_order
        Permutation taking the concatenated surviving ports to ``external``.
    max_block
        Largest intermediate block size (ports), the cost driver.
    """

    steps: tuple[tuple, ...]
    roots: tuple[int, ...]
    output_order: tuple[int, ...]
    max_block: int


def plan_reduction(
    block_ports: Sequence[Sequence[Hashable]],
    connections: Sequence[tuple[Hashable, Hashable]],
    external: Sequence[Hashable],
    *,
    order: Literal["min_degree", "sequential"] = "min_degree",
) -> ReductionPlan:
    """Plan the elimination order for :func:`reduce_network`.

    ``order="min_degree"`` greedily processes the connection whose execution
    leaves the smallest block (a merge of blocks with ``p_a`` and ``p_b`` ports
    sharing ``c`` connections leaves ``p_a + p_b - 2c`` ports), with
    self-connections inside a block always taken first. ``"sequential"``
    processes connections in the given order (useful as a cross-check).

    Examples
    --------
    >>> plan = plan_reduction([[("a", "o1"), ("a", "o2")], [("b", "o1"), ("b", "o2")]],
    ...                       [(("a", "o2"), ("b", "o1"))], [("a", "o1"), ("b", "o2")])
    >>> [s[0] for s in plan.steps], plan.max_block
    (['merge', 'inner'], 4)
    """
    ports: list[list[Hashable]] = [list(p) for p in block_ports]
    owner: dict[Hashable, int] = {}
    for slot, plist in enumerate(ports):
        for t in plist:
            if t in owner:
                raise ValueError(f"terminal {t!r} belongs to more than one block")
            owner[t] = slot
    for end_a, end_b in connections:
        for t in (end_a, end_b):
            if t not in owner:
                raise KeyError(f"connection references unknown terminal {t!r}")
    ext_set = set(external)
    for t in external:
        if t not in owner:
            raise KeyError(f"external terminal {t!r} is not a block port")

    parent = list(range(len(ports)))  # union-find over block slots

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    steps: list[tuple] = []
    max_block = max((len(p) for p in ports), default=0)
    pending = list(connections)

    def execute(a_term: Hashable, b_term: Hashable) -> None:
        nonlocal max_block
        ra, rb = find(owner[a_term]), find(owner[b_term])
        if ra != rb:
            steps.append(("merge", ra, rb))
            ports[ra] = ports[ra] + ports[rb]
            ports[rb] = []
            parent[rb] = ra
            max_block = max(max_block, len(ports[ra]))
        plist = ports[ra]
        k, l = plist.index(a_term), plist.index(b_term)  # noqa: E741
        steps.append(("inner", ra, k, l))
        ports[ra] = [t for t in plist if t != a_term and t != b_term]

    if order == "sequential":
        for end_a, end_b in pending:
            execute(end_a, end_b)
    elif order == "min_degree":
        # Connections grouped by unordered block pair; lazy-invalidated heap.
        version = [0] * len(ports)
        by_pair: dict[tuple[int, int], list[tuple[Hashable, Hashable]]] = {}
        for end_a, end_b in pending:
            ra, rb = owner[end_a], owner[end_b]
            by_pair.setdefault((min(ra, rb), max(ra, rb)), []).append((end_a, end_b))
        adjacency: dict[int, set[tuple[int, int]]] = {}
        for key in by_pair:
            adjacency.setdefault(key[0], set()).add(key)
            adjacency.setdefault(key[1], set()).add(key)
        heap: list[tuple[int, int, tuple[int, int], int, int]] = []
        counter = 0

        def cost(key: tuple[int, int]) -> int:
            a, b = key
            c = len(by_pair[key])
            if a == b:
                return len(ports[a]) - 2 * c - 10**9   # self-loops first
            return len(ports[a]) + len(ports[b]) - 2 * c

        def push(key: tuple[int, int]) -> None:
            nonlocal counter
            heapq.heappush(heap, (cost(key), counter, key, version[key[0]], version[key[1]]))
            counter += 1

        for key in by_pair:
            push(key)
        while heap:
            _, _, key, va, vb = heapq.heappop(heap)
            if key not in by_pair or version[key[0]] != va or version[key[1]] != vb:
                continue
            a, b = key
            conns = by_pair.pop(key)
            adjacency[a].discard(key)
            adjacency[b].discard(key)
            for ta, tb in conns:
                execute(ta, tb)
            if a == b:
                version[a] += 1
                for other in list(adjacency.get(a, ())):
                    push(other)
                continue
            # Block b is absorbed into a: re-key every pair that touched b.
            for old in list(adjacency.get(b, ())):
                x, y = old
                partner_slot = a if x == y else (y if x == b else x)
                moved = by_pair.pop(old)
                adjacency[partner_slot].discard(old)
                new = (min(a, partner_slot), max(a, partner_slot))
                by_pair.setdefault(new, []).extend(moved)
                adjacency.setdefault(new[0], set()).add(new)
                adjacency.setdefault(new[1], set()).add(new)
            adjacency[b] = set()
            version[a] += 1
            version[b] += 1
            for other in list(adjacency.get(a, ())):
                push(other)
    else:
        raise ValueError(f"unknown elimination order {order!r}; use 'min_degree' or 'sequential'")

    roots = tuple(sorted({find(owner[t]) for t in external}))
    survivors: list[Hashable] = []
    for r in roots:
        survivors.extend(ports[r])
    position = {t: i for i, t in enumerate(survivors)}
    missing = [t for t in survivors if t not in ext_set]
    if missing:
        raise ValueError(
            f"terminals {missing[:5]!r} are neither connected nor external; drop matched-load "
            "terminals from the blocks before planning"
        )
    return ReductionPlan(
        steps=tuple(steps),
        roots=roots,
        output_order=tuple(position[t] for t in external),
        max_block=max_block,
    )


def execute_plan(plan: ReductionPlan, blocks: Sequence[Array]) -> Array:
    """Run a :class:`ReductionPlan` on per-block matrices (``xp`` algebra).

    ``blocks[i]`` is the ``(..., p_i, p_i)`` matrix of block slot ``i`` with
    its ports in the order given to :func:`plan_reduction`. Returns the
    ``(..., m, m)`` matrix over the plan's external terminals. A plan depends
    only on topology, so one plan serves every parameter value.

    Examples
    --------
    >>> import numpy as np
    >>> d = lambda phi: np.array([[0, np.exp(-1j * phi)], [np.exp(-1j * phi), 0]])
    >>> plan = plan_reduction([[("a", "o1"), ("a", "o2")], [("b", "o1"), ("b", "o2")]],
    ...                       [(("a", "o2"), ("b", "o1"))], [("a", "o1"), ("b", "o2")])
    >>> S = execute_plan(plan, [d(0.3), d(0.4)])
    >>> bool(np.isclose(S[1, 0], np.exp(-0.7j)))
    True
    """
    slots: list[Array | None] = list(blocks)
    for step in plan.steps:
        if step[0] == "merge":
            _, a, b = step
            sa, sb = slots[a], slots[b]
            assert sa is not None and sb is not None
            slots[a] = block_diagonal(sa, sb)
            slots[b] = None
        else:
            _, a, k, l = step  # noqa: E741
            sa = slots[a]
            assert sa is not None
            slots[a] = innerconnect(sa, k, l)
    result: Array | None = None
    for r in plan.roots:
        block = slots[r]
        assert block is not None
        result = block if result is None else block_diagonal(result, block)
    if result is None:
        return xp.zeros((0, 0), dtype=complex)
    ns = _ns(result)
    idx = ns.asarray(plan.output_order, dtype=int)
    return ns.take(ns.take(result, idx, axis=-2), idx, axis=-1)


def reduce_network(
    blocks: Mapping[str, tuple[Array, Sequence[str]]],
    connections: Mapping[Terminal, Terminal],
    external: Sequence[Terminal] | None = None,
    *,
    order: Literal["min_degree", "sequential"] = "min_degree",
) -> tuple[Array, tuple[Terminal, ...]]:
    """Reduce a network of dense blocks to the S-matrix of its external ports.

    Parameters
    ----------
    blocks
        ``name -> (S, ports)`` with ``S`` shaped ``(..., p, p)`` (row = output,
        column = input) and ``ports`` the ``p`` port names in matrix order.
    connections
        ``(block, port) -> (block, port)`` internal links, each terminal at
        most once.
    external
        Terminals to keep, in output order. ``None`` keeps every terminal that
        is not connected, in block order. Terminals that are neither connected
        nor external are terminated in matched loads (dropped).
    order
        Elimination order, see :func:`plan_reduction`.

    Returns
    -------
    (S, terminals)
        ``S`` is ``(..., m, m)`` over ``terminals`` (row = output). Blocks
        unconnected to any external terminal are ignored.

    Examples
    --------
    Two cascaded phase delays add their phases:

    >>> import numpy as np
    >>> d = lambda phi: np.array([[0, np.exp(-1j * phi)], [np.exp(-1j * phi), 0]])
    >>> S, t = reduce_network({"a": (d(0.3), ("o1", "o2")), "b": (d(0.4), ("o1", "o2"))},
    ...                       {("a", "o2"): ("b", "o1")})
    >>> t
    (('a', 'o1'), ('b', 'o2'))
    >>> bool(np.isclose(S[1, 0], np.exp(-0.7j)))
    True
    """
    connected: set[Terminal] = set()
    for a, b in connections.items():
        for t in (a, b):
            if t in connected:
                raise ValueError(f"terminal {t!r} appears in more than one connection")
            connected.add(t)
    if external is None:
        external = [(name, p) for name, (_, plist) in blocks.items() for p in plist
                    if (name, p) not in connected]
    external = list(external)
    keep_set = connected | set(external)
    names = list(blocks)
    block_ports: list[list[Terminal]] = []
    mats: list[Array] = []
    for name in names:
        S, plist = blocks[name]
        ns = _ns(S)
        S = ns.asarray(S)
        idx = [i for i, p in enumerate(plist) if (name, p) in keep_set]
        if len(idx) != len(plist):
            sel = ns.asarray(idx, dtype=int)
            S = ns.take(ns.take(S, sel, axis=-2), sel, axis=-1)
        block_ports.append([(name, plist[i]) for i in idx])
        mats.append(S)
    plan = plan_reduction(block_ports, list(connections.items()), external, order=order)
    return execute_plan(plan, mats), tuple(external)
