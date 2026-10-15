"""Sparse LU solve of the circuit equations, differentiable in every JAX mode.

The circuit solver (:mod:`photonix.circuit.engine`) needs, for every batch
sample (wavelength, parameter point), the exposed block of

    X = (I - S Gamma)^{-1} S P_E

where ``S`` is the block-diagonal instance scattering matrix stored as
``S[out, in]``, ``Gamma`` the symmetric connection permutation and ``P_E`` the
``n x m`` selector of the ``m`` exposed terminals. Because ``Gamma`` is a
permutation, column ``j`` of ``S Gamma`` is column ``gamma(j)`` of ``S``: the
matrix ``A = I - S Gamma`` has at most ``nnz(S) + n`` nonzeros and is never
formed densely. Only ``m`` right-hand sides are solved and only the ``m x m``
block ``Y = X[E, :]`` is returned, so the undifferentiated solve needs
``O(batch * (nnz + m^2))`` memory.

Derivatives
-----------
``dX = A^{-1} dS (P_E + Gamma X)``. JAX vector-Jacobian products are
*transposes* (not Hermitian adjoints) of this holomorphic Jacobian: from a
cotangent ``Ybar`` scattered into rows ``E`` (``Xbar = P_E Ybar``), let
``Z = A^{-T} Xbar``. The cotangent of stored entry ``(r, c)`` of ``S`` is::

    vbar_rc = Z[r, k_c] [c is exposed terminal k_c]
              + sum_k Z[r, k] X[gamma(c), k]          (c connected)

(Giles 2008, "Collected matrix derivative results for forward and reverse
mode algorithmic differentiation"); :meth:`SparseExposedSolver.vjp_numpy`
implements it directly in NumPy, and the transposed solve reuses the SuperLU
factorization (``trans="T"``).

Under the JAX backend the solve runs on the host through
:func:`jax.pure_callback`. The primal ``Y`` is a callback returning only the
``m x m`` block, wrapped in :func:`jax.custom_jvp` whose tangent is the
expression above with ``A^{-1}`` applied by :func:`jax.lax.custom_linear_solve`
(SuperLU callbacks for the solve and the transposed solve, ``A(v)`` as the
differentiable ``matvec``). JAX derives reverse mode by transposing that
linear tangent, so ``jit``, ``vmap``, ``grad``, ``jvp``/``jacfwd`` and
higher-order derivatives (``jax.hessian``, reverse-over-reverse) all work.
Under differentiation the ``n x m`` solution ``X`` is a residual (``O(batch n
m)`` memory, versus ``O(batch n^2)`` for the dense backend), and consecutive
callbacks with the same values reuse one LU factorization per sample (bounded
by :attr:`SparseExposedSolver.FACTOR_CACHE_BYTES`). Under the NumPy backend
SciPy is called directly.

References
----------
* T. A. Davis, *Direct Methods for Sparse Linear Systems* (SIAM, 2006).
* X. S. Li, "An overview of SuperLU", ACM TOMS 31, 302 (2005).
* M. B. Giles, "Collected matrix derivative results for forward and reverse
  mode algorithmic differentiation", in *Advances in Automatic
  Differentiation* (Springer, 2008) 35-44.
"""
from __future__ import annotations

import hashlib
import threading
import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from photonix.core.backend import HAS_JAX
from photonix.core.types import Array

__all__ = [
    "SparseExposedSolver",
    "sparse_exposed_solve",
    "SingularSampleError",
    "PIVOT_RTOL",
]

#: A factorization whose smallest |U_ii| is below ``PIVOT_RTOL * max |U_ii|``
#: is treated as singular. For LU with partial pivoting a tiny pivot bounds
#: the smallest singular value of ``A`` from above, so the condition number
#: is then at least ~1e14 and double-precision results are not trustworthy.
PIVOT_RTOL = 1e-14


class SingularSampleError(ValueError):
    """Raised by :class:`SparseExposedSolver` for a singular batch sample.

    ``samples`` lists the flat batch indices that could not be solved.
    """

    def __init__(self, message: str, samples: list[int]):
        super().__init__(message)
        self.samples = samples


class _Factor:
    """A SuperLU factor of ``A`` or of ``A[:, perm]`` with the permutation undone."""

    __slots__ = ("lu", "perm")

    def __init__(self, lu: Any, perm: np.ndarray | None):
        self.lu, self.perm = lu, perm

    def solve(self, rhs: np.ndarray, trans: str = "N") -> np.ndarray:
        if self.perm is None:
            return self.lu.solve(rhs, trans=trans)
        if trans == "N":                    # (A P) y = b, x = P y  ->  x[perm] = y
            y = self.lu.solve(rhs)
            x = np.empty_like(y)
            x[self.perm] = y
            return x
        return self.lu.solve(np.asfortranarray(rhs[self.perm]), trans=trans)   # (A P)^T z = P^T c


class SparseExposedSolver:
    """Exposed-port solve of ``(I - S Gamma) X = S P_E`` for a fixed topology.

    The sparsity structure (CSC index arrays of ``A``, the scatter maps of the
    right-hand side and of the adjoint) is computed once; every call only
    fills values, factors with SuperLU and solves.

    Parameters
    ----------
    rows, cols
        ``(nnz,)`` output (row) and input (column) terminal index of every
        stored ``S`` entry. Pairs must be unique.
    partner
        ``(n,)`` connection partner of each terminal, ``-1`` if unconnected.
    exposed
        ``(m,)`` terminal index of each exposed port, in output order.
    n
        Number of terminals.
    ordering
        SuperLU column ordering. ``"auto"`` factors the first sample with
        both ``"NATURAL"`` (netlist order, often excellent for cascades and
        meshes) and ``"COLAMD"`` and keeps the one with less fill.

    Examples
    --------
    Two cascaded delays ``t1``, ``t2`` (terminals ``a.o1=0, a.o2=1, b.o1=2,
    b.o2=3``; ``a.o2`` joined to ``b.o1``):

    >>> import numpy as np
    >>> t1, t2 = np.exp(-0.3j), np.exp(-0.4j)
    >>> solver = SparseExposedSolver(rows=[1, 0, 3, 2], cols=[0, 1, 2, 3],
    ...                              partner=[-1, 2, 1, -1], exposed=[0, 3], n=4)
    >>> Y = solver.solve_numpy(np.array([t1, t1, t2, t2]))
    >>> bool(np.isclose(Y[1, 0], t1 * t2))        # Y[out, in]: o1 of a -> o2 of b
    True
    """

    def __init__(
        self,
        rows: Any,
        cols: Any,
        partner: Any,
        exposed: Any,
        n: int,
        *,
        ordering: Literal["auto", "COLAMD", "NATURAL", "MMD_AT_PLUS_A", "MMD_ATA"] = "auto",
    ) -> None:
        rows = np.asarray(rows, dtype=np.int64).ravel()
        cols = np.asarray(cols, dtype=np.int64).ravel()
        partner = np.asarray(partner, dtype=np.int64).ravel()
        exposed = np.asarray(exposed, dtype=np.int64).ravel()
        n = int(n)
        if rows.shape != cols.shape:
            raise ValueError("rows and cols must have the same length")
        if partner.shape != (n,):
            raise ValueError(f"partner must have length n={n}")
        if rows.size and (rows.min() < 0 or rows.max() >= n or cols.min() < 0 or cols.max() >= n):
            raise ValueError("entry indices must lie in [0, n)")
        if np.unique(rows * n + cols).size != rows.size:
            raise ValueError("(row, col) entry pairs must be unique")
        if exposed.size and (exposed.min() < 0 or exposed.max() >= n or np.unique(exposed).size != exposed.size):
            raise ValueError("exposed must be distinct terminal indices in [0, n)")
        connected = partner >= 0
        if np.any(partner[connected] >= n) or np.any(partner[partner[connected]] != np.nonzero(connected)[0]):
            raise ValueError("partner must be a symmetric pairing (partner[partner[j]] == j)")
        if np.any(connected[exposed]):
            raise ValueError("an exposed terminal cannot also be connected")
        self.n, self.m, self.nnz = n, int(exposed.size), int(rows.size)
        self.rows, self.cols, self.partner, self.exposed = rows, cols, partner, exposed
        self.ordering = ordering
        #: SuperLU ordering in use once the first sample has been factored.
        self.chosen_ordering: str | None = None
        self._perm_plan: tuple[Any, np.ndarray, np.ndarray, Any] | None = None

        # A = I - S Gamma: entry (r, c) of S lands at (r, gamma(c)) when c is connected.
        a_entries = np.nonzero(partner[cols] >= 0)[0]
        a_rows = rows[a_entries]
        a_cols = partner[cols[a_entries]]
        diag = np.arange(n, dtype=np.int64)
        key = np.concatenate([a_cols * n + a_rows, diag * n + diag])
        uniq, inverse = np.unique(key, return_inverse=True)
        ucols = uniq // n
        self._indices = (uniq % n).astype(np.int32)
        self._indptr = np.searchsorted(ucols, np.arange(n + 1)).astype(np.int32)
        self._nnz_a = int(uniq.size)
        self._slot_a = inverse[: a_entries.size]
        self._slot_diag = inverse[a_entries.size:]
        self._a_entries, self._a_rows, self._a_cols = a_entries, a_rows, a_cols

        # Right-hand side S P_E: entries whose input terminal is exposed.
        k_of = np.full(n, -1, dtype=np.int64)
        k_of[exposed] = np.arange(self.m)
        b_entries = np.nonzero(k_of[cols] >= 0)[0]
        self._b_entries = b_entries
        self._b_rows = rows[b_entries]
        self._b_cols = k_of[cols[b_entries]]
        self._jax_fn: Callable[[Array], Array] | None = None

        # Row-aligned layout of S Gamma for the JAX tangent: layer t holds the
        # t-th connected entry of every row (masked where a row has fewer), so
        # (S Gamma) x is a sum of row-aligned gathers and its transpose needs
        # no gather of the cotangent. ``None`` when padding would be wasteful
        # (a few very high-degree rows); a scatter is used then.
        degree = np.bincount(a_rows, minlength=n) if n else np.zeros(0, dtype=np.int64)
        d_max = int(degree.max()) if degree.size else 0
        self._sg_layers: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
        if d_max * n <= 2 * a_entries.size + n:
            order = np.argsort(a_rows, kind="stable")
            rank = np.arange(order.size) - np.searchsorted(a_rows[order], a_rows[order])
            layers = []
            for t in range(d_max):
                sel = order[rank == t]
                entry = np.zeros(n, dtype=np.int64)
                col = np.zeros(n, dtype=np.int64)
                mask = np.zeros(n, dtype=bool)
                entry[a_rows[sel]] = a_entries[sel]
                col[a_rows[sel]] = a_cols[sel]
                mask[a_rows[sel]] = True
                layers.append((entry, col, mask))
            self._sg_layers = layers

        # One batch of LU factors kept between the host callbacks of one
        # differentiated evaluation (primal solve, then tangent or transposed
        # solve with the same values): one factorization per sample.
        self._cache_lock = threading.Lock()
        self._factor_cache: tuple[tuple[Any, ...], list[_Factor | None]] | None = None

    # ------------------------------------------------------------------ #
    # NumPy kernels
    # ------------------------------------------------------------------ #
    def _matrix_data(self, values: np.ndarray) -> np.ndarray:
        data = np.zeros((values.shape[0], self._nnz_a), dtype=values.dtype)
        data[:, self._slot_diag] = 1.0
        data[:, self._slot_a] -= values[:, self._a_entries]
        return data

    def _factor(self, data_row: np.ndarray) -> _Factor | None:
        """SuperLU factor of one sample, or ``None`` if (numerically) singular.

        The column ordering depends only on the sparsity pattern, which every
        batch sample shares: it is chosen on the first factorization (least
        fill of ``NATURAL`` and ``COLAMD`` for ``ordering="auto"``) and a
        fill-reducing permutation is then applied once to the CSC structure,
        so later samples skip the ordering step (``permc_spec="NATURAL"``).
        """
        import scipy.sparse as sp
        from scipy.sparse.linalg import splu

        n = self.n
        if self._perm_plan is None:
            A = sp.csc_matrix((data_row, self._indices, self._indptr), shape=(n, n))
            specs = ("NATURAL", "COLAMD") if self.ordering == "auto" else (self.ordering,)
            best = None
            for spec in specs:
                try:
                    lu = splu(A, permc_spec=spec)
                except RuntimeError:  # "Factor is exactly singular"
                    continue
                if best is None or lu.nnz < best[1].nnz:
                    best = (spec, lu)
            if best is None:
                return None
            spec, lu = best
            self.chosen_ordering = spec
            if spec == "NATURAL":
                self._perm_plan = (None, self._indices, self._indptr, None)
            else:
                # A Pc = A[:, argsort(perm_c)] (SciPy's Pc convention).
                perm = np.argsort(lu.perm_c)
                starts, stops = self._indptr[perm], self._indptr[perm + 1]
                counts = (stops - starts).astype(np.int64)
                data_map = np.concatenate([np.arange(a, b) for a, b in zip(starts, stops, strict=True)]) \
                    if n else np.zeros(0, dtype=np.int64)
                indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
                self._perm_plan = (perm, self._indices[data_map], indptr, data_map)
            factor: _Factor | None = _Factor(lu, None)
        else:
            perm, indices, indptr, data_map = self._perm_plan
            data_p = data_row if data_map is None else data_row[data_map]
            A = sp.csc_matrix((data_p, indices, indptr), shape=(n, n))
            try:
                factor = _Factor(splu(A, permc_spec="NATURAL"), perm)
            except RuntimeError:
                return None
        assert factor is not None
        u_diag = np.abs(factor.lu.U.diagonal())
        if not np.all(np.isfinite(u_diag)) or u_diag.min() <= PIVOT_RTOL * max(u_diag.max(), 1e-300):
            return None
        return factor

    def _rhs(self, values_row: np.ndarray) -> np.ndarray:
        B = np.zeros((self.n, self.m), dtype=values_row.dtype, order="F")
        B[self._b_rows, self._b_cols] = values_row[self._b_entries]
        return B

    def _flat(self, values: Any) -> tuple[np.ndarray, tuple[int, ...]]:
        v = np.asarray(values)
        if v.shape[-1:] != (self.nnz,):
            raise ValueError(f"values must have trailing axis nnz={self.nnz}, got shape {v.shape}")
        dtype = np.result_type(v.dtype, np.complex64)
        batch = v.shape[:-1]
        return np.ascontiguousarray(v.reshape(-1, self.nnz), dtype=dtype), batch

    def solve_numpy(self, values: Any, *, on_singular: Literal["raise", "nan"] = "raise") -> np.ndarray:
        """Exposed block ``Y[..., out, in]`` for ``values`` shaped ``(..., nnz)``.

        ``on_singular="raise"`` raises :class:`SingularSampleError`; ``"nan"``
        fills the failing samples with NaN (used inside JAX callbacks, where
        exceptions are not part of the callback contract).
        """
        v, batch = self._flat(values)
        out = np.empty((v.shape[0], self.m, self.m), dtype=v.dtype)
        if self.m == 0:
            return out.reshape(*batch, 0, 0)
        data = self._matrix_data(v)
        bad: list[int] = []
        for s in range(v.shape[0]):
            lu = self._factor(data[s])
            X = None if lu is None else lu.solve(self._rhs(v[s]))
            if X is None or not np.all(np.isfinite(X[self.exposed])):
                bad.append(s)
                out[s] = np.nan
                continue
            out[s] = X[self.exposed]
        if bad:
            self._report_singular(bad, on_singular)
        return out.reshape(*batch, self.m, self.m)

    #: Upper bound (bytes of L + U) on the factors kept for reuse between callbacks.
    FACTOR_CACHE_BYTES = 256 * 2**20

    def _factors(self, v: np.ndarray, *, cache: bool) -> list[_Factor | None]:
        """Factors of every flat sample of ``v`` (``(S, nnz)``), reusing the last batch if identical."""
        key: tuple[Any, ...] | None = None
        if cache:
            key = (v.shape, v.dtype.str, hashlib.blake2b(v.tobytes(), digest_size=16).digest())
            with self._cache_lock:
                hit = self._factor_cache
            if hit is not None and hit[0] == key:
                return hit[1]
        data = self._matrix_data(v)
        factors = [self._factor(data[s]) for s in range(v.shape[0])]
        if key is not None:
            size = sum(16 * int(f.lu.nnz) for f in factors if f is not None)
            with self._cache_lock:
                self._factor_cache = (key, factors) if size <= self.FACTOR_CACHE_BYTES else None
        return factors

    def solve_full_numpy(
        self, values: Any, rhs: Any = None, *, trans: Literal["N", "T"] = "N",
        on_singular: Literal["raise", "nan"] = "raise", cache: bool = False,
    ) -> np.ndarray:
        """Solve ``A X = rhs`` (``trans="N"``) or ``A^T X = rhs`` (``"T"``) for every sample.

        ``values`` is ``(..., nnz)`` and ``rhs`` ``(..., n, k)`` (batch axes
        broadcast), by default ``S P_E`` (``k = m``, the circuit system);
        returns the full ``(..., n, k)`` solution. Used by the differentiable
        JAX path, which passes ``cache=True`` so that consecutive calls with
        the same ``values`` share one factorization per sample; failing
        samples are NaN with ``on_singular="nan"``.
        """
        v_in = np.asarray(values)
        r_in = None if rhs is None else np.asarray(rhs)
        batch = v_in.shape[:-1] if r_in is None else np.broadcast_shapes(v_in.shape[:-1], r_in.shape[:-2])
        v, _ = self._flat(np.broadcast_to(v_in, (*batch, self.nnz)))
        k = self.m if r_in is None else r_in.shape[-1]
        r = None if r_in is None else np.broadcast_to(r_in, (*batch, self.n, k)).reshape(-1, self.n, k)
        dtype = v.dtype if r is None else np.result_type(v.dtype, r.dtype)
        out = np.empty((v.shape[0], self.n, k), dtype=dtype)
        if k == 0 or self.n == 0:
            return out.reshape(*batch, self.n, k)
        bad: list[int] = []
        for s, lu in enumerate(self._factors(v, cache=cache)):
            b = self._rhs(v[s]) if r is None else np.asfortranarray(r[s], dtype=dtype)
            X = None if lu is None else lu.solve(b, trans=trans)
            if X is None or not np.all(np.isfinite(X)):
                bad.append(s)
                out[s] = np.nan
                continue
            out[s] = X
        if bad:
            self._report_singular(bad, on_singular)
        return out.reshape(*batch, self.n, k)

    def vjp_numpy(
        self, values: Any, y_bar: Any, *, on_singular: Literal["raise", "nan"] = "raise"
    ) -> np.ndarray:
        """Cotangent of ``values`` for the output cotangent ``y_bar`` (``(..., m, m)``).

        Implements the transpose-Jacobian product described in the module
        docstring; no ``n x n`` matrix is formed.
        """
        v, batch = self._flat(values)
        yb = np.asarray(y_bar).reshape(-1, self.m, self.m).astype(v.dtype, copy=False)
        if yb.shape[0] != v.shape[0]:
            yb = np.broadcast_to(yb, (v.shape[0], self.m, self.m))
        out = np.zeros_like(v)
        if self.m == 0:
            return out.reshape(*batch, self.nnz)
        data = self._matrix_data(v)
        bad: list[int] = []
        for s in range(v.shape[0]):
            lu = self._factor(data[s])
            if lu is None:
                bad.append(s)
                out[s] = np.nan
                continue
            X = lu.solve(self._rhs(v[s]))
            Xbar = np.zeros((self.n, self.m), dtype=v.dtype, order="F")
            Xbar[self.exposed] = yb[s]
            Z = lu.solve(Xbar, trans="T")
            g = out[s]
            g[self._b_entries] += Z[self._b_rows, self._b_cols]
            g[self._a_entries] += np.einsum("ij,ij->i", Z[self._a_rows], X[self._a_cols])
        if bad:
            self._report_singular(bad, on_singular)
        return out.reshape(*batch, self.nnz)

    @staticmethod
    def _report_singular(bad: list[int], on_singular: str) -> None:
        msg = (
            f"I - S Gamma is singular (or has a relative pivot below {PIVOT_RTOL:g}) at flat batch "
            f"sample(s) {bad[:10]}{' ...' if len(bad) > 10 else ''}: the circuit supports a "
            "self-sustained, unexcited resonance. Typical cause: a lossless loop decoupled from "
            "every exposed port (e.g. a ring with coupling=0 exactly on resonance, or a zero-phase "
            "zero-loss loop)."
        )
        if on_singular == "raise":
            raise SingularSampleError(msg, bad)
        warnings.warn(msg + " Those samples are returned as NaN.", RuntimeWarning, stacklevel=3)

    # ------------------------------------------------------------------ #
    # Backend-dispatching, differentiable entry point
    # ------------------------------------------------------------------ #
    def __call__(self, values: Array) -> Array:
        """Differentiable exposed block for ``values`` shaped ``(..., nnz)``.

        NumPy backend: :meth:`solve_numpy`. JAX backend: a host callback
        with a ``custom_jvp`` rule built on :func:`jax.lax.custom_linear_solve`
        (module docs), so every JAX transformation works, forward and reverse
        mode included; singular samples become NaN with a ``RuntimeWarning``
        because exceptions cannot cross a JAX callback reliably.
        """
        if not HAS_JAX:
            return self.solve_numpy(values)
        if self._jax_fn is None:
            self._jax_fn = self._build_jax()
        return self._jax_fn(values)

    def _build_jax(self) -> Callable[[Array], Array]:
        import inspect

        import jax
        import jax.numpy as jnp

        n, m, nnz = self.n, self.m, self.nnz
        exposed = self.exposed
        a_entries, a_rows, a_cols = self._a_entries, self._a_rows, self._a_cols
        b_entries, b_rows, b_cols = self._b_entries, self._b_rows, self._b_cols
        layers = self._sg_layers
        # ``vmap_method`` replaced ``vectorized`` in JAX 0.4.34; both mean
        # "call once with every batch axis present" here.
        batching: dict[str, Any] = (
            {"vmap_method": "broadcast_all"} if "vmap_method" in inspect.signature(jax.pure_callback).parameters
            else {"vectorized": True}
        )

        def s_gamma(v, x):
            """``(S Gamma) x`` for ``x`` shaped ``(..., n, k)``; bilinear in ``v`` and ``x``."""
            if layers is None:
                contrib = v[..., a_entries, None] * x[..., a_cols, :]
                out = jnp.zeros(contrib.shape[:-2] + x.shape[-2:], dtype=contrib.dtype)
                return out.at[..., a_rows, :].add(contrib)
            total = None
            for entry, col, mask in layers:
                term = jnp.where(mask, v[..., entry], 0)[..., None] * x[..., col, :]
                total = term if total is None else total + term
            if total is None:
                return jnp.zeros(jnp.broadcast_shapes(v.shape[:-1], x.shape[:-2]) + x.shape[-2:], dtype=x.dtype)
            return total

        def rhs(v):
            """``S P_E`` shaped ``(..., n, m)``; linear in ``v``."""
            return jnp.zeros((*v.shape[:-1], n, m), dtype=v.dtype).at[..., b_rows, b_cols].add(v[..., b_entries])

        def host_solve(v, r, trans):
            shape = jax.ShapeDtypeStruct(r.shape, r.dtype)
            return jax.pure_callback(
                lambda vv, rr: self.solve_full_numpy(vv, rr, trans=trans, on_singular="nan", cache=True).astype(
                    rr.dtype, copy=False), shape, v, r, **batching)

        def apply_inverse(v, b):
            """``A(v)^{-1} b`` as a :func:`jax.lax.custom_linear_solve`.

            The solve and the transposed solve are SuperLU host callbacks;
            JAX differentiates through ``A(v) = I - S Gamma`` (the ``matvec``)
            by the implicit-function rule and transposes the solve for reverse
            mode, so ``jvp``, ``vjp``, ``vmap`` and their compositions all work.
            """
            return jax.lax.custom_linear_solve(
                lambda x: x - s_gamma(v, x), b,
                lambda _matvec, r: host_solve(v, r, "N"), lambda _vecmat, r: host_solve(v, r, "T"))

        @jax.custom_jvp
        def full(v):
            """Full solution ``X = A^{-1} S P_E``, ``(..., n, m)``; the host builds ``S P_E``."""
            shape = jax.ShapeDtypeStruct((*v.shape[:-1], n, m), v.dtype)
            return jax.pure_callback(
                lambda vv: self.solve_full_numpy(vv, on_singular="nan", cache=True).astype(v.dtype, copy=False),
                shape, v, **batching)

        @full.defjvp
        def full_jvp(primals, tangents):
            # dX = A^{-1} (dS P_E + dS Gamma X): linear in dv and transposable;
            # X and A(v) stay differentiable for higher-order derivatives.
            (v,), (dv,) = primals, tangents
            X = full(v)
            return X, apply_inverse(v, rhs(dv) + s_gamma(dv, X))

        @jax.custom_jvp
        def solve(values):
            # Primal only: the host returns just the m x m exposed block.
            shape = jax.ShapeDtypeStruct((*values.shape[:-1], m, m), values.dtype)
            return jax.pure_callback(lambda vv: self.solve_numpy(vv, on_singular="nan"), shape, values, **batching)

        @solve.defjvp
        def solve_jvp(primals, tangents):
            X, dX = jax.jvp(full, primals, tangents)
            return X[..., exposed, :], dX[..., exposed, :]

        def entry(values):
            values = jnp.asarray(values)
            if values.shape[-1:] != (nnz,):
                raise ValueError(f"values must have trailing axis nnz={nnz}, got shape {values.shape}")
            if not jnp.issubdtype(values.dtype, jnp.complexfloating):
                values = values.astype(jnp.result_type(values.dtype, jnp.complex64))
            return solve(values)

        return entry


def sparse_exposed_solve(rows: Any, cols: Any, values: Array, partner: Any, exposed: Any, n: int) -> Array:
    """One-shot functional form of :class:`SparseExposedSolver`.

    Returns ``Y`` shaped ``(..., m, m)`` with ``Y[..., i, k]`` the wave leaving
    exposed terminal ``exposed[i]`` for unit excitation at ``exposed[k]``.
    Differentiable (forward and reverse mode) under JAX. Build a
    :class:`SparseExposedSolver` once instead when solving one topology
    repeatedly.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.exp(-0.5j)
    >>> Y = sparse_exposed_solve([1, 0], [0, 1], np.array([t, t]), [-1, -1], [0, 1], 2)
    >>> bool(np.allclose(np.asarray(Y), [[0, t], [t, 0]]))
    True
    """
    return SparseExposedSolver(rows, cols, partner, exposed, n)(values)
