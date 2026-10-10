"""Exact rasterization of polygonal cross-sections onto tensor-product grids.

Algorithm
---------
:func:`XSection.rasterize` computes, for every cell of a :class:`TensorGrid`,
the **exact** area of each region that is visible under the painter's rule
(higher ``order`` -- or, at equal order, later in the list -- paints over lower),
together with the exact length fraction of each cell *face* covered by each
region.  It is a trapezoidal sweep (scanline polygon rasterization):

1. The window is cut into horizontal slabs at every grid line, every polygon
   vertex and every crossing of two polygon edges.  Inside a slab no vertex
   occurs and no edges cross, so the edges spanning it are straight segments
   in a fixed left-to-right order.
2. Walking the sorted edges, nonzero winding numbers decide which regions
   contain each trapezoid between consecutive edges; the highest-priority
   one owns it.  Only edges where the owner changes are kept.
3. The integral ``int (clip(x_e(y), X_i, X_i+1) - X_i) dy`` of each kept edge
   over each cell column is exact by the trapezoidal rule once the edge is
   split where it crosses ``X_i`` and ``X_i+1`` (the integrand is piecewise
   linear).  Differences of these integrals between consecutive edges are
   trapezoid-cell intersection areas, i.e. Green's theorem cell by cell.

Overlapping, abutting and coincident edges (a core on a BOX, a heater over its
waveguide) need no polygon boolean operations, and the result reproduces
:meth:`photonix.pdk.LayerStack.cross_section` -- also exact for rectangles --
to rounding.  (A per-cell product rule ``f_i * prod(1 - f_j)`` would *not*: it
is wrong wherever two abutting shapes share a cut cell.)

The topology (slabs, sort orders, owners) is computed in NumPy from concrete
coordinates; the arithmetic is then repeated in ``jax.numpy`` when any polygon
holds JAX values, so areas -- and everything derived from them -- are
differentiable with respect to vertex coordinates under ``jax.grad`` (the
topology is piecewise constant; ``jax.jit`` is not supported).

Interface normals and tensor smoothing
--------------------------------------
For anisotropic subpixel smoothing (Farjadpour et al., Opt. Lett. 31, 2972,
2006; Kottke, Farjadpour & Johnson, Phys. Rev. E 77, 036611, 2008) every cut
cell needs an interface normal.  It is the direction of the cell-integrated
gradient of the property ``q`` being smoothed, computed exactly from the face
coverages by the divergence theorem::

    int_cell grad q dA = sum_r q_r [ (Fx_r(right) - Fx_r(left)) dy ,  (Fy_r(top) - Fy_r(bottom)) dx ]

For two materials this is exactly Kottke's ``int grad chi``; for more it is
weighted by the property jump across each interface.  The smoothed tensor uses
Kottke's ``tau`` average in the local (normal, tangent, z) frame, which reduces
for isotropic media to ``P <eps^-1>^-1 + (1 - P) <eps>`` with ``P = n n^T`` and
is the exact homogenization of a laminate (series across, parallel along).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from .properties import resolve_index, resolve_property
from .shapes import BAND_HALF_WIDTH, Region, _check_material, _concrete, _is_jax_array

__all__ = [
    "TensorGrid",
    "uniform_grid",
    "graded_grid",
    "Coverage",
    "XSection",
]

Smoothing = Literal["arithmetic", "tensor"]
Average = Literal["arithmetic", "harmonic", "tensor"]


# --------------------------------------------------------------------------- #
# Grids
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class TensorGrid:
    """Tensor-product (possibly graded) grid given by its cell edges (µm).

    Cell ``(j, i)`` is ``[x_edges[i], x_edges[i+1]] x [y_edges[j], y_edges[j+1]]``;
    arrays over the grid have shape ``(ny, nx)`` (rows = ``y``), the convention of
    :class:`photonix.em.CrossSection`.

    Examples
    --------
    >>> g = TensorGrid([0.0, 0.5, 1.5], [0.0, 1.0])
    >>> g.shape, g.xc.tolist(), g.areas.tolist()
    ((1, 2), [0.25, 1.0], [[0.5, 1.0]])
    """

    x_edges: Any
    y_edges: Any

    def __post_init__(self) -> None:
        for name in ("x_edges", "y_edges"):
            e = np.array(getattr(self, name), dtype=float)
            if e.ndim != 1 or e.size < 2:
                raise ValueError(f"{name} must be a 1-D array with at least two edges")
            if not np.all(np.isfinite(e)):
                raise ValueError(f"{name} must be finite")
            if np.any(np.diff(e) <= 0):
                raise ValueError(f"{name} must be strictly increasing")
            e.setflags(write=False)
            object.__setattr__(self, name, e)

    @property
    def nx(self) -> int:
        return int(self.x_edges.size - 1)

    @property
    def ny(self) -> int:
        return int(self.y_edges.size - 1)

    @property
    def shape(self) -> tuple[int, int]:
        """``(ny, nx)``."""
        return (self.ny, self.nx)

    @property
    def xc(self) -> np.ndarray:
        """Cell-centre ``x`` coordinates, shape ``(nx,)``."""
        return 0.5 * (self.x_edges[1:] + self.x_edges[:-1])

    @property
    def yc(self) -> np.ndarray:
        """Cell-centre ``y`` coordinates, shape ``(ny,)``."""
        return 0.5 * (self.y_edges[1:] + self.y_edges[:-1])

    @property
    def dx(self) -> np.ndarray:
        """Cell widths, shape ``(nx,)``."""
        return np.diff(self.x_edges)

    @property
    def dy(self) -> np.ndarray:
        """Cell heights, shape ``(ny,)``."""
        return np.diff(self.y_edges)

    @property
    def areas(self) -> np.ndarray:
        """Cell areas, shape ``(ny, nx)``."""
        return self.dy[:, None] * self.dx[None, :]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` of the window."""
        return (float(self.x_edges[0]), float(self.x_edges[-1]), float(self.y_edges[0]), float(self.y_edges[-1]))

    def is_uniform(self, rtol: float = 1e-9) -> bool:
        """``True`` if all cells share one width and one height (to ``rtol``)."""
        dx, dy = self.dx, self.dy
        return bool(np.allclose(dx, dx[0], rtol=rtol, atol=0) and np.allclose(dy, dy[0], rtol=rtol, atol=0))

    def shifted(self, dx: float = 0.0, dy: float = 0.0) -> TensorGrid:
        """The same grid translated by ``(dx, dy)`` (e.g. to a Yee sub-lattice)."""
        return TensorGrid(self.x_edges + dx, self.y_edges + dy)


def _centred_edges(n: int, h: float, centre: float) -> np.ndarray:
    c = (np.arange(n, dtype=float) - 0.5 * (n - 1)) * h + centre
    return np.concatenate([c - 0.5 * h, [c[-1] + 0.5 * h]])


def uniform_grid(
    bbox: Sequence[float],
    resolution: float = 40,
    margin: float = 1.5,
    *,
    min_cells: int = 8,
) -> TensorGrid:
    """Uniform cell-centred grid around ``bbox = (x0, x1, y0, y1)``.

    The construction is identical to :func:`photonix.em.rectangular_waveguide`
    and :meth:`photonix.pdk.LayerStack.cross_section`: spacing exactly
    ``1 / resolution``, ``ceil((extent + 2 margin) * resolution)`` cells per
    axis (never fewer than ``min_cells``) centred on the box, so the requested
    margin is never shortened.

    Examples
    --------
    >>> g = uniform_grid((-0.25, 0.25, -0.11, 0.11), resolution=20, margin=1.5)
    >>> g.shape, round(float(g.dx[0]), 12)
    ((65, 70), 0.05)
    """
    x0, x1, y0, y1 = (float(v) for v in bbox)
    if not (np.isfinite([x0, x1, y0, y1]).all() and x1 >= x0 and y1 >= y0):
        raise ValueError("bbox must be finite (x0, x1, y0, y1) with x1 >= x0 and y1 >= y0")
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be a positive finite number of cells per micrometre")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("margin must be non-negative and finite")
    h = 1.0 / float(resolution)
    nx = max(int(np.ceil((x1 - x0 + 2 * margin) * resolution)), int(min_cells))
    ny = max(int(np.ceil((y1 - y0 + 2 * margin) * resolution)), int(min_cells))
    return TensorGrid(_centred_edges(nx, h, 0.5 * (x0 + x1)), _centred_edges(ny, h, 0.5 * (y0 + y1)))


def _graded_axis(lo: float, hi: float, zones: list[tuple[float, float, float]], h_max: float,
                 ratio: float) -> np.ndarray:
    def target(x: float) -> float:
        t = h_max
        for a, b, h in zones:
            d = max(a - x, 0.0, x - b)
            t = min(t, h + (ratio - 1.0) * d)
        return t

    breaks = sorted({lo, hi, *(min(max(v, lo), hi) for a, b, _ in zones for v in (a, b))})
    edges = [lo]
    for p, q in zip(breaks[:-1], breaks[1:], strict=True):
        if q - p <= 1e-12 * max(1.0, abs(p), abs(q)):
            continue
        pos = [p]
        while pos[-1] < q - 1e-12 * (q - p):
            x = pos[-1]
            h = target(x)
            h = min(h, target(x + h))
            pos.append(x + h)
        arr = np.asarray(pos)
        arr = p + (arr - p) * ((q - p) / (arr[-1] - p))  # shrink (never stretch) to land on q
        edges.extend(arr[1:].tolist())
    out = np.asarray(edges)
    out[-1] = hi
    return out


def graded_grid(
    bbox: Sequence[float],
    refine: Sequence[Sequence[float]] = (),
    h_max: float = 0.2,
    ratio: float = 1.2,
) -> TensorGrid:
    """Graded tensor grid over the window ``bbox = (x0, x1, y0, y1)``.

    ``refine`` lists boxes ``(x0, x1, y0, y1, h)``: inside each box's ``x``
    (``y``) range the ``x`` (``y``) spacing is at most ``h``; away from it the
    target spacing grows linearly with distance at rate ``ratio - 1`` (so
    neighbouring cells differ by about ``ratio``) up to ``h_max``.  Box edges
    are grid lines.  Intended for the conduction/heat solvers that accept
    graded grids; the current FDE mode solvers need :func:`uniform_grid`.

    Examples
    --------
    >>> g = graded_grid((-5, 5, -3, 3), refine=[(-0.3, 0.3, -0.2, 0.2, 0.02)], h_max=0.5)
    >>> bool(g.dx.min() <= 0.02 + 1e-12 and g.dx.max() <= 0.5 + 1e-12 and not g.is_uniform())
    True
    """
    x0, x1, y0, y1 = (float(v) for v in bbox)
    if not (np.isfinite([x0, x1, y0, y1]).all() and x1 > x0 and y1 > y0):
        raise ValueError("bbox must be finite (x0, x1, y0, y1) with x1 > x0 and y1 > y0")
    if not (np.isfinite(h_max) and h_max > 0):
        raise ValueError("h_max must be positive")
    if not (np.isfinite(ratio) and ratio >= 1.0):
        raise ValueError("ratio must be >= 1")
    zx, zy = [], []
    for box in refine:
        if len(box) != 5:
            raise ValueError("each refine entry must be (x0, x1, y0, y1, h)")
        a, b, c, d, h = (float(v) for v in box)
        if not (h > 0 and b >= a and d >= c):
            raise ValueError("refine boxes need x1 >= x0, y1 >= y0 and h > 0")
        zx.append((a, b, min(h, h_max)))
        zy.append((c, d, min(h, h_max)))
    return TensorGrid(_graded_axis(x0, x1, zx, h_max, ratio), _graded_axis(y0, y1, zy, h_max, ratio))


# --------------------------------------------------------------------------- #
# Exact sweep
# --------------------------------------------------------------------------- #
@dataclass
class _Topology:
    """Concrete (NumPy) sweep structure shared by the NumPy and JAX arithmetic."""

    e_lo: np.ndarray          # vertex index of each kept edge's lower end
    e_hi: np.ndarray          # vertex index of its upper end
    pairs: np.ndarray         # (P, 2) edge-index pairs whose crossing is a slab boundary
    cand_kind: np.ndarray     # 0 grid line, 1 vertex, 2 crossing
    cand_ref: np.ndarray      # index into the grid lines / vertices / pairs
    bnd: np.ndarray           # candidate index of every slab boundary (sorted)
    slab_row: np.ndarray      # grid row of each slab
    first: np.ndarray         # slab is the lowest in its row
    last: np.ndarray          # slab is the highest in its row
    p_slab: np.ndarray        # per kept edge-in-slab piece: slab index
    p_edge: np.ndarray        # ... edge index
    p_left: np.ndarray        # ... owner of the trapezoid on its left (R = background)
    p_right: np.ndarray       # ... owner on its right


def _topology(conc: list[np.ndarray], ranks: np.ndarray, grid: TensorGrid) -> _Topology:
    R = len(conc)
    Y = grid.y_edges
    Y0, Y1 = float(Y[0]), float(Y[-1])
    sizes = [len(c) for c in conc]
    base = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int) if conc else np.zeros(0, int)
    V = np.concatenate(conc) if conc else np.zeros((0, 2))
    a = np.concatenate([b + np.arange(n) for b, n in zip(base, sizes, strict=True)]) if conc else np.zeros(0, int)
    b = np.concatenate([b + (np.arange(n) + 1) % n for b, n in zip(base, sizes, strict=True)]) \
        if conc else np.zeros(0, int)
    reg = np.concatenate([np.full(n, r) for r, n in enumerate(sizes)]) if conc else np.zeros(0, int)
    ya, yb = V[a, 1] if len(a) else np.zeros(0), V[b, 1] if len(b) else np.zeros(0)
    keep = (ya != yb) & (np.maximum(ya, yb) > Y0) & (np.minimum(ya, yb) < Y1)
    up = ya < yb
    e_lo = np.where(up, a, b)[keep]
    e_hi = np.where(up, b, a)[keep]
    e_dir = np.where(up, 1, -1)[keep]
    e_reg = reg[keep]
    ylo, yhi = V[e_lo, 1], V[e_hi, 1]
    xlo, xhi = V[e_lo, 0], V[e_hi, 0]
    slope = (xhi - xlo) / (yhi - ylo)
    E = len(e_lo)

    # Edge crossings (strictly inside both edges' y-ranges and the window).
    pairs_list = []
    ys_list = []
    block = 256
    for p0 in range(0, E, block):
        P = np.arange(p0, min(E, p0 + block))
        lo_max = np.maximum(ylo[P, None], ylo[None, :])
        hi_min = np.minimum(yhi[P, None], yhi[None, :])
        dk = slope[P, None] - slope[None, :]
        ok = (np.arange(E)[None, :] > P[:, None]) & (lo_max < hi_min) & (dk != 0)
        if not ok.any():
            continue
        num = xlo[None, :] - xlo[P, None] + ylo[P, None] * slope[P, None] - ylo[None, :] * slope[None, :]
        ys = num / np.where(ok, dk, 1.0)
        ok &= (ys > lo_max) & (ys < hi_min) & (ys > Y0) & (ys < Y1)
        pi, qi = np.nonzero(ok)
        if pi.size:
            pairs_list.append(np.stack([P[pi], qi], axis=1))
            ys_list.append(ys[pi, qi])
    pairs = np.concatenate(pairs_list) if pairs_list else np.zeros((0, 2), int)
    ys = np.concatenate(ys_list) if ys_list else np.zeros(0)

    vin = np.flatnonzero((V[:, 1] > Y0) & (V[:, 1] < Y1)) if len(V) else np.zeros(0, int)
    cand_val = np.concatenate([Y, V[vin, 1] if len(vin) else np.zeros(0), ys])
    cand_kind = np.concatenate([np.zeros(Y.size, int), np.ones(vin.size, int), np.full(ys.size, 2)])
    cand_ref = np.concatenate([np.arange(Y.size), vin, np.arange(ys.size)])
    order = np.lexsort((cand_kind, cand_val))
    vals = cand_val[order]
    tol = 1e-12 * max(Y1 - Y0, abs(Y0), abs(Y1))
    new_group = np.concatenate([[True], np.diff(vals) > tol])
    group = np.cumsum(new_group) - 1
    # Representative: the lowest kind (grid line > vertex > crossing) of each group.
    G = int(group[-1]) + 1
    rep = np.full(G, -1)
    best_kind = np.full(G, 9)
    for idx in range(order.size):  # small (grid lines + vertices + crossings)
        g = group[idx]
        k = cand_kind[order[idx]]
        if k < best_kind[g]:
            best_kind[g] = k
            rep[g] = order[idx]
    bvals = cand_val[rep]
    mids = 0.5 * (bvals[:-1] + bvals[1:])
    slab_row = np.clip(np.searchsorted(Y, mids, side="right") - 1, 0, grid.ny - 1)
    S = mids.size
    first = np.ones(S, bool)
    last = np.ones(S, bool)
    first[1:] = slab_row[1:] != slab_row[:-1]
    last[:-1] = slab_row[:-1] != slab_row[1:]

    p_slab, p_edge, p_left, p_right = [], [], [], []
    BG = R
    if E:
        for s in range(S):
            m = mids[s]
            act = np.flatnonzero((ylo <= m) & (yhi >= m))
            if act.size == 0:
                continue
            xm = xlo[act] + (m - ylo[act]) * slope[act]
            act = act[np.argsort(xm, kind="stable")]
            K = act.size
            wind = np.zeros((K, R))
            wind[np.arange(K), e_reg[act]] = -e_dir[act]
            wind = np.cumsum(wind, axis=0)
            score = np.where(wind != 0, ranks[None, :], -1)
            owner_after = np.where(score.max(axis=1) >= 0, np.argmax(score, axis=1), BG)
            owner_before = np.concatenate([[BG], owner_after[:-1]])
            sel = owner_before != owner_after
            if sel.any():
                n_sel = int(sel.sum())
                p_slab.append(np.full(n_sel, s))
                p_edge.append(act[sel])
                p_left.append(owner_before[sel])
                p_right.append(owner_after[sel])

    def cat(parts):
        return np.concatenate(parts).astype(int) if parts else np.zeros(0, int)

    return _Topology(e_lo=e_lo, e_hi=e_hi, pairs=pairs, cand_kind=cand_kind, cand_ref=cand_ref, bnd=rep,
                     slab_row=slab_row, first=first, last=last, p_slab=cat(p_slab), p_edge=cat(p_edge),
                     p_left=cat(p_left), p_right=cat(p_right))


def _scatter_add(xpm: Any, target: Any, index: tuple[Any, Any], values: Any) -> Any:
    if xpm is np:
        np.add.at(target, index, values)
        return target
    return target.at[index].add(values)


def _phase2(V: Any, Y: Any, X: Any, dy: Any, area: Any, fx: Any, fyb: Any, fyt: Any, e_lo: Any, e_hi: Any,
            vin: Any, pairs: Any, bnd: Any, s: Any, e: Any, L: Any, Rt: Any, row: Any, wfirst: Any, wlast: Any,
            *, xpm: Any) -> tuple[Any, Any, Any]:
    """Arithmetic of the sweep for a fixed topology (traceable and jit-able under JAX)."""
    lo_v, hi_v = V[e_lo], V[e_hi]
    slope = (hi_v[:, 0] - lo_v[:, 0]) / (hi_v[:, 1] - lo_v[:, 1])
    # Slab-boundary candidates (grid lines, vertices, edge crossings) in the traced arithmetic.
    if pairs.shape[0]:
        p, q = pairs[:, 0], pairs[:, 1]
        ys = (lo_v[q, 0] - lo_v[p, 0] + lo_v[p, 1] * slope[p] - lo_v[q, 1] * slope[q]) / (slope[p] - slope[q])
    else:
        ys = xpm.zeros(0)
    cand = xpm.concatenate([Y, V[vin, 1], ys])
    bvals = cand[bnd]
    lo, hi = bvals[:-1][s], bvals[1:][s]
    h = (hi - lo)[:, None]
    xb = (lo_v[e, 0] + (lo - lo_v[e, 1]) * slope[e])[:, None]
    xt = (lo_v[e, 0] + (hi - lo_v[e, 1]) * slope[e])[:, None]
    X0, X1 = X[None, :-1], X[None, 1:]
    Wc = X1 - X0
    D = xt - xb
    Ds = xpm.where(D == 0, 1.0, D)

    def g(x):
        return xpm.clip(x, X0, X1) - X0

    uA, uB = (X0 - xb) / Ds, (X1 - xb) / Ds
    u1 = xpm.clip(xpm.minimum(uA, uB), 0.0, 1.0)
    u2 = xpm.clip(xpm.maximum(uA, uB), 0.0, 1.0)
    g0, g1, g2, g3 = g(xb), g(xb + u1 * D), g(xb + u2 * D), g(xt)
    C = h * 0.5 * ((g0 + g1) * u1 + (g1 + g2) * (u2 - u1) + (g2 + g3) * (1.0 - u2))
    if xpm is not np:
        # The cell area is C1 in the vertex positions even where an edge end
        # touches a column line, but the breakpoint formula's clip/min ties
        # would split its derivative between neighbouring cells there.  Keep
        # the exact value and take the gradient from the tie-free
        # antiderivative form (G(xt) - G(xb)) / D, G' = g (straight-through).
        import jax

        def G(x):
            c = xpm.clip(x, X0, X1)
            return 0.5 * (c - X0) ** 2 + Wc * xpm.maximum(x - X1, 0.0)

        steep = xpm.abs(D) > 1e-3 * Wc
        Dz = xpm.where(steep, D, 1.0)
        smooth = h * xpm.where(steep, (G(xt) - G(xb)) / Dz, g(0.5 * (xb + xt)))
        C = smooth + jax.lax.stop_gradient(C - smooth)
    # Fraction of the slab height over which the edge lies left of each vertical line X.
    Xall = X[None, :]
    u = (Xall - xb) / Ds
    step = xpm.where(xb < Xall, 1.0, xpm.where(xb == Xall, 0.5, 0.0))
    H = xpm.where(D > 0, xpm.clip(u, 0.0, 1.0), xpm.where(D < 0, 1.0 - xpm.clip(u, 0.0, 1.0), step))
    area = _scatter_add(xpm, area, (L, row), C)
    area = _scatter_add(xpm, area, (Rt, row), -C)
    fx = _scatter_add(xpm, fx, (L, row), -H * h)
    fx = _scatter_add(xpm, fx, (Rt, row), H * h)
    Gb = g0 * wfirst[:, None]
    fyb = _scatter_add(xpm, fyb, (L, row), Gb)
    fyb = _scatter_add(xpm, fyb, (Rt, row), -Gb)
    Gt = g3 * wlast[:, None]
    fyt = _scatter_add(xpm, fyt, (L, row), Gt)
    fyt = _scatter_add(xpm, fyt, (Rt, row), -Gt)
    return _finish(xpm, area, fx, fyb, fyt, Wc[0], dy)


def _finish(xpm: Any, area: Any, fx: Any, fyb: Any, fyt: Any, W: Any, dy: Any) -> tuple[Any, Any, Any]:
    frac = area / (dy[:, None] * W[None, :])[None]
    face_x = fx / dy[None, :, None]
    Wn = W[None, None, :]
    inner = 0.5 * (fyt[:, :-1] + fyb[:, 1:]) / Wn
    face_y = xpm.concatenate([fyb[:, :1] / Wn, inner, fyt[:, -1:] / Wn], axis=1)
    return frac, face_x, face_y


_PHASE2_JIT: Any = None


def _phase2_jax() -> Any:
    """``_phase2`` jitted for jax.numpy (compiled once per problem shape)."""
    global _PHASE2_JIT
    if _PHASE2_JIT is None:
        from functools import partial

        import jax
        import jax.numpy as jnp

        _PHASE2_JIT = jax.jit(partial(_phase2, xpm=jnp))
    return _PHASE2_JIT


def _sweep(polys: list[Any], ranks: np.ndarray, grid: TensorGrid) -> tuple[Any, Any, Any]:
    """Exact visible area fractions and face coverages, shapes (R+1, ny, nx) etc."""
    use_jax = any(_is_jax_array(p) for p in polys)
    conc = [_concrete(p) for p in polys]
    topo = _topology(conc, ranks, grid)
    R, (ny, nx) = len(polys), grid.shape
    W, dy = grid.dx, grid.dy
    area = np.zeros((R + 1, ny, nx))
    area[R] = dy[:, None] * W[None, :]
    fx = np.zeros((R + 1, ny, nx + 1))
    fx[R] = dy[:, None]
    fyb = np.zeros((R + 1, ny, nx))
    fyb[R] = W[None, :]
    fyt = fyb.copy()
    if not topo.p_edge.size:
        out = _finish(np, area, fx, fyb, fyt, W, dy)
        if use_jax:
            import jax.numpy as jnp

            return tuple(jnp.asarray(o) for o in out)  # type: ignore[return-value]
        return out
    s = topo.p_slab
    args = (topo.e_lo, topo.e_hi, topo.cand_ref[topo.cand_kind == 1], topo.pairs, topo.bnd, s, topo.p_edge,
            topo.p_left, topo.p_right, topo.slab_row[s], topo.first[s].astype(float), topo.last[s].astype(float))
    if use_jax:
        import jax.numpy as jnp

        V = jnp.concatenate([jnp.asarray(p, dtype=float) for p in polys])
        return _phase2_jax()(V, grid.y_edges, grid.x_edges, dy, area, fx, fyb, fyt, *args)
    V = np.concatenate([np.asarray(p, dtype=float) for p in polys])
    return _phase2(V, grid.y_edges, grid.x_edges, dy, area, fx, fyb, fyt, *args, xpm=np)


# --------------------------------------------------------------------------- #
# Coverage (rasterized geometry) and material evaluation
# --------------------------------------------------------------------------- #
def _xp_of(*arrays: Any) -> Any:
    if any(_is_jax_array(a) for a in arrays):
        import jax.numpy as jnp

        return jnp
    return np


def _stack_values(values: list[Any]) -> Any:
    xpm = _xp_of(*values)
    return xpm.stack([xpm.asarray(v) for v in values])


@dataclass(frozen=True, eq=False)
class Coverage:
    """Rasterized geometry of an :class:`XSection` on one grid.

    Geometry is the expensive part; evaluating materials (any wavelength,
    temperature or physical quantity) on a :class:`Coverage` is cheap, so
    rasterize once and sweep.

    Attributes
    ----------
    grid
        The :class:`TensorGrid`.
    fractions
        ``(R + 1, ny, nx)`` exact *visible* area fraction of each region (in
        ``XSection.regions`` order) per cell; the last entry is the background.
        Sums to one in every cell.
    face_x
        ``(R + 1, ny, nx + 1)`` fraction of each vertical cell face
        (``x = x_edges[i]``) covered by each region.
    face_y
        ``(R + 1, ny + 1, nx)`` the same for horizontal faces
        (``y = y_edges[j]``).  An interface lying exactly on a face counts half
        to each side.
    """

    xsection: XSection
    grid: TensorGrid
    fractions: Any
    face_x: Any
    face_y: Any

    # -- helpers ------------------------------------------------------------ #
    @property
    def regions(self) -> tuple[Region, ...]:
        return tuple(self.xsection.regions)

    def _present(self) -> np.ndarray:
        f = _concrete(self.fractions.reshape(self.fractions.shape[0], -1).max(axis=1))
        present = f > 0.0
        if self.xsection.background is None:  # rasterize() verified it is covered (to rounding)
            present[-1] = False
        return present

    def _material(self, r: int) -> Any:
        return self.xsection.background if r == len(self.regions) else self.regions[r].material

    def _index_values(self, wl: float, temperature: float | None) -> tuple[Any, bool]:
        """Per-region ``n^2`` (``(R+1,)`` or ``(R+1, 3)``) and whether anisotropic."""
        present = self._present()
        vals: list[Any] = []
        for r in range(len(self.regions) + 1):
            if not present[r]:
                vals.append(None)
                continue
            n = resolve_index(self._material(r), wl, temperature)
            vals.append(n * n)
        # Broadcast isotropic entries to 3 components when any region is anisotropic.
        aniso = any(v is not None and np.shape(v) == (3,) for v in vals)
        fill = np.ones(3) if aniso else np.asarray(1.0)
        out = []
        for v in vals:
            if v is None:
                v = fill
            elif aniso and np.shape(v) == ():
                v = v * np.ones(3)
            out.append(v)
        return _stack_values(out), aniso

    def _property_values(self, quantity: str, temperature: float | None) -> Any:
        present = self._present()
        vals: list[Any] = []
        for r in range(len(self.regions) + 1):
            if not present[r]:
                vals.append(1.0)
            elif r == len(self.regions):
                vals.append(resolve_property(_BackgroundRegion(self.xsection), quantity, temperature))
            else:
                vals.append(resolve_property(self.regions[r], quantity, temperature))
        return _stack_values(vals)

    # -- public ------------------------------------------------------------- #
    def mask(self, tag: str) -> Any:
        """Visible area fraction ``(ny, nx)`` of all regions carrying ``tag``.

        ``"background"`` selects the background.
        """
        idx = [r for r, reg in enumerate(self.regions) if tag in reg.tags]
        if tag == "background":
            idx.append(len(self.regions))
        if not idx:
            raise KeyError(f"no region is tagged {tag!r}")
        return self.fractions[np.asarray(idx)].sum(axis=0)

    def normals(self, weights: Any) -> Any:
        """Unit interface normals ``(ny, nx, 2)`` from per-region ``weights``.

        ``weights`` (length ``R + 1``, last = background) are the values of the
        property whose gradient defines the normal (e.g. ``Re(eps)``).  Cells
        with no interface -- or whose integrated gradient vanishes, such as a
        sliver centred in its cell -- get ``(0, 0)``.
        """
        xpm = _xp_of(weights, self.face_x)
        w = xpm.asarray(weights)
        dx, dy = self.grid.dx, self.grid.dy
        gx = xpm.tensordot(w, self.face_x[:, :, 1:] - self.face_x[:, :, :-1], axes=1) * dy[:, None]
        gy = xpm.tensordot(w, self.face_y[:, 1:, :] - self.face_y[:, :-1, :], axes=1) * dx[None, :]
        m2 = gx * gx + gy * gy
        wc = _concrete(w)
        span = float(np.max(wc) - np.min(wc)) if wc.size else 0.0
        floor = 1e-9 * max(span, 1e-300) * float(min(dx.min(), dy.min()))
        ok = m2 > floor * floor
        safe = xpm.sqrt(xpm.where(ok, m2, 1.0))  # double-where: finite gradients where masked
        nx_ = xpm.where(ok, gx / safe, 0.0)
        ny_ = xpm.where(ok, gy / safe, 0.0)
        return xpm.stack([nx_, ny_], axis=-1)

    def _smooth(self, values: Any, kind: str, *, offdiagonal: bool = False) -> Any:
        """Average per-region values (``(R+1,)`` or ``(R+1, 3)``) over each cell."""
        f = self.fractions
        xpm = _xp_of(values, f)
        v = xpm.asarray(values)
        aniso = v.ndim == 2
        if kind == "arithmetic":
            return xpm.tensordot(v.T if aniso else v, f, axes=1)
        if kind == "harmonic":
            if aniso:
                raise ValueError("harmonic averaging of an anisotropic quantity is not defined; use 'tensor'")
            return 1.0 / xpm.tensordot(1.0 / v, f, axes=1)
        if kind != "tensor":
            raise ValueError(f"unknown averaging {kind!r}")
        vx = v[:, 0] if aniso else v
        vy = v[:, 1] if aniso else v
        vz = v[:, 2] if aniso else v
        weights = xpm.real(0.5 * (vx + vy))
        n = self.normals(weights)
        c, s = n[..., 0], n[..., 1]
        has_n = (c * c + s * s) > 0.5
        c = xpm.where(has_n, c, 1.0)
        s = xpm.where(has_n, s, 0.0)
        c2, s2, cs = c * c, s * s, c * s
        # Per-region rotated tensors in the (n, t) frame, then Kottke's tau average.
        exr = vx[:, None, None]
        eyr = vy[:, None, None]
        e_nn = c2 * exr + s2 * eyr
        e_tt = s2 * exr + c2 * eyr
        e_nt = cs * (eyr - exr)
        t_nn = xpm.sum(f * (-1.0 / e_nn), axis=0)
        t_nt = xpm.sum(f * (e_nt / e_nn), axis=0)
        t_tt = xpm.sum(f * (e_tt - e_nt * e_nt / e_nn), axis=0)
        a_nn = -1.0 / t_nn
        a_nt = -t_nt / t_nn
        a_tt = t_tt - t_nt * t_nt / t_nn
        exx = c2 * a_nn - 2 * cs * a_nt + s2 * a_tt
        eyy = s2 * a_nn + 2 * cs * a_nt + c2 * a_tt
        exy = cs * a_nn + (c2 - s2) * a_nt - cs * a_tt
        ar_x = xpm.tensordot(vx, f, axes=1)
        ar_y = xpm.tensordot(vy, f, axes=1)
        exx = xpm.where(has_n, exx, ar_x)
        eyy = xpm.where(has_n, eyy, ar_y)
        exy = xpm.where(has_n, exy, 0.0)
        ezz = xpm.tensordot(vz, f, axes=1)
        comps = [exx, eyy, ezz, exy] if offdiagonal else [exx, eyy, ezz]
        return xpm.stack(comps)

    def eps(
        self,
        wl: float = 1.55,
        temperature: float | None = None,
        *,
        smoothing: Smoothing = "arithmetic",
        offdiagonal: bool = False,
    ) -> Any:
        """Subpixel-smoothed relative permittivity on the grid.

        Parameters
        ----------
        wl
            Scalar wavelength (µm) at which the materials are evaluated.
        temperature
            Passed to temperature-aware materials (K); ``None`` = their default.
        smoothing
            ``"arithmetic"`` -- exact area average ``<eps>`` (the convention of
            :func:`photonix.em.rectangular_waveguide`): shape ``(ny, nx)``, or
            ``(3, ny, nx)`` (``xx, yy, zz``) if any material is anisotropic.
            ``"tensor"`` -- Kottke anisotropic smoothing, shape ``(3, ny, nx)``
            with the diagonal ``xx, yy, zz`` (``offdiagonal=True`` appends
            ``xy`` -> ``(4, ny, nx)``).  Intended for dielectrics; averaging
            ``1/eps`` across a metal (``Re eps < 0``) interface is ill-posed.

        Returns
        -------
        array
            Complex when any index is complex, otherwise real.  NumPy unless the
            geometry or materials carry JAX values.
        """
        if smoothing not in ("arithmetic", "tensor"):
            raise ValueError("smoothing must be 'arithmetic' or 'tensor'")
        vals, _aniso = self._index_values(wl, temperature)
        return self._smooth(vals, smoothing, offdiagonal=offdiagonal)

    def coefficient(
        self,
        quantity: str,
        temperature: float | None = None,
        *,
        average: Average = "arithmetic",
        offdiagonal: bool = False,
    ) -> Any:
        """Cell-averaged non-optical coefficient (``"k_th"``, ``"rho_cp"``, ``"sigma"``...).

        ``average``: ``"arithmetic"`` (volume average -- exact for capacities such
        as ``rho_cp``), ``"harmonic"`` (series average), or ``"tensor"`` (the
        laminate homogenization used for conductivities and ``eps_static``: series
        across the local interface, parallel along it; returns ``(3, ny, nx)``).
        Values come from :func:`photonix.xsection.resolve_property`; regions that
        do not appear on the grid need not define the quantity.
        """
        vals = self._property_values(quantity, temperature)
        return self._smooth(vals, average, offdiagonal=offdiagonal)


class _BackgroundRegion:
    """Region-like view of an XSection's background for property look-up."""

    name = "background"
    tags = frozenset({"background"})

    def __init__(self, xs: XSection) -> None:
        self.properties = xs.background_properties
        self.material = xs.background


# --------------------------------------------------------------------------- #
# Cross-section
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class XSection:
    """A 2-D multi-material cross-section: painted polygonal regions + background.

    One geometry object feeds every physics: :meth:`eps` for mode solvers,
    :meth:`coefficient` for heat/conduction solvers, :meth:`mask` for doping or
    contacts.  ``background`` fills everything no region covers; with
    ``background=None`` the regions must cover the whole window (as with
    :class:`photonix.pdk.LayerStack`) or rasterization raises.

    Examples
    --------
    >>> from photonix.xsection import XSection, rect, uniform_grid
    >>> xs = XSection([rect(-0.25, 0.25, -0.11, 0.11, 3.4757, tags={"core"})], background=1.444)
    >>> grid = uniform_grid(xs.bbox(), resolution=20)
    >>> eps = xs.eps(grid, wl=1.55)
    >>> eps.shape == grid.shape
    True
    >>> round(float((xs.mask(grid, "core") * grid.areas).sum()), 12)       # exact area
    0.11
    """

    regions: Sequence[Region]
    background: Any = None
    background_properties: Mapping[str, Any] = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self) -> None:
        regs = tuple(self.regions)
        for r in regs:
            if not isinstance(r, Region):
                raise TypeError(f"regions must be Region objects, got {type(r).__name__}")
        object.__setattr__(self, "regions", regs)
        if self.background is not None:
            _check_material(self.background)
        object.__setattr__(self, "background_properties", MappingProxyType(dict(self.background_properties)))

    # -- geometry ----------------------------------------------------------- #
    @property
    def ranks(self) -> np.ndarray:
        """Painter's rank of each region (higher paints over lower)."""
        order = sorted(range(len(self.regions)), key=lambda i: (self.regions[i].order, i))
        ranks = np.empty(len(self.regions), int)
        ranks[np.asarray(order, int)] = np.arange(len(self.regions))
        return ranks

    def bbox(self, include_bands: bool = False) -> tuple[float, float, float, float]:
        """``(x0, x1, y0, y1)`` of the bounded regions (bands only if asked)."""
        boxes = [r.bbox for r in self.regions if include_bands or not r.band]
        if not boxes:
            raise ValueError("the cross-section has no bounded region to define a window")
        b = np.asarray(boxes)
        return (float(b[:, 0].min()), float(b[:, 1].max()), float(b[:, 2].min()), float(b[:, 3].max()))

    def tagged(self, tag: str) -> tuple[Region, ...]:
        """Regions carrying ``tag``."""
        return tuple(r for r in self.regions if tag in r.tags)

    def with_regions(self, *regions: Region) -> XSection:
        """Copy with ``regions`` appended (painted last at their order)."""
        return XSection((*self.regions, *regions), self.background, self.background_properties, self.name)

    def translated(self, dx: float = 0.0, dy: float = 0.0) -> XSection:
        """Copy with every region moved by ``(dx, dy)`` (bands only vertically)."""
        return XSection(tuple(r.translated(dx, dy) for r in self.regions), self.background,
                        self.background_properties, self.name)

    def uniform_grid(self, resolution: float = 40, margin: float = 1.5) -> TensorGrid:
        """:func:`uniform_grid` around :meth:`bbox`."""
        return uniform_grid(self.bbox(), resolution, margin)

    def rasterize(self, grid: TensorGrid) -> Coverage:
        """Exact visible-area and face coverage of every region on ``grid``."""
        if not isinstance(grid, TensorGrid):
            raise TypeError("grid must be a TensorGrid (see uniform_grid / graded_grid)")
        x0, x1, _, _ = grid.bbox
        if max(abs(x0), abs(x1)) >= BAND_HALF_WIDTH:
            raise ValueError(f"the grid must lie within |x| < {BAND_HALF_WIDTH} um of the origin")
        frac, face_x, face_y = _sweep([r.polygon for r in self.regions], self.ranks, grid)
        if self.background is None:
            bg = _concrete(frac[-1])
            if bg.max() > 1e-9:
                j, i = np.unravel_index(int(np.argmax(bg)), bg.shape)
                raise ValueError(
                    f"no region covers the window near (x, y) = ({grid.xc[i]:.4g}, {grid.yc[j]:.4g}) um and "
                    "background=None; add a background material or a covering band"
                )
        return Coverage(self, grid, frac, face_x, face_y)

    # -- convenience wrappers ---------------------------------------------- #
    def fractions(self, grid: TensorGrid) -> Any:
        """``(R + 1, ny, nx)`` exact visible area fractions (last = background)."""
        return self.rasterize(grid).fractions

    def eps(self, grid: TensorGrid, wl: float = 1.55, temperature: float | None = None, *,
            smoothing: Smoothing = "arithmetic", offdiagonal: bool = False) -> Any:
        """Permittivity on ``grid`` (see :meth:`Coverage.eps`)."""
        return self.rasterize(grid).eps(wl, temperature, smoothing=smoothing, offdiagonal=offdiagonal)

    def normals(self, grid: TensorGrid, wl: float = 1.55, temperature: float | None = None) -> Any:
        """Interface normals ``(ny, nx, 2)`` weighted by ``Re(eps)`` at ``wl``."""
        cov = self.rasterize(grid)
        vals, aniso = cov._index_values(wl, temperature)
        w = (vals[:, 0] + vals[:, 1]) / 2 if aniso else vals
        return cov.normals(_xp_of(w).real(w))

    def coefficient(self, grid: TensorGrid, quantity: str, temperature: float | None = None, *,
                    average: Average = "arithmetic", offdiagonal: bool = False) -> Any:
        """Cell-averaged non-optical coefficient (see :meth:`Coverage.coefficient`)."""
        return self.rasterize(grid).coefficient(quantity, temperature, average=average, offdiagonal=offdiagonal)

    def mask(self, grid: TensorGrid, tag: str) -> Any:
        """Visible fraction ``(ny, nx)`` of the regions tagged ``tag``."""
        return self.rasterize(grid).mask(tag)

    def to_em(self, wl: float = 1.55, resolution: float = 40, margin: float = 1.5,
              temperature: float | None = None):
        """Drop-in :class:`photonix.em.CrossSection` (arithmetic smoothing, uniform grid).

        ``solve_modes_fullvector(eps=cs.eps, grid=(cs.x, cs.y))`` then solves the
        modes of this geometry exactly as it would a ``rectangular_waveguide``.

        Examples
        --------
        >>> from photonix.xsection import strip
        >>> cs = strip(0.5, 0.22).to_em(resolution=20)
        >>> cs.eps.shape, bool(cs.eps.max() > 12.0)
        ((65, 70), True)
        """
        from photonix.em.geometry import CrossSection as _EmCrossSection

        grid = self.uniform_grid(resolution, margin)
        eps = np.asarray(_concrete_or_array(self.eps(grid, wl, temperature)))
        if eps.ndim != 2:
            raise ValueError("to_em needs isotropic materials; use photonix.xsection.solve_modes for anisotropy")
        if np.iscomplexobj(eps) and not np.any(eps.imag):
            eps = eps.real
        return _EmCrossSection(eps=eps, x=grid.xc.copy(), y=grid.yc.copy())


def _concrete_or_array(value: Any) -> Any:
    if _is_jax_array(value):
        try:
            return np.asarray(value)
        except Exception:  # noqa: BLE001
            return value
    return value
