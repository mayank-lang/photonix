"""Full-vector modes of an :class:`~photonix.xsection.XSection`, with tensor smoothing.

:func:`solve_modes` discretizes the same Yee-grid (Rumpf) full-vector
eigenproblem as :func:`photonix.em.solve_modes_fullvector`, but lets each field
component see its own permittivity::

    Omega = P Q,  Q = [[Dhx Dey, -(Dhx Dex) - eps_yy], [Dhy Dey + eps_xx, -(Dhy Dex)]]
                  P = [[Dex eps_zz^-1 Dhy, -(Dex eps_zz^-1 Dhx) - 1], [Dey eps_zz^-1 Dhy + 1, -(Dey eps_zz^-1 Dhx)]]

On this staggered grid ``Ez`` sits at the cell centres ``(x_i, y_j)``, ``Ex`` at
``(x_i + dx/2, y_j)`` and ``Ey`` at ``(x_i, y_j + dy/2)``.  With
``smoothing="tensor"`` each component's permittivity is the Kottke-smoothed
tensor element averaged over a cell centred *on that component*: ``eps_xx`` (and
the off-diagonal ``eps_xy``) on the grid shifted by ``dx/2``, ``eps_yy`` on the
grid shifted by ``dy/2``, and ``eps_zz = <eps>`` (``z`` is tangential to every
interface) on the centred grid.  The off-diagonal term of slanted interfaces
couples ``D_x = eps_xx Ex + eps_xy A Ey`` and ``D_y = eps_yy Ey + A^T (eps_xy Ex)``,
``A`` being the 4-point average of ``Ey`` onto the ``Ex`` positions, which keeps
the discrete permittivity operator symmetric (cf. Oskooi et al., Opt. Lett. 34,
2778, 2009, for the analogous FDTD treatment).  It is essential: without it a
45-degree-rotated core converges only at first order.

With ``smoothing="arithmetic"`` all three components use the cell-centred
area average, which is exactly the operator of
:func:`photonix.em.solve_modes_fullvector` (the tests check identical
``n_eff``); unlike that solver this one also accepts anisotropic and lossy
(complex) media.

Each sample's average uses the ``kernel`` of :meth:`XSection.rasterize`:
``"box"`` (the cell centred on the component, default) or ``"tent"`` (the
bilinear hat reaching the neighbouring samples).

Evidence (``tests/test_xsection.py``; numbers for a 500 x 200 nm Si/SiO2 core
unless stated):

* rotation invariance -- the continuum ``n_eff`` of a rotated core does not
  depend on the angle.  At 45 deg and 40 / 80 px/um the tensor (box) error
  against the axis-aligned Richardson limit 2.36506 is 2.3e-3 / 4.0e-4 versus
  2.0e-2 / 1.0e-2 for arithmetic smoothing (first order);
* on grids whose lines contain the horizontal faces, rectangle and trapezoid
  converge with observed order ~2 (tensor) and both schemes extrapolate to the
  same limit within their GCI;
* tent kernel on ordinary (unaligned) grids: :func:`n_eff_convergence` over
  30/40/60 px/um recovers 2.36506 to 2.2e-4 (axis-aligned) and 2.9e-4
  (45 deg), inside its 3.5e-3 / 4.2e-3 error bounds; sub-cell shifts of an
  80-deg, 220 nm trapezoid move ``n_eff`` by about 4e-4 at 40 px/um (box: 5.6e-3);
* coupler supermode splitting reproduces ``SOI_STRIP_SPLITTING_FIT`` within
  its ~2 % trust band at 60 px/um (gaps 0.1-0.4 um);
* the sparse operator equals the matrix-free one used by the adjoint, and the
  arithmetic/box path reproduces the legacy solver to rounding.

Limitation: with the default box kernel the second-order interface error
depends on where a horizontal or vertical face cuts its cell (as for the 1-D
TE slab with cell-averaged ``eps``): ``n_eff`` of a 220 nm core moves by
~5e-3 at 40 px/um under sub-cell vertical shifts, and refinement sequences on
ordinary grids are often not monotone, so a Richardson estimate built from
them is unreliable (the vertical-shift sensitivity is still 3e-3 at
60 px/um).  With the tent kernel that sensitivity falls quickly with
resolution (3.4e-4 at 40, 1.5e-4 at 60 px/um; below ~20 px/um both kernels
move by several 1e-3), at the price of a larger, systematic error at one
resolution (about -6e-3 at 40 px/um for the 220 nm strip, about 2x-3x the
typical box error): use it with :func:`n_eff_convergence`.  These are
convergence (level 1) and exact-symmetry checks, not measurements.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from photonix.core.backend import HAS_JAX, xp
from photonix.em.convergence import GridConvergenceResult, estimate_convergence
from photonix.em.fde_vector import VectorModeData

from .raster import Kernel, TensorGrid, XSection, uniform_grid
from .shapes import _host, _is_jax_array

__all__ = ["XSectionModes", "yee_permittivity", "solve_modes", "n_eff", "n_eff_eps_tensor", "n_eff_convergence"]

if HAS_JAX:  # pragma: no branch
    import jax
    import jax.numpy as jnp


@dataclass
class XSectionModes(VectorModeData):
    """:class:`photonix.em.VectorModeData` plus both transverse fields and the grid.

    ``ex[i]``, ``ey[i]`` are mode ``i``'s transverse electric fields on their
    staggered Yee positions (arrays ``(ny, nx)``), normalized so that
    ``sum(|ex|^2 + |ey|^2) = 1``.
    """

    ex: np.ndarray | None = None
    ey: np.ndarray | None = None
    grid: TensorGrid | None = None
    smoothing: str = "arithmetic"
    kernel: str = "box"


def _ddx_fwd(n: int, h: float) -> sp.csc_matrix:
    e = np.ones(n)
    return sp.diags([-e, e[:-1]], [0, 1], format="csc") / h


def _avg_ey_to_ex(ny: int, nx: int) -> sp.csc_matrix:
    """4-point average of ``Ey`` onto the ``Ex`` positions (``A_y = A_x^T``)."""
    iy = sp.identity(ny) + sp.diags(np.ones(ny - 1), -1)
    ix = sp.identity(nx) + sp.diags(np.ones(nx - 1), 1)
    return (0.25 * sp.kron(iy, ix)).tocsc()


def _assemble(comps: tuple[np.ndarray, ...], dx: float, dy: float, k0: float):
    """Sparse ``Omega = P Q`` for ``comps = (exx, eyy, ezz[, exy])`` on the Yee positions.

    ``exy`` (sampled at the ``Ex`` positions) couples ``D_x = exx Ex + exy A Ey``
    and, symmetrically, ``D_y = eyy Ey + A^T (exy Ex)`` with ``A`` the 4-point
    average of ``Ey`` onto ``Ex`` -- a symmetric discrete permittivity operator.
    """
    exx, eyy, ezz = comps[:3]
    ny, nx = exx.shape
    n = ny * nx
    dex = sp.kron(sp.identity(ny), _ddx_fwd(nx, dx)) / k0
    dey = sp.kron(_ddx_fwd(ny, dy), sp.identity(nx)) / k0
    dhx = -dex.T.tocsc()
    dhy = -dey.T.tocsc()
    erxx = sp.diags(exx.reshape(-1))
    eryy = sp.diags(eyy.reshape(-1))
    izz = sp.diags(1.0 / ezz.reshape(-1))
    ident = sp.identity(n)
    P = sp.bmat([[dex @ izz @ dhy, -(dex @ izz @ dhx) - ident],
                 [dey @ izz @ dhy + ident, -(dey @ izz @ dhx)]], format="csc")
    q11, q22 = dhx @ dey, -(dhy @ dex)
    if len(comps) == 4:
        avg = _avg_ey_to_ex(ny, nx)
        exy = sp.diags(comps[3].reshape(-1))
        q11 = q11 - avg.T @ exy
        q22 = q22 + exy @ avg
    Q = sp.bmat([[q11, -(dhx @ dex) - eryy],
                 [dhy @ dey + erxx, q22]], format="csc")
    return (P @ Q).tocsc()


def _apply(comps: tuple[Any, ...], field: Any, shape: tuple[int, int], dx: float, dy: float, k0: float):
    """Matrix-free ``Omega @ [Ex; Ey]`` in ``xp`` (single source for the adjoint).

    ``comps = (exx, eyy, ezz[, exy])`` as flat arrays.
    """
    ny, nx = shape
    n = ny * nx
    exx, eyy, ezz = comps[:3]
    izz = 1.0 / ezz

    def dex(f):
        f2 = f.reshape(ny, nx)
        fp = xp.pad(f2, ((0, 0), (0, 1)))
        return ((fp[:, 1:] - f2) / (k0 * dx)).reshape(-1)

    def dey(f):
        f2 = f.reshape(ny, nx)
        fp = xp.pad(f2, ((0, 1), (0, 0)))
        return ((fp[1:, :] - f2) / (k0 * dy)).reshape(-1)

    def dhx(f):
        f2 = f.reshape(ny, nx)
        fp = xp.pad(f2, ((0, 0), (1, 0)))
        return ((f2 - fp[:, :-1]) / (k0 * dx)).reshape(-1)

    def dhy(f):
        f2 = f.reshape(ny, nx)
        fp = xp.pad(f2, ((1, 0), (0, 0)))
        return ((f2 - fp[:-1, :]) / (k0 * dy)).reshape(-1)

    ex, ey = field[:n], field[n:]
    dx_ = exx * ex
    dy_ = eyy * ey
    if len(comps) == 4:
        exy = comps[3]
        e2 = ey.reshape(ny, nx)
        p = xp.pad(e2, ((1, 0), (0, 1)))  # p[j+1, i] = Ey[j, i]; zero above/right
        a_ey = 0.25 * (p[1:, :-1] + p[:-1, :-1] + p[1:, 1:] + p[:-1, 1:])
        dx_ = dx_ + exy * a_ey.reshape(-1)
        t = (exy * ex).reshape(ny, nx)
        q = xp.pad(t, ((0, 1), (1, 0)))   # q[j, i+1] = t[j, i]
        at = 0.25 * (q[:-1, 1:] + q[:-1, :-1] + q[1:, 1:] + q[1:, :-1])
        dy_ = dy_ + at.reshape(-1)
    qa = dhx(dey(ex)) - dhx(dex(ey)) - dy_
    qb = dhy(dey(ex)) + dx_ - dhy(dex(ey))
    pa = dex(izz * dhy(qa)) - dex(izz * dhx(qb)) - qb
    pb = dey(izz * dhy(qa)) + qa - dey(izz * dhx(qb))
    return xp.concatenate([pa, pb])


def _solve(comps, dx, dy, k0, num_modes, want_left=False):
    A = _assemble(comps, dx, dy, k0)
    exx, eyy = comps[0], comps[1]
    ny, nx = exx.shape
    n = ny * nx
    lossless = not any(np.iscomplexobj(e) and np.any(np.imag(e)) for e in comps)
    nmax2 = float(max(np.max(np.real(exx)), np.max(np.real(eyy))))
    k = int(min(max(num_modes, 1), A.shape[0] - 2))
    vals, vecs = spla.eigs(A, k=k, sigma=-nmax2 * 1.0001, which="LM")
    neff2 = -vals.astype(complex)
    if lossless:
        scale = np.maximum(1.0, np.abs(np.real(neff2)))
        if np.any(np.abs(np.imag(neff2)) > 1e-7 * scale):
            raise RuntimeError("lossless full-vector solve returned materially complex eigenvalues")
        neff2 = np.real(neff2).astype(complex)
    order = np.argsort(np.real(neff2))[::-1]
    vals, vecs, neff2 = vals[order], vecs[:, order], neff2[order]
    neff = np.sqrt(neff2)
    neff = np.where((np.real(neff2) < 0) & (np.imag(neff) > 0), -neff, neff)  # decaying evanescent branch
    if lossless:
        neff = np.where(np.real(neff2) >= 0, np.real(neff), neff)
    exs, eys, te = [], [], []
    for i in range(vecs.shape[1]):
        v = vecs[:, i]
        v = v / np.sqrt(np.sum(np.abs(v) ** 2))
        v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
        if lossless:
            v = np.real(v)
        exs.append(v[:n].reshape(ny, nx))
        eys.append(v[n:].reshape(ny, nx))
        fx, fy = float(np.sum(np.abs(v[:n]) ** 2)), float(np.sum(np.abs(v[n:]) ** 2))
        te.append(fx / (fx + fy))
    out = (neff[:num_modes], np.array(exs)[:num_modes], np.array(eys)[:num_modes], np.array(te)[:num_modes])
    if not want_left:
        return out
    v0 = np.real(vecs[:, 0])
    v0 = v0 / np.sqrt(np.sum(v0**2))
    _, lvecs = spla.eigs(A.T.tocsc(), k=1, sigma=complex(vals[0]), which="LM")
    u0 = np.real(lvecs[:, 0])
    return out, v0, u0


def _check_uniform(grid: TensorGrid) -> tuple[float, float]:
    if not grid.is_uniform():
        raise ValueError("the FDE mode solvers need a uniform grid (see uniform_grid)")
    return float(grid.dx[0]), float(grid.dy[0])


def yee_permittivity(
    xs: XSection,
    grid: TensorGrid,
    wl: float = 1.55,
    temperature: float | None = None,
    *,
    smoothing: Literal["arithmetic", "tensor"] = "tensor",
    offdiagonal: bool = True,
    kernel: Kernel = "box",
) -> tuple[Any, ...]:
    """Permittivity components on the Yee positions of a uniform ``grid``.

    ``"tensor"``: Kottke smoothing over cells centred on ``Ex`` (grid shifted
    by ``+dx/2``: ``eps_xx`` and, with ``offdiagonal``, ``eps_xy``), ``Ey``
    (``+dy/2``: ``eps_yy``) and ``Ez`` (the grid itself: ``eps_zz``).
    Returns ``(exx, eyy, ezz, exy)`` or, without ``offdiagonal``,
    ``(exx, eyy, ezz)``.
    ``"arithmetic"``: the cell-centred area average for all three diagonal
    components (the convention of :func:`photonix.em.solve_modes_fullvector`);
    always ``(exx, eyy, ezz)``.
    ``kernel`` selects the averaging weight of each sample (see
    :meth:`XSection.rasterize`): the cell centred on the component (``"box"``)
    or the tent spanning its neighbours (``"tent"``).

    Examples
    --------
    >>> from photonix.xsection import strip
    >>> xs = strip(0.5, 0.22)
    >>> exx, eyy, ezz, exy = yee_permittivity(xs, xs.uniform_grid(20), smoothing="tensor")
    >>> exx.shape == eyy.shape == ezz.shape == exy.shape
    True
    """
    _check_uniform(grid)
    dx, dy = float(grid.dx[0]), float(grid.dy[0])
    if smoothing == "arithmetic":
        e = xs.eps(grid, wl, temperature, kernel=kernel)
        if e.ndim == 3:
            return e[0], e[1], e[2]
        return e, e, e
    if smoothing != "tensor":
        raise ValueError("smoothing must be 'arithmetic' or 'tensor'")
    ex_cell = xs.eps(grid.shifted(0.5 * dx, 0.0), wl, temperature, smoothing="tensor", offdiagonal=offdiagonal,
                     kernel=kernel)
    eyy = xs.eps(grid.shifted(0.0, 0.5 * dy), wl, temperature, smoothing="tensor", kernel=kernel)[1]
    ezz = xs.eps(grid, wl, temperature, smoothing="tensor", kernel=kernel)[2]
    if offdiagonal:
        return ex_cell[0], eyy, ezz, ex_cell[3]
    return ex_cell[0], eyy, ezz


def _exterior_index(*comps: np.ndarray) -> float:
    vals = []
    for e in comps:
        e = np.real(e)
        vals.append(np.concatenate((e[0], e[-1], e[1:-1, 0], e[1:-1, -1])).max())
    return float(np.sqrt(max(vals)))


def solve_modes(
    xs: XSection,
    *,
    wl: float = 1.55,
    num_modes: int = 1,
    resolution: float = 40,
    margin: float = 1.5,
    smoothing: Literal["arithmetic", "tensor"] = "tensor",
    temperature: float | None = None,
    grid: TensorGrid | None = None,
    offdiagonal: bool = True,
    kernel: Kernel = "box",
) -> XSectionModes:
    """Full-vector hybrid modes of a cross-section.

    Parameters
    ----------
    xs
        The cross-section.
    wl
        Wavelength (µm).
    num_modes
        Number of modes, ordered by decreasing ``Re(n_eff)``.
    resolution, margin
        Uniform grid around ``xs.bbox()`` (see :func:`uniform_grid`), unless
        ``grid`` is given.
    smoothing
        ``"tensor"`` (default): staggered Kottke smoothing, second-order
        accurate at material interfaces.  ``"arithmetic"``: the legacy
        area-averaged permittivity (observed order ~1.7).
    temperature
        Passed to temperature-aware materials.
    grid
        Explicit uniform :class:`TensorGrid` (overrides ``resolution`` and ``margin``).
    offdiagonal
        Include the ``eps_xy`` term of slanted interfaces (tensor smoothing only).
    kernel
        Averaging kernel (see :meth:`XSection.rasterize`).  ``"box"`` (default)
        has the smaller error at one resolution but that error depends on where
        horizontal/vertical faces cut their cells (``n_eff`` of a 220 nm core
        moves by ~5e-3 at 40 px/um under sub-cell shifts), so refinement
        sequences are not monotone.  ``"tent"`` makes the error smooth in the
        grid spacing (~4e-4 shift spread) at the price of a larger, systematic
        ``O(h**2)`` bias -- use it for Richardson extrapolation
        (:func:`n_eff_convergence`).

    Returns
    -------
    XSectionModes
        ``n_eff`` is complex when a material is lossy (``Im < 0`` is loss,
        ``loss_dB/um = 20 log10(e) * 2 pi |Im n_eff| / wl``).

    Examples
    --------
    >>> from photonix.xsection import strip
    >>> m = solve_modes(strip(0.5, 0.22), resolution=20)
    >>> bool(2.3 < m.neff0 < 2.5 and m.te_fraction[0] > 0.9)
    True
    """
    if (not isinstance(num_modes, (int, np.integer)) or isinstance(num_modes, (bool, np.bool_))
            or num_modes <= 0):
        raise ValueError("num_modes must be a positive integer")
    if not np.isfinite(wl) or wl <= 0:
        raise ValueError("wl must be positive and finite")
    grid = uniform_grid(xs.bbox(), resolution, margin) if grid is None else grid
    dx, dy = _check_uniform(grid)
    comps = tuple(_host(e) for e in yee_permittivity(xs, grid, wl, temperature, smoothing=smoothing,
                                                     offdiagonal=offdiagonal, kernel=kernel))
    k0 = 2.0 * np.pi / wl
    neff, ex, ey, te = _solve(comps, dx, dy, k0, num_modes)
    fields = np.array([a if t >= 0.5 else b for a, b, t in zip(ex, ey, te, strict=True)])
    guided = np.real(neff) > _exterior_index(*comps[:3])
    return XSectionModes(n_eff=neff, fields=fields, x=grid.xc.copy(), y=grid.yc.copy(), wl=wl,
                         polarization="full", te_fraction=te, guided=guided, ex=ex, ey=ey, grid=grid,
                         smoothing=smoothing, kernel=kernel)


if HAS_JAX:

    def _host_solve(e, shape, dx, dy, k0):
        ny, nx = shape
        n = ny * nx
        e = np.asarray(e, float)
        comps = tuple(e[k * n:(k + 1) * n].reshape(ny, nx) for k in range(e.size // n))
        (neff, _ex, _ey, _te), v0, u0 = _solve(comps, dx, dy, k0, 1, want_left=True)
        return (np.asarray(np.real(neff[0]), np.float64), np.asarray(v0, np.float64),
                np.asarray(u0, np.float64))

    def _callback(e, shape, dx, dy, k0):
        m = 2 * shape[0] * shape[1]
        return jax.pure_callback(
            partial(_host_solve, shape=shape, dx=dx, dy=dy, k0=k0),
            (jax.ShapeDtypeStruct((), jnp.float64), jax.ShapeDtypeStruct((m,), jnp.float64),
             jax.ShapeDtypeStruct((m,), jnp.float64)),
            e,
        )

    @partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
    def n_eff_eps_tensor(eps3_flat, shape, dx, dy, k0):
        """Differentiable fundamental ``n_eff`` from stacked Yee permittivity components.

        ``eps3_flat`` is ``concatenate([exx.ravel(), eyy.ravel(), ezz.ravel()])``
        or, with the off-diagonal term, ``[..., exy.ravel()]`` (real; see
        :func:`yee_permittivity`).  The VJP is the non-symmetric eigenvalue-perturbation adjoint
        ``d lambda = u^T dOmega v / u^T v`` with ``lambda = -n_eff^2``, built on
        the matrix-free operator that the sparse assembly is tested against.
        """
        neff, _v, _u = _callback(eps3_flat, shape, dx, dy, k0)
        return neff

    def _fwd(eps3_flat, shape, dx, dy, k0):
        neff, v, u = _callback(eps3_flat, shape, dx, dy, k0)
        return neff, (eps3_flat, v, u, neff)

    def _bwd(shape, dx, dy, k0, res, g):
        e, v, u, neff = res
        n = shape[0] * shape[1]

        def bilinear(ee):
            comps = tuple(ee[k * n:(k + 1) * n] for k in range(ee.size // n))
            return jnp.sum(u * _apply(comps, v, shape, dx, dy, k0))

        dlam = jax.grad(bilinear)(e) / jnp.sum(u * v)
        return (g * dlam * (-1.0 / (2.0 * neff)),)

    n_eff_eps_tensor.defvjp(_fwd, _bwd)

else:  # pragma: no cover - NumPy fallback has no autodiff

    def n_eff_eps_tensor(eps3_flat, shape, dx, dy, k0):
        """Differentiable ``n_eff`` from stacked permittivity components (requires JAX)."""
        raise RuntimeError("n_eff_eps_tensor requires JAX. Install photonix[jax].")


def n_eff(
    xs: XSection,
    *,
    wl: float = 1.55,
    resolution: float = 40,
    margin: float = 1.5,
    smoothing: Literal["arithmetic", "tensor"] = "tensor",
    temperature: float | None = None,
    grid: TensorGrid | None = None,
    offdiagonal: bool = True,
    kernel: Kernel = "box",
) -> Any:
    """Fundamental (highest-index) real ``n_eff``; differentiable through the geometry.

    Arguments as for :func:`solve_modes`.  When the polygons of ``xs`` hold
    JAX values (e.g. built from a traced width), the result is a JAX scalar
    whose gradient flows through the exact rasterization (either kernel) and
    the eigenvalue adjoint (lossless materials).  Otherwise a Python float.

    Examples
    --------
    >>> from photonix.xsection import strip
    >>> 2.3 < n_eff(strip(0.5, 0.22), resolution=20) < 2.5
    True
    """
    grid = uniform_grid(xs.bbox(), resolution, margin) if grid is None else grid
    dx, dy = _check_uniform(grid)
    comps = yee_permittivity(xs, grid, wl, temperature, smoothing=smoothing, offdiagonal=offdiagonal,
                             kernel=kernel)
    k0 = 2.0 * np.pi / wl
    if not any(_is_jax_array(c) for c in comps):
        neff = _solve(tuple(np.asarray(c) for c in comps), dx, dy, k0, 1)[0]
        return float(np.real(neff[0]))
    if not HAS_JAX:  # pragma: no cover
        raise RuntimeError("differentiating n_eff requires JAX")
    if any(jnp.iscomplexobj(c) for c in comps):
        raise ValueError("the differentiable n_eff supports lossless (real) permittivity only")
    flat = jnp.concatenate([jnp.ravel(c) for c in comps])
    return n_eff_eps_tensor(flat, grid.shape, dx, dy, k0)


def n_eff_convergence(
    xs: XSection,
    resolutions: Sequence[int] = (20, 30, 40),
    *,
    wl: float = 1.55,
    margin: float = 1.5,
    smoothing: Literal["arithmetic", "tensor"] = "tensor",
    kernel: Kernel = "tent",
    temperature: float | None = None,
    offdiagonal: bool = True,
    order: float | None = 2.0,
    rtol: float = 1e-3,
    atol: float = 0.0,
) -> GridConvergenceResult:
    """Grid-refinement study of the fundamental ``n_eff`` with Richardson extrapolation.

    Solves on ``uniform_grid(xs.bbox(), r, margin)`` for every ``r`` in
    ``resolutions`` (px/µm, increasing, at least three) and hands the values to
    :func:`photonix.em.estimate_convergence`.  The default ``kernel="tent"``
    with tensor smoothing is the combination whose error is a smooth
    ``C h**2`` (observed order 1.6-2.1 in the tests, independent of how the
    faces sit in the cells), so the extrapolation and its GCI are meaningful;
    with ``kernel="box"`` the sequence is generally not monotone and the study
    reports ``asymptotic=False``.

    Returns
    -------
    photonix.em.GridConvergenceResult
        ``extrapolated`` is the Richardson value, ``estimated_absolute_error``
        the safety-factored error bound (GCI); check ``asymptotic`` before use.
        Only the discretization error is estimated -- the window ``margin``,
        the material data and the geometry are not.

    Examples
    --------
    >>> from photonix.xsection import strip
    >>> study = n_eff_convergence(strip(0.5, 0.22), (10, 14, 20), margin=1.0)
    >>> bool(study.asymptotic), bool(2.40 < float(study.extrapolated) < 2.50)
    (True, True)
    """
    res = [int(r) for r in resolutions]
    if [float(r) for r in resolutions] != [float(r) for r in res]:
        raise ValueError("resolutions must be integers (px/um)")
    if len(res) < 3 or res[0] <= 0 or any(b <= a for a, b in zip(res[:-1], res[1:], strict=True)):
        raise ValueError("resolutions must be at least three positive, strictly increasing values")
    values = [float(n_eff(xs, wl=wl, resolution=r, margin=margin, smoothing=smoothing, kernel=kernel,
                          temperature=temperature, offdiagonal=offdiagonal)) for r in res]
    return estimate_convergence(res, np.asarray(values), order=order, rtol=rtol, atol=atol)
