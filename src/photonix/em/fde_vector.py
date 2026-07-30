"""Polarization-resolved (semivectorial) and full-vector 2-D FDE mode solver.

This module advances the EM core described in ``full_vector_fde_design.md`` past
the scalar :mod:`photonix.em.fde` solver, which cannot represent the normal-field
discontinuity at high-index-contrast sidewalls and so over-estimates the strip TE
index and cannot separate TE from TM.

Two solvers are provided:

* **Semivectorial** (:func:`solve_modes_vector`): solves the dominant transverse
  electric field with an index-weighted second derivative across the interface it
  is *normal* to and a plain second derivative along the interface it is
  *tangential* to::

      quasi-TE (Ex): d/dx[(1/eps) d(eps Ex)/dx] + d2Ex/dy2 + (k0^2 eps - b^2)Ex = 0
      quasi-TM (Ey): d2Ey/dx2 + d/dy[(1/eps) d(eps Ey)/dy] + (k0^2 eps - b^2)Ey = 0

* **Full-vector** (:func:`solve_modes_fullvector`): the Yee-grid FDFD formulation
  that solves both ``Ex`` and ``Ey`` simultaneously, resolving the hybrid mode,
  its polarization fraction, and -- with stretched-coordinate PML --
  bend/radiation loss (complex n_eff) via :func:`bend_loss_fullvector`.

Both operators are **non-symmetric** -- unlike the scalar Helmholtz operator.
That has one consequence for differentiability: the eigenvalue-perturbation
adjoint needs the *left* eigenvector as well as the right one (for a symmetric
operator they coincide, which is why the scalar solver could use the cheap closed
form ``x_k^2 / (2 n_eff)``). :func:`n_eff_eps_vector` and
:func:`n_eff_eps_fullvector` implement that non-symmetric adjoint via
``jax.custom_vjp`` and the frozen-bilinear-form trick ``d lambda = (uT dA v)/(uT v)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from photonix.core.backend import HAS_JAX, xp

__all__ = [
    "VectorModeData",
    "solve_modes_vector",
    "n_eff_vector",
    "n_eff_eps_vector",
    "solve_modes_fullvector",
    "n_eff_fullvector",
    "n_eff_eps_fullvector",
    "bend_loss_fullvector",
    "BendMode",
    "fullvector_transverse_fields",
    "power_overlap",
]


# --------------------------------------------------------------------------- #
# Semivectorial operator coefficients (single source of truth, used by both the
# scipy assembler and the xp matvec so the eigenproblem and the adjoint stay
# numerically identical).
# --------------------------------------------------------------------------- #
def _coeffs(eps, dx, dy, k0, pol, mod):
    """Five-point stencil coefficients for the semivectorial operator.

    ``eps`` has shape ``(ny, nx)``. Returns ``(center, cxp, cxm, cyp, cym)``,
    each ``(ny, nx)``: the diagonal and the couplings to the +x, -x, +y, -y
    neighbours. ``mod`` is the array module (``numpy`` for assembly, ``xp`` for
    the differentiable matvec). ``pol`` is ``"te"`` (Ex, x-weighted) or ``"tm"``
    (Ey, y-weighted).
    """
    epsp = mod.pad(eps, 1, mode="edge")
    eps_c = epsp[1:-1, 1:-1]
    eps_xp = epsp[1:-1, 2:]
    eps_xm = epsp[1:-1, :-2]
    eps_yp = epsp[2:, 1:-1]
    eps_ym = epsp[:-2, 1:-1]

    def weighted(eps_plus, eps_minus, h):
        half_p = 0.5 * (eps_c + eps_plus)
        half_m = 0.5 * (eps_c + eps_minus)
        cp = eps_plus / (h ** 2 * half_p)
        cm = eps_minus / (h ** 2 * half_m)
        cc = -eps_c / h ** 2 * (1.0 / half_p + 1.0 / half_m)
        return cp, cm, cc

    def plain(h, shape):
        cp = mod.ones(shape) / h ** 2
        cm = mod.ones(shape) / h ** 2
        cc = -2.0 / h ** 2
        return cp, cm, cc

    if pol == "te":
        cxp, cxm, cxc = weighted(eps_xp, eps_xm, dx)
        cyp, cym, cyc = plain(dy, eps_c.shape)
    elif pol == "tm":
        cxp, cxm, cxc = plain(dx, eps_c.shape)
        cyp, cym, cyc = weighted(eps_yp, eps_ym, dy)
    else:
        raise ValueError(f"pol must be 'te' or 'tm', got {pol!r}")

    center = cxc + cyc + k0 ** 2 * eps_c
    return center, cxp, cxm, cyp, cym


def _assemble(eps, dx, dy, k0, pol):
    """Sparse semivectorial operator ``A`` (CSC) for the eigensolve."""
    ny, nx = eps.shape
    center, cxp, cxm, cyp, cym = _coeffs(np.asarray(eps, float), dx, dy, k0, pol, np)
    n = ny * nx
    iy, ix = np.divmod(np.arange(n), nx)

    rows = [np.arange(n)]
    cols = [np.arange(n)]
    vals = [center.reshape(-1)]

    def add(mask, col, val):
        rows.append(np.arange(n)[mask])
        cols.append(col[mask])
        vals.append(val.reshape(-1)[mask])

    add(ix < nx - 1, np.arange(n) + 1, cxp)
    add(ix > 0, np.arange(n) - 1, cxm)
    add(iy < ny - 1, np.arange(n) + nx, cyp)
    add(iy > 0, np.arange(n) - nx, cym)

    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    return A.tocsc()


def _apply(eps_flat, field_flat, shape, dx, dy, k0, pol):
    """Differentiable matvec ``A(eps) @ field`` (semivectorial)."""
    ny, nx = shape
    eps = eps_flat.reshape(ny, nx)
    field = field_flat.reshape(ny, nx)
    center, cxp, cxm, cyp, cym = _coeffs(eps, dx, dy, k0, pol, xp)
    fp = xp.pad(field, 1, mode="constant")
    f_xp = fp[1:-1, 2:]
    f_xm = fp[1:-1, :-2]
    f_yp = fp[2:, 1:-1]
    f_ym = fp[:-2, 1:-1]
    out = center * field + cxp * f_xp + cxm * f_xm + cyp * f_yp + cym * f_ym
    return out.reshape(-1)


def _solve(eps, dx, dy, k0, pol, num_modes, want_left=False):
    """Right (and optionally left) fundamental eigenpair(s) (semivectorial)."""
    A = _assemble(eps, dx, dy, k0, pol)
    n_max = float(np.sqrt(np.asarray(eps).max()))
    sigma = (n_max * k0) ** 2 * 1.0001
    k = int(min(max(num_modes, 1), A.shape[0] - 2))
    vals, vecs = spla.eigs(A, k=k, sigma=sigma, which="LM")
    order = np.argsort(np.real(vals))[::-1]
    vals, vecs = vals[order], vecs[:, order]

    betas = np.sqrt(vals.astype(complex))
    neff = betas / k0
    ny, nx = eps.shape
    fields = np.array([np.real(vecs[:, i]).reshape(ny, nx) for i in range(vecs.shape[1])])
    for i in range(fields.shape[0]):
        flat = fields[i].reshape(-1)
        if flat[np.argmax(np.abs(flat))] < 0:
            fields[i] *= -1.0
    v0 = np.real(vecs[:, 0])
    v0 = v0 / np.sqrt(np.sum(v0 ** 2))

    if not want_left:
        return neff[:num_modes], fields[:num_modes], v0

    lam0 = complex(vals[0])
    lvals, lvecs = spla.eigs(A.T.tocsc(), k=1, sigma=lam0, which="LM")
    u0 = np.real(lvecs[:, 0])
    return neff[:num_modes], fields[:num_modes], v0, u0, float(np.real(lam0))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@dataclass
class VectorModeData:
    """Result of a semivectorial or full-vector FDE solve.

    ``te_fraction[i]`` is the fraction of mode ``i``'s transverse electric energy
    in ``Ex`` (``> 0.5`` => TE-like, ``< 0.5`` => TM-like). It is ``None`` for the
    semivectorial solver, which fixes the polarization by construction.

    ``guided[i]`` is ``True`` when ``n_clad < Re(n_eff[i])``; modes below the
    cladding index are box modes of the truncated domain and carry no physical
    meaning (see PHYSICS_AUDIT §C2).

    For a full-vector solve, ``fields[i]`` stores the dominant transverse
    component (``Ex`` or ``Ey``) only. Use :func:`fullvector_transverse_fields`
    when both electric and magnetic transverse components are required.
    """

    n_eff: np.ndarray
    fields: np.ndarray
    x: np.ndarray
    y: np.ndarray
    wl: float
    polarization: str
    te_fraction: np.ndarray | None = None
    guided: np.ndarray | None = None

    @property
    def neff0(self) -> float:
        return float(np.real(self.n_eff[0]))

    @property
    def n_guided(self) -> int:
        """Number of guided modes (``Re(n_eff) > n_clad``)."""
        if self.guided is None:
            return len(self.n_eff)
        return int(np.sum(self.guided))


def _exterior_index(eps: np.ndarray) -> float:
    """Highest refractive index on the boundary of a custom cross-section."""
    boundary = np.concatenate((eps[0], eps[-1], eps[1:-1, 0], eps[1:-1, -1]))
    return float(np.sqrt(np.max(np.real(boundary))))


def solve_modes_vector(
    *,
    wl: float = 1.55,
    width: float = 0.5,
    thickness: float = 0.22,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    polarization: str = "te",
    num_modes: int = 1,
    resolution: int = 40,
    margin: float = 1.5,
    eps: np.ndarray | None = None,
    grid: tuple | None = None,
) -> VectorModeData:
    """Solve semivectorial quasi-TE / quasi-TM modes of a cross-section.

    Examples
    --------
    >>> r = solve_modes_vector(width=0.5, thickness=0.22, resolution=30, polarization="te")
    >>> 1.444 < r.neff0 < 3.4757
    True
    """
    if (not isinstance(num_modes, (int, np.integer))
            or isinstance(num_modes, (bool, np.bool_)) or num_modes <= 0):
        raise ValueError("num_modes must be a positive integer")
    if not np.isfinite(wl) or wl <= 0:
        raise ValueError("wl must be positive and finite")
    eps_user = eps is not None
    if eps is None:
        from .geometry import rectangular_waveguide

        cs = rectangular_waveguide(
            width=width, thickness=thickness, n_core=n_core, n_clad=n_clad,
            margin=margin, resolution=resolution,
        )
        eps, x, y, dx, dy = cs.eps, cs.x, cs.y, cs.dx, cs.dy
    else:
        from .geometry import _validate_eps_grid

        eps, x, y, dx, dy = _validate_eps_grid(
            eps, grid, where="solve_modes_vector"
        )
    from .geometry import as_real_eps

    k0 = 2.0 * np.pi / wl
    neff, fields, _ = _solve(as_real_eps(eps, where="solve_modes_vector"), dx, dy, k0, polarization, num_modes)
    n_clad_eff = _exterior_index(eps) if eps_user else n_clad
    guided = np.real(neff) > n_clad_eff
    return VectorModeData(
        n_eff=neff, fields=fields, x=np.asarray(x), y=np.asarray(y),
        wl=wl, polarization=polarization, guided=guided,
    )


def n_eff_vector(
    *,
    wl: float = 1.55,
    width: float = 0.5,
    thickness: float = 0.22,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    polarization: str = "te",
    resolution: int = 40,
    margin: float = 1.5,
    richardson: bool = False,
) -> float:
    """Fundamental semivectorial effective index for ``polarization``."""
    def _ne(res):
        return solve_modes_vector(
            wl=wl, width=width, thickness=thickness, n_core=n_core, n_clad=n_clad,
            polarization=polarization, num_modes=1, resolution=res, margin=margin,
        ).neff0

    if not richardson:
        return _ne(resolution)
    n_c, n_f = _ne(resolution), _ne(2 * resolution)
    return (4.0 * n_f - n_c) / 3.0


# --------------------------------------------------------------------------- #
# Semivectorial differentiable adjoint (non-symmetric: needs the left eigenvector)
# --------------------------------------------------------------------------- #
if HAS_JAX:
    import jax
    import jax.numpy as jnp

    def _solve_callback(eps_flat, shape, dx, dy, k0, pol):
        ny, nx = shape

        def host(e):
            eps = np.asarray(e, float).reshape(ny, nx)
            neff, _f, v0, u0, _lam = _solve(eps, dx, dy, k0, pol, 1, want_left=True)
            return (
                np.asarray(np.real(neff[0]), np.float64),
                np.asarray(v0, np.float64),
                np.asarray(u0, np.float64),
            )

        n = ny * nx
        return jax.pure_callback(
            host,
            (
                jax.ShapeDtypeStruct((), jnp.float64),
                jax.ShapeDtypeStruct((n,), jnp.float64),
                jax.ShapeDtypeStruct((n,), jnp.float64),
            ),
            eps_flat,
        )

    @partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5))
    def n_eff_eps_vector(eps_flat, shape, dx, dy, k0, pol):
        """Differentiable fundamental ``n_eff`` from a flattened permittivity grid.

        Non-symmetric eigenvalue-perturbation adjoint: with right/left
        eigenvectors ``v``/``u`` of ``A(eps) v = lambda v``, ``A^T u = lambda u``,

            d lambda / d eps = (u^T (dA/d eps) v) / (u^T v),
            d n_eff / d eps  = (d lambda / d eps) / (2 k0^2 n_eff).
        """
        neff, _v, _u = _solve_callback(eps_flat, shape, dx, dy, k0, pol)
        return neff

    def _fwd(eps_flat, shape, dx, dy, k0, pol):
        neff, v, u = _solve_callback(eps_flat, shape, dx, dy, k0, pol)
        return neff, (eps_flat, v, u, neff)

    def _bwd(shape, dx, dy, k0, pol, res, g):
        eps_flat, v, u, neff = res
        denom = jnp.sum(u * v)

        def bilinear(e):
            return jnp.sum(u * _apply(e, v, shape, dx, dy, k0, pol))

        dlam_deps = jax.grad(bilinear)(eps_flat) / denom
        dneff_deps = dlam_deps / (2.0 * k0 ** 2 * neff)
        return (g * dneff_deps,)

    n_eff_eps_vector.defvjp(_fwd, _bwd)

else:  # pragma: no cover - NumPy fallback has no autodiff
    def n_eff_eps_vector(eps_flat, shape, dx, dy, k0, pol):
        raise RuntimeError("n_eff_eps_vector requires JAX. Install photonix[jax].")


# =========================================================================== #
# Full-vector solver (Yee-grid FDFD formulation)
# --------------------------------------------------------------------------- #
# The semivectorial solver keeps only the dominant polarization. The true high-
# contrast strip mode is hybrid: it has both Ex and Ey, with bend loss and
# polarization rotation that need the *full* vector field. This solver discretizes
# the first-order curl equations on a staggered (Yee) grid -- the same staggering
# the FDFD solver already uses -- which enforces the dielectric-interface
# continuity conditions automatically through the grid offset plus the discrete
# adjoint relation ``DH = -DE^T`` (no hand-rolled interface coefficients). The
# transverse-E eigenproblem is ``Omega [Ex; Ey] = -n_eff^2 [Ex; Ey]`` with
# ``Omega = P @ Q`` (Rumpf FDFD mode formulation, non-magnetic ``UR = I``).
#
# Validated on the canonical 500x220 nm SOI strip: fundamental ``n_eff ~ 2.45``
# (literature full-vector value), TM0 ~ 1.81, fundamental ~98% Ex-polarized --
# below the semivectorial 2.485 and the scalar 2.611, as it must be.
# =========================================================================== #
def _ddx_fwd(n, h):
    """Square E-to-H derivative for the equal-size staggered Yee field spaces.

    Unlike the scalar cell-to-face gradients in :mod:`photonix.em.eme` and
    :mod:`photonix.em.fdfd`, this curl block intentionally has one exterior row.
    Its paired H-to-E derivative is ``-_ddx_fwd(...).T`` and supplies the
    complementary boundary on the dual grid. Replacing this block by the scalar
    ``(n + 1, n)`` operator would make the P/Q component spaces incompatible;
    doing so correctly requires a full component-staggered grid refactor.
    """
    e = np.ones(n)
    return sp.diags([-e, e[:-1]], [0, 1], format="csc") / h


def _assemble_fullvector(eps, dx, dy, k0):
    """Sparse full-vector operator ``Omega = P @ Q`` (CSC); eigenvalues ``-n_eff^2``.

    **Approximation (C1)**: ``erxx``, ``eryy``, and ``erzz_inv`` are all built
    from the same arithmetic subpixel-averaged permittivity array.  On a Yee
    grid ``Ex``, ``Ey`` and ``Ez`` sit at three different staggered locations,
    and the field component *normal* to a dielectric interface should use
    harmonic (inverse) averaging to recover clean second-order convergence.
    With the current scalar treatment the observed convergence order at
    high-contrast interfaces is ~1.73 instead of 2.  Implementing proper
    anisotropic subpixel smoothing (à la Farjadpour et al.) would fix this.
    """
    ny, nx = eps.shape
    n = ny * nx
    dex = sp.kron(sp.identity(ny), _ddx_fwd(nx, dx)) / k0
    dey = sp.kron(_ddx_fwd(ny, dy), sp.identity(nx)) / k0
    dhx = -dex.T.tocsc()
    dhy = -dey.T.tocsc()
    er = np.asarray(eps, float).reshape(-1)
    erxx = sp.diags(er)
    eryy = sp.diags(er)
    erzz_inv = sp.diags(1.0 / er)
    ident = sp.identity(n)
    P = sp.bmat(
        [[dex @ erzz_inv @ dhy, -(dex @ erzz_inv @ dhx) - ident],
         [dey @ erzz_inv @ dhy + ident, -(dey @ erzz_inv @ dhx)]],
        format="csc",
    )
    Q = sp.bmat(
        [[dhx @ dey, -(dhx @ dex) - eryy],
         [dhy @ dey + erxx, -(dhy @ dex)]],
        format="csc",
    )
    return (P @ Q).tocsc()


def _apply_fullvector(eps_flat, field_flat, shape, dx, dy, k0):
    """Differentiable matvec ``Omega(eps) @ [Ex; Ey]`` (shift form, no sparse matrix)."""
    ny, nx = shape
    n = ny * nx
    er = eps_flat

    def dex(f):
        f2 = f.reshape(ny, nx)
        fp = xp.pad(f2, ((0, 0), (0, 1)), mode="constant")
        return ((fp[:, 1:] - f2) / (k0 * dx)).reshape(-1)

    def dey(f):
        f2 = f.reshape(ny, nx)
        fp = xp.pad(f2, ((0, 1), (0, 0)), mode="constant")
        return ((fp[1:, :] - f2) / (k0 * dy)).reshape(-1)

    def dhx(f):
        f2 = f.reshape(ny, nx)
        fp = xp.pad(f2, ((0, 0), (1, 0)), mode="constant")
        return ((f2 - fp[:, :-1]) / (k0 * dx)).reshape(-1)

    def dhy(f):
        f2 = f.reshape(ny, nx)
        fp = xp.pad(f2, ((1, 0), (0, 0)), mode="constant")
        return ((f2 - fp[:-1, :]) / (k0 * dy)).reshape(-1)

    ex = field_flat[:n]
    ey = field_flat[n:]
    qa = dhx(dey(ex)) - dhx(dex(ey)) - er * ey
    qb = dhy(dey(ex)) + er * ex - dhy(dex(ey))
    zi = 1.0 / er
    pa = dex(zi * dhy(qa)) - dex(zi * dhx(qb)) - qb
    pb = dey(zi * dhy(qa)) + qa - dey(zi * dhx(qb))
    return xp.concatenate([pa, pb])


def _solve_fullvector(eps, dx, dy, k0, num_modes, want_left=False):
    """Fundamental full-vector eigenpair(s) with polarization fractions."""
    A = _assemble_fullvector(eps, dx, dy, k0)
    ny, nx = eps.shape
    n = ny * nx
    nmax2 = float(np.asarray(eps).max())
    sigma = -nmax2 * 1.0001
    k = int(min(max(num_modes, 1), A.shape[0] - 2))
    vals, vecs = spla.eigs(A, k=k, sigma=sigma, which="LM")
    neff2 = -vals.astype(complex)
    scale = np.maximum(1.0, np.abs(np.real(neff2)))
    if np.any(np.abs(np.imag(neff2)) > 1e-7 * scale):
        raise RuntimeError("lossless full-vector solve returned materially complex eigenvalues")
    neff2 = np.real(neff2).astype(complex)
    order = np.argsort(np.real(neff2))[::-1]
    vals, vecs, neff2 = vals[order], vecs[:, order], neff2[order]
    neff = np.sqrt(neff2)
    neff = np.where(np.imag(neff) > 0, -neff, neff)

    fields, te_frac = [], []
    for i in range(vecs.shape[1]):
        ex = vecs[:n, i]
        ey = vecs[n:, i]
        fx = float(np.sum(np.abs(ex) ** 2))
        fy = float(np.sum(np.abs(ey) ** 2))
        te_frac.append(fx / (fx + fy))
        dom = ex if fx >= fy else ey
        dom = np.real(dom).reshape(ny, nx)
        flat = dom.reshape(-1)
        if flat[np.argmax(np.abs(flat))] < 0:
            dom = -dom
        fields.append(dom)
    fields = np.array(fields)
    te_frac = np.array(te_frac)

    v0 = np.real(vecs[:, 0])
    v0 = v0 / np.sqrt(np.sum(v0 ** 2))
    if not want_left:
        return neff[:num_modes], fields[:num_modes], te_frac[:num_modes], v0

    lam0 = complex(vals[0])
    lvals, lvecs = spla.eigs(A.T.tocsc(), k=1, sigma=lam0, which="LM")
    u0 = np.real(lvecs[:, 0])
    return neff[:num_modes], fields[:num_modes], te_frac[:num_modes], v0, u0, float(np.real(lam0))


def solve_modes_fullvector(
    *,
    wl: float = 1.55,
    width: float = 0.5,
    thickness: float = 0.22,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    num_modes: int = 1,
    resolution: int = 40,
    margin: float = 1.5,
    eps: np.ndarray | None = None,
    grid: tuple | None = None,
) -> VectorModeData:
    """Solve full-vector hybrid modes of a cross-section.

    Unlike :func:`solve_modes_vector`, the polarization is *not* fixed: each mode
    carries both ``Ex`` and ``Ey`` and is labelled by ``te_fraction``.

    Notes
    -----
    The operator uses a single arithmetic subpixel-averaged permittivity for all
    tensor components (``erxx = eryy = diag(eps)``).  On a Yee grid the field
    component normal to an interface should use harmonic averaging; without it
    the convergence order at high-contrast interfaces is ~1.73 instead of 2
    (see PHYSICS_AUDIT §C1 and ``_assemble_fullvector``).

    Examples
    --------
    >>> r = solve_modes_fullvector(width=0.5, thickness=0.22, resolution=30)
    >>> 1.444 < r.neff0 < 3.4757
    True
    """
    if (not isinstance(num_modes, (int, np.integer))
            or isinstance(num_modes, (bool, np.bool_)) or num_modes <= 0):
        raise ValueError("num_modes must be a positive integer")
    if not np.isfinite(wl) or wl <= 0:
        raise ValueError("wl must be positive and finite")
    eps_user = eps is not None
    if eps is None:
        from .geometry import rectangular_waveguide

        cs = rectangular_waveguide(
            width=width, thickness=thickness, n_core=n_core, n_clad=n_clad,
            margin=margin, resolution=resolution,
        )
        eps, x, y, dx, dy = cs.eps, cs.x, cs.y, cs.dx, cs.dy
    else:
        from .geometry import _validate_eps_grid

        eps, x, y, dx, dy = _validate_eps_grid(
            eps, grid, where="solve_modes_fullvector"
        )
    from .geometry import as_real_eps

    k0 = 2.0 * np.pi / wl
    neff, fields, te_frac, _ = _solve_fullvector(
        as_real_eps(eps, where="solve_modes_fullvector"), dx, dy, k0, num_modes
    )
    n_clad_eff = _exterior_index(eps) if eps_user else n_clad
    guided = np.real(neff) > n_clad_eff
    return VectorModeData(
        n_eff=neff, fields=fields, x=np.asarray(x), y=np.asarray(y),
        wl=wl, polarization="full", te_fraction=te_frac, guided=guided,
    )


def n_eff_fullvector(
    *,
    wl: float = 1.55,
    width: float = 0.5,
    thickness: float = 0.22,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    resolution: int = 40,
    margin: float = 1.5,
    richardson: bool = False,
) -> float:
    """Fundamental full-vector effective index.

    Richardson extrapolation is **off by default** (see PHYSICS_AUDIT §B2/D1).
    All 2-D solvers (``n_eff``, ``n_eff_vector``, ``n_eff_fullvector``) now
    share the same default.
    """
    def _ne(res):
        return solve_modes_fullvector(
            wl=wl, width=width, thickness=thickness, n_core=n_core, n_clad=n_clad,
            num_modes=1, resolution=res, margin=margin,
        ).neff0

    if not richardson:
        return _ne(resolution)
    n_c, n_f = _ne(resolution), _ne(2 * resolution)
    return (4.0 * n_f - n_c) / 3.0


if HAS_JAX:

    def _solve_callback_fv(eps_flat, shape, dx, dy, k0):
        ny, nx = shape

        def host(e):
            eps = np.asarray(e, float).reshape(ny, nx)
            neff, _f, _t, v0, u0, _lam = _solve_fullvector(eps, dx, dy, k0, 1, want_left=True)
            return (
                np.asarray(np.real(neff[0]), np.float64),
                np.asarray(v0, np.float64),
                np.asarray(u0, np.float64),
            )

        m = 2 * ny * nx
        return jax.pure_callback(
            host,
            (
                jax.ShapeDtypeStruct((), jnp.float64),
                jax.ShapeDtypeStruct((m,), jnp.float64),
                jax.ShapeDtypeStruct((m,), jnp.float64),
            ),
            eps_flat,
        )

    @partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
    def n_eff_eps_fullvector(eps_flat, shape, dx, dy, k0):
        """Differentiable fundamental full-vector ``n_eff`` from a permittivity grid.

        Same non-symmetric eigenvalue-perturbation adjoint as the semivectorial
        case, but the operator eigenvalue is ``lambda = -n_eff^2``, so

            d n_eff / d eps = (u^T (dOmega/d eps) v)/(u^T v) * (-1/(2 n_eff)).
        """
        neff, _v, _u = _solve_callback_fv(eps_flat, shape, dx, dy, k0)
        return neff

    def _fwd_fv(eps_flat, shape, dx, dy, k0):
        neff, v, u = _solve_callback_fv(eps_flat, shape, dx, dy, k0)
        return neff, (eps_flat, v, u, neff)

    def _bwd_fv(shape, dx, dy, k0, res, g):
        eps_flat, v, u, neff = res
        denom = jnp.sum(u * v)

        def bilinear(e):
            return jnp.sum(u * _apply_fullvector(e, v, shape, dx, dy, k0))

        dlam_deps = jax.grad(bilinear)(eps_flat) / denom
        dneff_deps = dlam_deps * (-1.0 / (2.0 * neff))
        return (g * dneff_deps,)

    n_eff_eps_fullvector.defvjp(_fwd_fv, _bwd_fv)

else:  # pragma: no cover - NumPy fallback has no autodiff
    def n_eff_eps_fullvector(eps_flat, shape, dx, dy, k0):
        raise RuntimeError("n_eff_eps_fullvector requires JAX. Install photonix[jax].")


# =========================================================================== #
# Stretched-coordinate PML + bend (radiation) loss
# --------------------------------------------------------------------------- #
# Reusing the same complex-coordinate-stretching idea as the FDFD PML, the Yee
# derivatives are scaled by 1/s(u) inside absorbing layers at the domain edges.
# The operator becomes complex and ``n_eff`` acquires an imaginary part = modal
# loss. Bends are handled by the conformal map ``n -> n (1 + x/R)`` (x measured
# outward from the guide centre), which turns a curved guide into a straight
# graded-index one whose leaky mode radiates past the caustic at
# ``x_c ~ R (n_eff/n_clad - 1)`` -- so the outer window must reach beyond it.
#
# Validated: a straight guide with PML keeps ``Im(n_eff) ~ 1e-8`` (non-
# perturbing); bend loss rises monotonically as the radius tightens
# (~1e-4 -> ~6e-3 dB/90deg over R = 2.0 -> 1.0 um for a 500x220 strip), with the
# physical mode selected by overlap with the straight fundamental.
# =========================================================================== #
@dataclass
class BendMode:
    """Result of a bent-waveguide (or straight, ``bend_radius=None``) solve."""

    n_eff: complex
    loss_db_per_90deg: float
    overlap: float
    bend_radius: float | None


def _pml_stretch(coord, k0, t_pml, smax, m=3, bounds=None):
    """Stretched-coordinate PML factor ``s(u) = 1 - i smax (d/t)^m`` at both ends."""
    edge_lo, edge_hi = (coord.min(), coord.max()) if bounds is None else bounds
    lo = edge_lo + t_pml
    hi = edge_hi - t_pml
    dl = np.clip((lo - coord) / t_pml, 0.0, None)
    dh = np.clip((coord - hi) / t_pml, 0.0, None)
    return 1.0 - 1j * smax * (dl ** m + dh ** m) / k0


def _assemble_fullvector_pml(eps, dx, dy, k0, sx, sy):
    """Complex full-vector operator with staggered ``(integer, half)`` stretches."""
    ny, nx = eps.shape
    n = ny * nx
    dex0 = sp.kron(sp.identity(ny), _ddx_fwd(nx, dx)) / k0
    dey0 = sp.kron(_ddx_fwd(ny, dy), sp.identity(nx)) / k0
    sx_integer, sx_half = sx
    sy_integer, sy_half = sy
    sxi = sp.diags(np.tile(1.0 / sx_integer, ny))
    sxh = sp.diags(np.tile(1.0 / sx_half, ny))
    syi = sp.diags(np.repeat(1.0 / sy_integer, nx))
    syh = sp.diags(np.repeat(1.0 / sy_half, nx))
    # Forward E->H differences live on half cells; their transpose H->E
    # differences land on integer cells. Sharing a stretch misregisters the PML.
    dex = sxh @ dex0
    dhx = sxi @ (-dex0.T.tocsc())
    dey = syh @ dey0
    dhy = syi @ (-dey0.T.tocsc())
    er = np.asarray(eps).reshape(-1).astype(complex)
    erxx = sp.diags(er)
    eryy = sp.diags(er)
    erzz_inv = sp.diags(1.0 / er)
    ident = sp.identity(n)
    P = sp.bmat(
        [[dex @ erzz_inv @ dhy, -(dex @ erzz_inv @ dhx) - ident],
         [dey @ erzz_inv @ dhy + ident, -(dey @ erzz_inv @ dhx)]],
        format="csc",
    )
    Q = sp.bmat(
        [[dhx @ dey, -(dhx @ dex) - eryy],
         [dhy @ dey + erxx, -(dhy @ dex)]],
        format="csc",
    )
    return (P @ Q).tocsc()


def _bend_grid(width, thickness, bend_radius, n_core, n_clad, resolution,
               t_pml, inner, n_guess):
    """Asymmetric cross-section reaching beyond the outer radiation caustic.

    ``inner`` is the **cladding gap** between the core's inner edge and the
    start of the inner PML, so the absorber never overlaps the core (an
    overlapping PML fakes ~1e-5 modal loss on a lossless straight guide).
    """
    x_in = width / 2 + inner + t_pml
    if bend_radius is None:
        x_out = width / 2 + 1.0 + t_pml
    else:
        # The equivalent-index map is singular at x=-R.  Keep the complete
        # innermost grid cell on the physical r=R+x>0 side of that singularity;
        # otherwise squaring n*(1+x/R) creates an unphysical mirrored medium.
        min_safe_radius = x_in + 0.5 / resolution
        if bend_radius <= min_safe_radius:
            raise ValueError(
                f"bend_radius ({bend_radius}) is too small for the inward domain: "
                f"it must exceed width/2 + inner + pml_thickness + half a cell "
                f"({min_safe_radius:.6g}) so the grid does not cross x=-R"
            )
        x_caustic = bend_radius * (n_guess / n_clad - 1.0)
        x_out = x_caustic + 0.6 + t_pml
    x = np.arange(-x_in, x_out + 1e-9, 1.0 / resolution)
    y = np.arange(-(thickness / 2 + 0.8 + t_pml), thickness / 2 + 0.8 + t_pml + 1e-9,
                  1.0 / resolution)
    xx, yy = np.meshgrid(x, y)
    core = (np.abs(yy) <= thickness / 2) & (xx >= -width / 2) & (xx <= width / 2)
    n_straight = np.where(core, n_core, n_clad)
    n_bent = n_straight * (1.0 + xx / bend_radius) if bend_radius is not None else n_straight
    inx = (x > x.min() + t_pml) & (x < x.max() - t_pml)
    iny = (y > y.min() + t_pml) & (y < y.max() - t_pml)
    mask2 = np.tile((iny[:, None] & inx[None, :]).reshape(-1), 2)
    return (n_bent ** 2).astype(float), (n_straight ** 2).astype(float), x, y, mask2


def _fundamental_pml(eps, dx, dy, k0, sx, sy, n_guess, num_modes):
    A = _assemble_fullvector_pml(eps, dx, dy, k0, sx, sy)
    sigma = -(n_guess ** 2) * (1.0 + 1e-7j)  # tiny offset avoids exact-singular shift
    k = int(min(max(num_modes, 1), A.shape[0] - 2))
    vals, vecs = spla.eigs(A, k=k, sigma=sigma, which="LM")
    neff = np.sqrt(-vals + 0j)
    neff = np.where(neff.real < 0, -neff, neff)
    return neff, vecs


def bend_loss_fullvector(
    *,
    width: float = 0.5,
    thickness: float = 0.22,
    bend_radius: float | None = 2.0,
    wl: float = 1.55,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    resolution: int = 34,
    pml: tuple = (0.6, 4.0),
    inner: float = 0.8,
    n_guess: float = 2.45,
    num_modes: int = 8,
) -> BendMode:
    """Full-vector bend (radiation) loss via conformal map + stretched-coord PML.

    ``bend_radius=None`` solves the straight guide with PML (a useful check: the
    loss should be ~0). Otherwise returns the complex ``n_eff`` and the loss in
    dB per 90-degree bend, with the physical leaky mode selected by overlap with
    the straight fundamental (rejecting spurious PML modes).

    Parameters of note: ``inner`` is the cladding gap (µm) between the core's
    inner edge and the inner PML -- the grid is built so the PML never overlaps
    the core.

    Approximations (know before using for design sign-off):

    * The bend is handled by the scalar equivalent-index map ``n -> n(1+x/R)``.
      This is the standard first-order conformal treatment; a rigorous
      high-contrast full-vector bend uses the anisotropic *transformed material
      tensors* (Shyroki, arXiv:physics/0605002). Expect the absolute loss to be
      approximate (trends/monotonicity are reliable).
    * ``loss_db_per_90deg`` is computed from ``|Im(n_eff)|`` because the leaky
      root's imaginary sign depends on the eigensolver branch; the *signed*
      complex ``n_eff`` is preserved on the returned :class:`BendMode` so callers
      can inspect the branch themselves.

    Examples
    --------
    >>> m = bend_loss_fullvector(bend_radius=None, resolution=24)
    >>> abs(m.n_eff.imag) < 1e-6
    True
    """
    for name, value in (
        ("width", width), ("thickness", thickness), ("wl", wl),
        ("n_core", n_core), ("n_clad", n_clad), ("resolution", resolution),
        ("n_guess", n_guess),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if n_core <= n_clad:
        raise ValueError("n_core must be greater than n_clad")
    if bend_radius is not None and (not np.isfinite(bend_radius) or bend_radius <= 0):
        raise ValueError("bend_radius must be positive and finite, or None")
    if not isinstance(pml, (tuple, list)) or len(pml) != 2:
        raise ValueError("pml must be a two-item (thickness, strength) tuple")
    t_pml, smax = pml
    if not np.isfinite(t_pml) or t_pml <= 0:
        raise ValueError("PML thickness must be positive and finite")
    if not np.isfinite(smax) or smax < 0:
        raise ValueError("PML strength must be non-negative and finite")
    if not np.isfinite(inner) or inner < 0:
        raise ValueError("inner must be non-negative and finite")
    if (not isinstance(num_modes, (int, np.integer))
            or isinstance(num_modes, (bool, np.bool_)) or num_modes <= 0):
        raise ValueError("num_modes must be a positive integer")

    k0 = 2.0 * np.pi / wl
    dx = dy = 1.0 / resolution
    eps_b, eps_s, x, y, mask2 = _bend_grid(
        width, thickness, bend_radius, n_core, n_clad, resolution, t_pml, inner, n_guess
    )
    x_bounds = (x[0] - 0.5 * dx, x[-1] + 0.5 * dx)
    y_bounds = (y[0] - 0.5 * dy, y[-1] + 0.5 * dy)
    sx = (
        _pml_stretch(x, k0, t_pml, smax, bounds=x_bounds),
        _pml_stretch(x + 0.5 * dx, k0, t_pml, smax, bounds=x_bounds),
    )
    sy = (
        _pml_stretch(y, k0, t_pml, smax, bounds=y_bounds),
        _pml_stretch(y + 0.5 * dy, k0, t_pml, smax, bounds=y_bounds),
    )

    # straight reference fundamental on the same grid
    nes, ves = _fundamental_pml(eps_s, dx, dy, k0, sx, sy, n_guess, num_modes)
    iref = max((i for i in range(len(nes)) if n_clad < nes[i].real < n_core),
               key=lambda i: nes[i].real)
    vref = ves[:, iref] * mask2
    vref = vref / (np.linalg.norm(vref) + 1e-30)

    if bend_radius is None:
        nb = nes[iref]
        ov = 1.0
    else:
        neb, veb = _fundamental_pml(eps_b, dx, dy, k0, sx, sy, n_guess, num_modes)
        best, bi = -1.0, iref
        for i in range(veb.shape[1]):
            if not (n_clad < neb[i].real < n_core):
                continue
            vi = veb[:, i] * mask2
            nv = np.linalg.norm(vi)
            if nv == 0:
                continue
            ov_i = abs(np.vdot(vref, vi / nv))
            if ov_i > best:
                best, bi = ov_i, i
        nb, ov = neb[bi], best

    ni = abs(nb.imag)
    arc = np.pi * (bend_radius if bend_radius is not None else 1.0) / 2.0
    loss_db = 4.343 * (2.0 * k0 * ni) * arc
    return BendMode(
        n_eff=complex(nb),
        loss_db_per_90deg=float(loss_db if bend_radius is not None else 0.0),
        overlap=float(ov),
        bend_radius=bend_radius,
    )


# =========================================================================== #
# 2-D vectorial transverse fields + bi-orthogonal power overlap
# --------------------------------------------------------------------------- #
# Foundation for a future 2-D hybrid EME. The full-vector operator solves for the
# transverse electric field ``e = [Ex; Ey]``; the companion operator ``Q`` maps it
# to the transverse magnetic field ``h = [Hx; Hy] = Q @ e``. The modes are then
# bi-orthonormal under the *unconjugated* reciprocity overlap
#
#     <a|b> = integral (Ex_a Hy_b - Ey_a Hx_b) dA   ->  delta_ab   (validated to ~1e-14)
#
# (equal to time-averaged Poynting power only for real lossless-mode fields; see
# ``power_overlap``).
#
# This bi-orthogonality (and the transparent self-interface it implies) is the
# prerequisite for vectorial mode-matching. NOTE: a *reciprocal* multi-section
# cascade interface on this basis is not yet implemented -- a truncated-basis
# junction built naively from these overlaps is energy-bounded and has a
# machine-precision transparent limit, but is not reciprocal, so the full 2-D
# hybrid EME cascade remains future work. These tools (modes + overlap) are
# correct and useful on their own (e.g. coupling coefficients, mode overlaps).
# =========================================================================== #
def _pq_operators(eps, dx, dy, k0):
    """The Rumpf ``P`` and ``Q`` blocks (``Omega = P @ Q``); ``Q`` maps E_t -> H_t."""
    ny, nx = eps.shape
    n = ny * nx
    dex = sp.kron(sp.identity(ny), _ddx_fwd(nx, dx)) / k0
    dey = sp.kron(_ddx_fwd(ny, dy), sp.identity(nx)) / k0
    dhx = -dex.T.tocsc()
    dhy = -dey.T.tocsc()
    er = np.asarray(eps, float).reshape(-1)
    erxx = sp.diags(er)
    eryy = sp.diags(er)
    erzz_inv = sp.diags(1.0 / er)
    ident = sp.identity(n)
    P = sp.bmat(
        [[dex @ erzz_inv @ dhy, -(dex @ erzz_inv @ dhx) - ident],
         [dey @ erzz_inv @ dhy + ident, -(dey @ erzz_inv @ dhx)]],
        format="csc",
    )
    Q = sp.bmat(
        [[dhx @ dey, -(dhx @ dex) - eryy],
         [dhy @ dey + erxx, -(dhy @ dex)]],
        format="csc",
    )
    return P, Q


def fullvector_transverse_fields(eps, dx, dy, k0, num_modes=1):
    """Bi-orthonormal 2-D full-vector modes with transverse E and H fields.

    Returns ``(neff, Et, Ht)`` where ``Et``/``Ht`` are ``(2*ny*nx, num_modes)``
    arrays stacked as ``[Ex; Ey]`` / ``[Hx; Hy]``, normalized so the
    **unconjugated** (reciprocity-based) overlap
    ``integral (Ex Hy - Ey Hx) dA = 1`` for each mode. The modes are then
    bi-orthonormal under :func:`power_overlap` (validated to ~1e-14).

    Note: this is the pairing mode-matching theory requires -- *not* the
    time-averaged Poynting power, which conjugates one field
    (``Re integral (E x H*) . z dA``). The two coincide only when the mode
    fields are real up to a global phase (guided modes of a lossless, PML-free
    section). Do not use the unconjugated overlap as physical power for complex
    (leaky/PML) fields -- a global phase rotation changes it.
    """
    if (not isinstance(num_modes, (int, np.integer))
            or isinstance(num_modes, (bool, np.bool_)) or num_modes <= 0):
        raise ValueError("num_modes must be a positive integer")
    P, Q = _pq_operators(np.asarray(eps, float), dx, dy, k0)
    Omega = (P @ Q).tocsc()
    nmax2 = float(np.asarray(eps).max())
    k = int(min(max(num_modes, 1), Omega.shape[0] - 2))
    vals, vecs = spla.eigs(Omega, k=k, sigma=-nmax2 * 1.0001, which="LM")
    neff = np.sqrt(-vals + 0j)
    neff = np.where(neff.real < 0, -neff, neff)
    order = np.argsort(-neff.real)
    neff, Et = neff[order], vecs[:, order]
    if np.any(np.abs(neff) < 1e-14):
        raise ValueError("cannot reconstruct transverse H for a mode at cutoff")
    # The first-order eigen-equation is Q E_t = n_eff H_t.  Dividing by the
    # modal index is essential for the physical wave impedance across sections.
    Ht = (Q @ Et) / neff[None, :]
    n = (np.asarray(eps).size)
    dA = dx * dy
    for kk in range(Et.shape[1]):
        power = np.sum(Et[:n, kk] * Ht[n:, kk] - Et[n:, kk] * Ht[:n, kk]) * dA
        s = np.sqrt(power)
        Et[:, kk] /= s
        Ht[:, kk] /= s
    return neff[:num_modes], Et[:, :num_modes], Ht[:, :num_modes]


def power_overlap(EtA, HtB, dA):
    """Unconjugated modal overlap ``M[m,k] = integral (Ex_k^A Hy_m^B - Ey_k^A Hx_m^B) dA``.

    With ``EtA``, ``HtB`` from :func:`fullvector_transverse_fields` on the *same*
    section, ``M`` is the identity (bi-orthonormality). Across sections it is the
    vectorial mode-overlap used by vectorial mode-matching.

    This is the **reciprocity** product (no complex conjugation) -- the correct
    bi-orthogonal pairing for mode-matching, including lossy/leaky modes. It is
    *not* time-averaged Poynting power for complex fields: a global e^{j phi}
    phase on a mode multiplies it by e^{2j phi}, while physical power
    (``Re integral E x H* dA``) is phase-invariant. For the real fields of
    guided lossless modes the two agree, which is the sense in which the
    normalization here is "power"-like.
    """
    n = EtA.shape[0] // 2
    ExA, EyA = EtA[:n], EtA[n:]
    HxB, HyB = HtB[:n], HtB[n:]
    return (HyB.T @ ExA - HxB.T @ EyA) * dA
