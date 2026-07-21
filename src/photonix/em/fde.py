"""Finite-Difference Eigenmode (FDE) waveguide solver.

Computes waveguide cross-section modes by solving the scalar finite-difference
Helmholtz eigenproblem ``A psi = (n_eff k0)^2 psi`` for the most-confined modes.

Accuracy: subpixel permittivity averaging (``photonix.em.geometry``) plus
``richardson=True`` (combining resolutions ``r`` and ``2r`` to cancel the leading
O(h^2) error) reaches <0.1% vs the analytic slab on modest, CPU-friendly grids.

Differentiability: :func:`n_eff_eps` returns ``n_eff`` as a function of the
permittivity grid with an exact analytic gradient (eigenvalue perturbation /
Hellmann-Feynman: ``d n_eff/d eps_k = x_k^2 / (2 n_eff)``) supplied via
``jax.custom_vjp``. This is the adjoint quantity topology optimization needs. The
eigensolve runs on the host through ``jax.pure_callback`` so it stays traceable.

Scope: this is the scalar FDE. The full-vectorial / polarization-resolved
(TE/TM/hybrid) solver is the next EM increment -- see
``docs/DESIGN_EM_SOLVERS.md``. Scalar n_eff slightly overestimates the true
vectorial TE index for high-contrast strips; it is exact in the scalar limit and
validated to <0.1% against the analytic slab.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import scipy.sparse.linalg as spla

from photonix.core.backend import HAS_JAX

from .operators import helmholtz_operator

__all__ = ["ModeData", "solve_modes", "n_eff", "n_eff_eps", "group_index", "slab_neff"]


@dataclass
class ModeData:
    """Result of an FDE solve."""

    n_eff: np.ndarray
    fields: np.ndarray
    x: np.ndarray
    y: np.ndarray
    wl: float

    @property
    def neff0(self) -> float:
        return float(np.real(self.n_eff[0]))


def _solve_eps(eps, dy, dx, k0, num_modes):
    """Core host solve: returns (n_eff array, fields array, normalized v0)."""
    ny, nx = eps.shape
    A = helmholtz_operator(eps, dy, dx, k0)
    n_max = float(np.sqrt(eps.max()))
    sigma = (n_max * k0) ** 2 * 1.0001
    k = min(max(num_modes, 1), A.shape[0] - 2)
    vals, vecs = spla.eigsh(A, k=k, sigma=sigma, which="LM")
    order = np.argsort(np.real(vals))[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    betas = np.sqrt(np.clip(vals, 0.0, None).astype(complex))
    neff = betas / k0
    fields = np.array([vecs[:, i].reshape(ny, nx) for i in range(vecs.shape[1])])
    for i in range(fields.shape[0]):
        flat = fields[i].reshape(-1)
        if flat[np.argmax(np.abs(flat))] < 0:
            fields[i] *= -1.0
    v0 = vecs[:, 0]
    v0 = v0 / np.sqrt(np.sum(v0 ** 2))
    return neff[:num_modes], fields[:num_modes], v0


def _cross_section(width, thickness, n_core, n_clad, resolution, margin):
    from .geometry import rectangular_waveguide

    return rectangular_waveguide(
        width=width, thickness=thickness, n_core=n_core, n_clad=n_clad,
        margin=margin, resolution=resolution,
    )


def solve_modes(
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
) -> ModeData:
    """Solve the first ``num_modes`` guided modes of a waveguide cross-section.

    Examples
    --------
    >>> r = solve_modes(wl=1.55, width=0.5, thickness=0.22, resolution=25)
    >>> 1.444 < r.neff0 < 3.4757
    True
    """
    if eps is None:
        cs = _cross_section(width, thickness, n_core, n_clad, resolution, margin)
        eps, x, y = cs.eps, cs.x, cs.y
        dx, dy = cs.dx, cs.dy
    else:
        x, y = grid if grid is not None else (np.arange(eps.shape[1]), np.arange(eps.shape[0]))
        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
    from .geometry import as_real_eps

    k0 = 2.0 * np.pi / wl
    neff, fields, _ = _solve_eps(as_real_eps(eps, where="solve_modes"), dy, dx, k0, num_modes)
    return ModeData(n_eff=neff, fields=fields, x=np.asarray(x), y=np.asarray(y), wl=wl)


def slab_neff(
    *,
    thickness: float = 0.22,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    wl: float = 1.55,
    resolution: int = 40,
    margin: float = 2.0,
    richardson: bool = True,
) -> float:
    """Fundamental effective index of a 1-D symmetric slab (scalar/TE).

    Same Helmholtz discretization as the 2-D solver, restricted to 1-D, with
    subpixel averaging and optional Richardson extrapolation. Matches the
    closed-form slab solution to <0.1% -- the rigorous accuracy anchor.

    Examples
    --------
    >>> ne = slab_neff(thickness=0.22, resolution=40)
    >>> 1.444 < ne < 3.4757
    True
    """
    import scipy.sparse as sp

    from .geometry import slab_profile

    def _ne(res):
        eps, y = slab_profile(thickness=thickness, n_core=n_core, n_clad=n_clad,
                              margin=margin, resolution=res)
        h = float(y[1] - y[0])
        k0 = 2.0 * np.pi / wl
        n = len(eps)
        e = np.ones(n)
        L = sp.diags([e[:-1], -2.0 * e, e[:-1]], [-1, 0, 1]) / h ** 2
        A = (L + sp.diags(k0 ** 2 * eps)).tocsr()
        val, _ = spla.eigsh(A, k=1, sigma=(n_core * k0) ** 2 * 1.0001, which="LM")
        return float(np.sqrt(val[0]) / k0)

    if not richardson:
        return _ne(resolution)
    n_c, n_f = _ne(resolution), _ne(2 * resolution)
    return (4.0 * n_f - n_c) / 3.0


def n_eff(
    *,
    wl: float = 1.55,
    width: float = 0.5,
    thickness: float = 0.22,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    resolution: int = 40,
    margin: float = 1.5,
    richardson: bool = True,
) -> float:
    """Fundamental-mode effective index (Richardson-extrapolated by default)."""
    def _ne(res):
        return solve_modes(
            wl=wl, width=width, thickness=thickness, n_core=n_core, n_clad=n_clad,
            num_modes=1, resolution=res, margin=margin,
        ).neff0

    if not richardson:
        return _ne(resolution)
    n_coarse = _ne(resolution)
    n_fine = _ne(2 * resolution)
    return (4.0 * n_fine - n_coarse) / 3.0


def group_index(*, wl: float = 1.55, dwl: float = 0.005, **kwargs) -> float:
    """Group index ``n_g = n_eff - wl * d n_eff/d wl`` via central difference."""
    n_p = n_eff(wl=wl + dwl, **kwargs)
    n_m = n_eff(wl=wl - dwl, **kwargs)
    n_0 = n_eff(wl=wl, **kwargs)
    return float(n_0 - wl * (n_p - n_m) / (2 * dwl))


if HAS_JAX:
    import jax
    import jax.numpy as jnp

    def _solve_callback(eps_flat, shape, dy, dx, k0):
        ny, nx = shape

        def host(e):
            eps = np.asarray(e, float).reshape(ny, nx)
            neff, _f, v0 = _solve_eps(eps, dy, dx, k0, 1)
            return (np.asarray(np.real(neff[0]), np.float64), np.asarray(v0, np.float64))

        return jax.pure_callback(
            host,
            (jax.ShapeDtypeStruct((), jnp.float64), jax.ShapeDtypeStruct((ny * nx,), jnp.float64)),
            eps_flat,
        )

    @partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
    def n_eff_eps(eps_flat, shape, dy, dx, k0):
        """Differentiable fundamental ``n_eff`` from a flattened permittivity grid.

        Gradient ``d n_eff/d eps_k = x_k^2 / (2 n_eff)`` (exact eigenvalue
        perturbation of the symmetric scalar operator), supplied analytically.
        """
        neff, _v = _solve_callback(eps_flat, shape, dy, dx, k0)
        return neff

    def _neff_fwd(eps_flat, shape, dy, dx, k0):
        neff, v = _solve_callback(eps_flat, shape, dy, dx, k0)
        return neff, (v, neff)

    def _neff_bwd(shape, dy, dx, k0, res, g):
        v, neff = res
        return (g * (v ** 2) / (2.0 * neff),)

    n_eff_eps.defvjp(_neff_fwd, _neff_bwd)

else:  # pragma: no cover
    def n_eff_eps(eps_flat, shape, dy, dx, k0):
        raise RuntimeError("n_eff_eps requires JAX. Install photonix[jax].")
