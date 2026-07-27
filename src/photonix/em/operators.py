"""Finite-difference operator assembly for the EM solvers.

Builds the sparse 2-D scalar Helmholtz operator used by the FDE mode solver.
Operators are assembled with SciPy sparse matrices; differentiability is provided
separately in :mod:`photonix.em.fde` via analytic adjoints, so these builders
deal in plain NumPy/SciPy.

Grid convention: ``eps`` has shape ``(ny, nx)``, raster-ordered row-major (y
outer, x inner) when flattened, matching ``numpy.reshape``.

The full-vectorial and semivectorial polarization-resolved operators are in
:mod:`photonix.em.fde_vector`; this module provides the scalar Helmholtz
operator used by :mod:`photonix.em.fde`.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["laplacian", "helmholtz_operator"]


def _d2_1d(n: int, h: float) -> sp.csr_matrix:
    """1-D second-difference operator with Dirichlet (zero) boundaries."""
    e = np.ones(n)
    return sp.diags([e[:-1], -2.0 * e, e[:-1]], [-1, 0, 1], format="csr") / h ** 2


def laplacian(ny: int, nx: int, dy: float, dx: float) -> sp.csr_matrix:
    """2-D scalar Laplacian (Dirichlet BC) for a ``(ny, nx)`` grid."""
    Lx = _d2_1d(nx, dx)
    Ly = _d2_1d(ny, dy)
    return sp.kron(sp.identity(ny), Lx) + sp.kron(Ly, sp.identity(nx))


def helmholtz_operator(eps: np.ndarray, dy: float, dx: float, k0: float) -> sp.csr_matrix:
    """Scalar Helmholtz operator ``A = Lap + k0**2 * diag(eps)``.

    Eigenvalues of ``A`` are ``beta**2 = (n_eff * k0)**2``; the largest give the
    most-confined modes. Symmetric, so left == right eigenvectors (which makes the
    eigenvalue-perturbation adjoint in :mod:`photonix.em.fde` exact and cheap).
    """
    ny, nx = eps.shape
    A = laplacian(ny, nx, dy, dx)
    A = A + sp.diags(k0 ** 2 * eps.reshape(-1))
    return A.tocsr()
