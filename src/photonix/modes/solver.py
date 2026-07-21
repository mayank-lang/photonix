"""Scalar finite-difference frequency-domain (FDFD) waveguide mode solver.

Solves the 2-D scalar Helmholtz eigenproblem for a waveguide cross-section::

    (d^2/dx^2 + d^2/dy^2 + k0^2 eps(x, y)) psi = beta^2 psi

for the most-confined guided modes. The effective index is
``n_eff = beta / k0`` and the field ``psi`` is the (scalar) mode profile.

The scalar model is fast, robust, and exact for the scalar equation (validated
against the analytic scalar-slab limit). It is an excellent approximation for
low/medium index-contrast guides and a good first-order estimate for high-
contrast strips; a full vectorial solver can be layered on the same interface.

This is the one place in photonix where SciPy's sparse eigensolver is used; the
public results are returned as plain arrays/floats so they feed component models
directly (e.g. ``photonix.components.straight(neff=lambda wl: ...)``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from photonix.core.constants import C0  # noqa: F401  (re-exported context)

from .geometry import CrossSection, rectangular_waveguide

__all__ = ["ModeResult", "solve_modes", "n_eff", "group_index", "overlap"]


@dataclass
class ModeResult:
    """Result of a mode solve.

    Attributes
    ----------
    n_eff : ndarray
        Effective indices of the computed modes, descending.
    fields : ndarray
        Mode field profiles, shape ``(num_modes, ny, nx)``.
    x, y : ndarray
        Coordinate arrays (µm).
    wl : float
        Wavelength of the solve (µm).
    """

    n_eff: np.ndarray
    fields: np.ndarray
    x: np.ndarray
    y: np.ndarray
    wl: float

    @property
    def neff0(self) -> float:
        """Fundamental-mode effective index."""
        return float(self.n_eff[0])


def _build_operator(cs: CrossSection, k0: float):
    ny, nx = cs.eps.shape
    dx, dy = cs.dx, cs.dy
    n = nx * ny

    # 1-D second-difference operators with Dirichlet BC.
    ex = np.ones(nx)
    Lx = sp.diags([ex[:-1], -2 * ex, ex[:-1]], [-1, 0, 1]) / dx**2
    ey = np.ones(ny)
    Ly = sp.diags([ey[:-1], -2 * ey, ey[:-1]], [-1, 0, 1]) / dy**2

    Ix = sp.identity(nx)
    Iy = sp.identity(ny)
    lap = sp.kron(Iy, Lx) + sp.kron(Ly, Ix)  # 2-D Laplacian (row-major y,x)

    eps_diag = sp.diags(cs.eps.reshape(n) * k0**2)
    return (lap + eps_diag).tocsr(), (ny, nx)


def solve_modes(
    *,
    wl: float = 1.55,
    width: float = 0.5,
    thickness: float = 0.22,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    num_modes: int = 1,
    resolution: int = 40,
    cross_section: CrossSection | None = None,
) -> ModeResult:
    """Solve for the first ``num_modes`` guided modes of a waveguide.

    Parameters
    ----------
    wl : float
        Wavelength in µm.
    width, thickness, n_core, n_clad, resolution
        Geometry parameters (ignored if ``cross_section`` is given).
    num_modes : int
        Number of modes to return (descending n_eff).
    cross_section : CrossSection, optional
        Provide a custom cross-section instead of a rectangle.

    Returns
    -------
    ModeResult

    Examples
    --------
    >>> r = solve_modes(wl=1.55, width=0.5, thickness=0.22, resolution=25)
    >>> 1.444 < r.neff0 < 3.4757
    True
    """
    cs = cross_section or rectangular_waveguide(
        width=width, thickness=thickness, n_core=n_core, n_clad=n_clad, resolution=resolution
    )
    k0 = 2.0 * np.pi / wl
    A, (ny, nx) = _build_operator(cs, k0)

    # Largest algebraic eigenvalues of A = beta^2 -> most-confined modes.
    k = min(num_modes, A.shape[0] - 2)
    sigma = (max(np.sqrt(cs.eps.max()), 1.0) * k0) ** 2  # shift near the core
    vals, vecs = spla.eigsh(A, k=max(k, 1), sigma=sigma, which="LM")

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    betas = np.sqrt(np.clip(vals, 0.0, None))
    neff = betas / k0

    fields = np.array([vecs[:, i].reshape(ny, nx) for i in range(len(order))])
    # normalize sign so peak is positive
    for i in range(fields.shape[0]):
        if fields[i].ravel()[np.argmax(np.abs(fields[i]))] < 0:
            fields[i] *= -1.0
    return ModeResult(n_eff=neff[:num_modes], fields=fields[:num_modes], x=cs.x, y=cs.y, wl=wl)


def n_eff(
    *,
    wl: float = 1.55,
    width: float = 0.5,
    thickness: float = 0.22,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    resolution: int = 40,
) -> float:
    """Fundamental-mode effective index (convenience wrapper).

    Examples
    --------
    >>> ne = n_eff(wl=1.55, width=0.5, thickness=0.22, resolution=25)
    >>> 1.444 < ne < 3.4757
    True
    """
    return solve_modes(
        wl=wl, width=width, thickness=thickness, n_core=n_core, n_clad=n_clad,
        num_modes=1, resolution=resolution,
    ).neff0


def group_index(
    *,
    wl: float = 1.55,
    dwl: float = 0.005,
    **kwargs,
) -> float:
    """Group index ``n_g = n_eff - wl * d n_eff / d wl`` via central difference.

    Extra keyword args are forwarded to :func:`n_eff`.

    Examples
    --------
    >>> ng = group_index(wl=1.55, width=0.5, thickness=0.22, resolution=22)
    >>> ng > n_eff(wl=1.55, width=0.5, thickness=0.22, resolution=22)
    True
    """
    n_plus = n_eff(wl=wl + dwl, **kwargs)
    n_minus = n_eff(wl=wl - dwl, **kwargs)
    n_mid = n_eff(wl=wl, **kwargs)
    dneff = (n_plus - n_minus) / (2 * dwl)
    return float(n_mid - wl * dneff)


def overlap(field_a, field_b) -> float:
    """Normalized field overlap integral |<a|b>|^2 / (<a|a><b|b>).

    Examples
    --------
    >>> import numpy as np
    >>> f = np.random.rand(10, 10)
    >>> abs(overlap(f, f) - 1.0) < 1e-12
    True
    """
    a = np.asarray(field_a).ravel()
    b = np.asarray(field_b).ravel()
    num = abs(np.vdot(a, b)) ** 2
    den = np.vdot(a, a).real * np.vdot(b, b).real
    return float(num / den)
