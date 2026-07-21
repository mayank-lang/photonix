"""Eigenmode Expansion (EME) propagator.

EME computes the scattering matrix of a z-varying waveguide by slicing it into
piecewise-uniform sections, expanding the field in the guided modes of each
section, mode-matching at the interfaces, propagating the modal phases, and
cascading everything with the Redheffer star product. It is bidirectional (unlike
BPM), so it captures reflections, and length sweeps are cheap (re-cascade only).

This implementation works on 1-D cross-sections (index varies in x; propagation in
z) -- the natural setting for in-plane analysis of planar PIC components (tapers,
MMIs, mode converters). Modes come from the same validated finite-difference
discretization used elsewhere in :mod:`photonix.em`.

Polarization (``polarization="te"`` or ``"tm"``):

* **TE** -- field ``Ey`` (continuous), scalar Helmholtz, modes power-orthonormal
  under ``sum(psi_l psi_m) dx``; overlap ``O_{lk} = <psi_l^B | psi_k^A>``.
* **TM** -- field ``Hy`` (continuous), the generalized eigenproblem
  ``A Hy = beta^2 B Hy`` with ``A = d/dx[(1/eps) d/dx] + k0^2`` and
  ``B = diag(1/eps)``. TM power flows as ``~ (beta/eps) |Hy|^2``, so modes are
  orthonormal under the **1/eps-weighted** inner product and the interface uses
  the same vectorial weight: ``O_{lk} = integral (1/eps_B) Hy_l^B Hy_k^A dx``.
  This is what makes the overlap reciprocal and the cascade energy-conserving for
  the discontinuous-D polarization. Validated: transparent interface to 1e-16,
  energy conserved to ~1e-12, and a smooth taper stays adiabatic with reflection
  at the ~1e-5 discretization floor.

Interface S-matrix (modes orthonormal within a section, amplitudes power-
normalized via ``D = diag(beta)``):

    R_f = (D_A + Oᵀ D_B O)^{-1} (D_A - Oᵀ D_B O)
    T_b = (D_A + Oᵀ D_B O)^{-1} (2 Oᵀ D_B)
    T_f = O (I + R_f)
    R_b = O T_b - I

For identical sections ``O = I`` and the interface is transparent
(R_f = R_b = 0, T = I), as it must be.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from photonix.core.types import SDict

__all__ = ["Section", "slab_modes", "eme_smatrix", "EMEResult"]


@dataclass
class Section:
    """A z-uniform EME section: a 1-D permittivity profile and a length."""

    eps: np.ndarray   # (nx,) permittivity along x
    length: float     # z length (µm)


def slab_modes(eps: np.ndarray, dx: float, wl: float, num_modes: int,
               polarization: str = "te"):
    """Guided modes of a 1-D cross-section: returns ``(betas, fields, weight)``.

    ``betas`` are propagation constants ``n_eff k0`` (descending). ``fields`` are
    real and power-orthonormalized under the polarization inner product, whose
    per-point ``weight`` is also returned (``1`` for TE, ``1/eps`` for TM) for use
    in the interface overlap.
    """
    from .geometry import as_real_eps

    eps = as_real_eps(eps, where="slab_modes/EME")
    k0 = 2.0 * np.pi / wl
    n = len(eps)
    if polarization == "te":
        e = np.ones(n)
        A = (sp.diags([e[:-1], -2.0 * e, e[:-1]], [-1, 0, 1]) / dx**2
             + sp.diags(k0**2 * eps)).tocsr()
        kk = min(num_modes, n - 2)
        vals, vecs = spla.eigsh(A, k=kk, sigma=(np.sqrt(eps.max()) * k0) ** 2 * 1.0001,
                                which="LM")
        weight = np.ones(n)
        B = None
    elif polarization == "tm":
        epsf = 0.5 * (eps[:-1] + eps[1:])          # face permittivity (sharp)
        ap = np.zeros(n)
        am = np.zeros(n)
        ap[:-1] = (1.0 / epsf) / dx**2
        am[1:] = (1.0 / epsf) / dx**2
        main = -(ap + am) + k0**2
        A = sp.diags([ap[:-1], main, ap[:-1]], [-1, 0, 1]).tocsr()
        weight = 1.0 / eps
        B = sp.diags(weight).tocsr()
        kk = min(num_modes, n - 2)
        vals, vecs = spla.eigsh(A, k=kk, M=B, sigma=(np.sqrt(eps.max()) * k0) ** 2 * 1.0001,
                                which="LM")
    else:
        raise ValueError("polarization must be 'te' or 'tm'")

    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    betas = np.sqrt(np.clip(vals, 0.0, None))
    # power-orthonormalize under the (weighted) inner product: sum(w psi^2) dx = 1
    for i in range(vecs.shape[1]):
        nrm = np.sqrt(np.sum(weight * vecs[:, i] ** 2) * dx)
        vecs[:, i] /= nrm
        if vecs[np.argmax(np.abs(vecs[:, i])), i] < 0:
            vecs[:, i] *= -1.0
    return betas[:num_modes], vecs[:, :num_modes], weight


def _interface(betaA, fieldsA, betaB, fieldsB, dx, weightB=None):
    """Bidirectional interface S-blocks (R_f, T_f, T_b, R_b).

    ``weightB`` is the destination-section inner-product weight (``1/eps`` for TM,
    ``None``/ones for TE), making the overlap the vectorial power overlap.
    """
    if weightB is None:
        O = (fieldsB.T @ fieldsA) * dx            # (NB, NA) overlap <B|A>
    else:
        O = (fieldsB.T @ (weightB[:, None] * fieldsA)) * dx
    DA = np.diag(betaA)
    DB = np.diag(betaB)
    K = O.T @ DB @ O                              # NA x NA
    inv = np.linalg.inv(DA + K)
    Rf = inv @ (DA - K)
    Tb = inv @ (2.0 * O.T @ DB)
    Tf = O @ (np.eye(len(betaA)) + Rf)
    Rb = O @ Tb - np.eye(len(betaB))
    # floor avoids a divide-by-zero for non-propagating (beta ~ 0) modes; guided
    # modes (beta > 0) are unaffected. EME here represents guided modes only --
    # radiation-continuum modes would need a PML-discretized cross-section.
    sA = np.sqrt(np.maximum(betaA, 1e-12))
    sB = np.sqrt(np.maximum(betaB, 1e-12))
    Rf = (sA[:, None] * Rf) / sA[None, :]
    Tb = (sA[:, None] * Tb) / sB[None, :]
    Tf = (sB[:, None] * Tf) / sA[None, :]
    Rb = (sB[:, None] * Rb) / sB[None, :]
    return Rf, Tf, Tb, Rb


def _prop(beta, length):
    P = np.diag(np.exp(-1j * beta * length))
    Z = np.zeros_like(P)
    return Z, P, P, Z  # Rf, Tf, Tb, Rb


def _star(S1, S2):
    """Redheffer star product of two S-blocks (Rf, Tf, Tb, Rb)."""
    Rf1, Tf1, Tb1, Rb1 = S1
    Rf2, Tf2, Tb2, Rb2 = S2
    n2 = Rf2.shape[0]
    n1 = Rb1.shape[0]
    X1 = np.linalg.inv(np.eye(n2) - Rf2 @ Rb1)
    X2 = np.linalg.inv(np.eye(n1) - Rb1 @ Rf2)
    Rf = Rf1 + Tb1 @ X1 @ Rf2 @ Tf1
    Tf = Tf2 @ X2 @ Tf1
    Tb = Tb1 @ X1 @ Tb2
    Rb = Rb2 + Tf2 @ X2 @ Rb1 @ Tb2
    return Rf, Tf, Tb, Rb


@dataclass
class EMEResult:
    """Result of an EME propagation."""

    Rf: np.ndarray
    Tf: np.ndarray
    Tb: np.ndarray
    Rb: np.ndarray
    betas_in: np.ndarray
    betas_out: np.ndarray

    def sdict(self, n_in: int = 1, n_out: int = 1) -> SDict:
        """Scattering dict over the first ``n_in``/``n_out`` modes."""
        out: SDict = {}
        for i in range(n_in):
            for j in range(n_out):
                out[(f"in{i}", f"out{j}")] = complex(self.Tf[j, i])
                out[(f"out{j}", f"in{i}")] = complex(self.Tb[i, j])
            for k in range(n_in):
                out[(f"in{i}", f"in{k}")] = complex(self.Rf[k, i])
        return out


def eme_smatrix(sections: list[Section], dx: float, wl: float, num_modes: int = 6,
                polarization: str = "te") -> EMEResult:
    """Cascade EME over ``sections`` and return the total S-matrix.

    ``polarization`` selects TE (default) or the vectorial TM formulation.

    Examples
    --------
    >>> import numpy as np, photonix.em as em
    >>> x = np.linspace(-3, 3, 241); dx = x[1]-x[0]
    >>> eps = np.where(np.abs(x) < 0.25, 3.4757**2, 1.444**2)
    >>> r = em.eme.eme_smatrix([em.eme.Section(eps, 5.0)], dx, 1.55, num_modes=4)
    >>> abs(abs(r.Tf[0, 0]) - 1.0) < 1e-6     # straight WG: lossless
    True
    """
    modes = [slab_modes(s.eps, dx, wl, num_modes, polarization) for s in sections]
    betas0, _f0, _w0 = modes[0]
    S = _prop(betas0, sections[0].length)
    for i in range(1, len(sections)):
        bA, fA, _wA = modes[i - 1]
        bB, fB, wB = modes[i]
        weightB = None if polarization == "te" else wB
        S = _star(S, _interface(bA, fA, bB, fB, dx, weightB))
        S = _star(S, _prop(bB, sections[i].length))
    Rf, Tf, Tb, Rb = S
    return EMEResult(Rf, Tf, Tb, Rb, modes[0][0], modes[-1][0])
