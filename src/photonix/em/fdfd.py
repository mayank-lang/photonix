"""Finite-Difference Frequency-Domain (FDFD) Maxwell solver with adjoint gradients.

Solves the 2-D scalar Helmholtz equation on a Yee-style grid with
stretched-coordinate PML::

    (D_x^pml + D_y^pml + k0^2 eps) E = -i*omega*mu0 * J  ~  A(eps) e = b

A is large, sparse, complex; we factor it with a sparse LU (``scipy``). The
**adjoint** gives the gradient of any field objective w.r.t. every permittivity
pixel from a single extra solve with A^T -- i.e. topology optimization
out of the box (the ceviche approach).

Both polarizations are supported: TE (out-of-plane ``Ez``, scalar Helmholtz) and
TM (out-of-plane ``Hz``, the ``div((1/eps) grad Hz) + k0^2 Hz`` form with face
permittivity). The TM path gives an independent full-wave check of the TM EME
(they agree on step transmission to ~0.3%).

This is the rigorous, geometry-based frequency-domain workhorse: it produces
fields and S-parameters from the actual permittivity distribution and exposes the
adjoint sensitivity that drives inverse design.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from photonix.core.types import SDict

__all__ = ["FDFD", "scpml_stretch", "point_source", "focus_objective", "optimize_focus",
           "waveguide_mode", "mode_source", "waveguide_sparams"]


def scpml_stretch(n: int, dh: float, npml: int, k0: float, m: int = 3, log_R: float = -12.0):
    """Stretched-coordinate PML factors ``s`` at integer and half-integer points.

    ``s = 1 - i*sigma/(omega*eps0)`` graded polynomially over the PML region.
    Returns ``(s_int, s_half)`` complex arrays of length ``n``.
    """
    eta0 = 376.730313668
    sigma_max = -(m + 1) * log_R / (2 * eta0 * npml * dh)

    def sigma(p):  # p in [0, npml] distance into PML (in cells)
        return sigma_max * (p / npml) ** m if npml > 0 else 0.0 * p

    s_int = np.ones(n, dtype=complex)
    s_half = np.ones(n, dtype=complex)
    omega_eps0 = k0 / eta0  # omega*eps0 = k0/eta0 (since omega*eps0 = k0*c*eps0 = k0/eta0)
    for i in range(n):
        # distance into the PML (cells) for integer point i
        d_int = max(npml - i, 0) + max(i - (n - 1 - npml), 0)
        d_half = max(npml - (i + 0.5), 0) + max((i + 0.5) - (n - 1 - npml), 0)
        s_int[i] = 1 - 1j * sigma(d_int) / omega_eps0
        s_half[i] = 1 - 1j * sigma(d_half) / omega_eps0
    return s_int, s_half


def _d_forward(n, h):
    e = np.ones(n)
    return sp.diags([-e, e[:-1]], [0, 1]) / h  # forward difference (Dirichlet outer)


class FDFD:
    """A 2-D scalar FDFD simulation on a permittivity grid ``eps`` (ny, nx)."""

    def __init__(self, eps: np.ndarray, dx: float, dy: float, wl: float, npml: int = 12,
                 polarization: str = "te"):
        from .geometry import as_real_eps

        self.eps = as_real_eps(eps, where="FDFD")
        self.ny, self.nx = self.eps.shape
        self.dx, self.dy, self.wl = dx, dy, wl
        self.k0 = 2 * np.pi / wl
        self.npml = npml
        self.polarization = polarization
        self._A = None
        self._lu = None

    def _operator(self):
        if self.polarization == "tm":
            return self._operator_tm()
        ny, nx, k0 = self.ny, self.nx, self.k0
        sxi, sxh = scpml_stretch(nx, self.dx, self.npml, k0)
        syi, syh = scpml_stretch(ny, self.dy, self.npml, k0)
        Dxf = _d_forward(nx, self.dx)
        Dyf = _d_forward(ny, self.dy)
        # stretched 2nd derivative: diag(1/s_int) Dxb diag(1/s_half) Dxf, Dxb = -Dxf^T
        Axx = sp.diags(1 / sxi) @ (-Dxf.T) @ sp.diags(1 / sxh) @ Dxf
        Ayy = sp.diags(1 / syi) @ (-Dyf.T) @ sp.diags(1 / syh) @ Dyf
        Lap = sp.kron(sp.identity(ny), Axx) + sp.kron(Ayy, sp.identity(nx))
        A = Lap + sp.diags(k0**2 * self.eps.reshape(-1))
        return A.tocsc()

    def _operator_tm(self):
        """TM (Hz) operator: div((1/eps) grad Hz) + k0^2 Hz, with face eps + PML."""
        ny, nx, k0 = self.ny, self.nx, self.k0
        sxi, sxh = scpml_stretch(nx, self.dx, self.npml, k0)
        syi, syh = scpml_stretch(ny, self.dy, self.npml, k0)
        Dxf = sp.kron(sp.identity(ny), _d_forward(nx, self.dx))
        Dyf = sp.kron(_d_forward(ny, self.dy), sp.identity(nx))
        Dxb = -Dxf.T.tocsc()
        Dyb = -Dyf.T.tocsc()
        eps = self.eps
        ex = 0.5 * (eps + np.roll(eps, -1, axis=1))
        ex[:, -1] = eps[:, -1]
        ey = 0.5 * (eps + np.roll(eps, -1, axis=0))
        ey[-1, :] = eps[-1, :]
        sxi2 = np.tile(sxi, ny)
        sxh2 = np.tile(sxh, ny)
        syi2 = np.repeat(syi, nx)
        syh2 = np.repeat(syh, nx)
        Ax = sp.diags(1 / sxi2) @ Dxb @ sp.diags(1 / (sxh2 * ex.reshape(-1))) @ Dxf
        Ay = sp.diags(1 / syi2) @ Dyb @ sp.diags(1 / (syh2 * ey.reshape(-1))) @ Dyf
        return (Ax + Ay + sp.diags(k0**2 * np.ones(nx * ny))).tocsc()

    def factor(self):
        self._A = self._operator()
        self._lu = spla.splu(self._A)
        return self

    def solve(self, source: np.ndarray) -> np.ndarray:
        """Solve ``A e = source`` (source shaped like the grid). Returns field e."""
        if self._lu is None:
            self.factor()
        b = np.asarray(source, complex).reshape(-1)
        e = self._lu.solve(b)
        return e.reshape(self.ny, self.nx)

    def solve_adjoint(self, rhs: np.ndarray) -> np.ndarray:
        """Solve ``A^T x = rhs`` (for gradients)."""
        if self._lu is None:
            self.factor()
        x = self._lu.solve(np.asarray(rhs, complex).reshape(-1), trans="T")
        return x.reshape(self.ny, self.nx)


def point_source(ny, nx, iy, ix, amp=1.0):
    """A unit point (current) source on the grid."""
    b = np.zeros((ny, nx), complex)
    b[iy, ix] = amp
    return b


def focus_objective(eps, *, dx, dy, wl, source, target, npml=12):
    """Field intensity at ``target`` from a ``source``, plus its adjoint gradient.

    Returns ``(fom, grad, field)`` where ``fom = |E(target)|**2`` and ``grad`` is
    ``d fom / d eps`` (same shape as ``eps``) computed by one adjoint solve --
    exact to machine precision. This is the building block for topology
    optimization.

    Examples
    --------
    >>> import numpy as np
    >>> from photonix.em.fdfd import focus_objective, point_source
    >>> ny=nx=60; eps=np.full((ny,nx),1.444**2)
    >>> b=point_source(ny,nx,ny//2,8)
    >>> fom,grad,E = focus_objective(eps,dx=0.05,dy=0.05,wl=1.55,source=b,target=(ny//2,nx-10))
    >>> grad.shape == eps.shape
    True
    """
    sim = FDFD(eps, dx, dy, wl, npml=npml).factor()
    E = sim.solve(source)
    ty, tx = target
    fom = float(abs(E[ty, tx]) ** 2)
    rhs = np.zeros_like(E)
    rhs[ty, tx] = np.conj(E[ty, tx])
    adj = sim.solve_adjoint(rhs)
    grad = -2.0 * sim.k0**2 * np.real(adj * E)
    return fom, grad, E


def optimize_focus(eps0, design_mask, *, dx, dy, wl, source, target,
                   eps_lo, eps_hi, steps=20, lr=None, npml=12):
    """Gradient-ascent topology optimization to maximize intensity at ``target``.

    Permittivity inside ``design_mask`` is updated by the adjoint gradient and
    projected to ``[eps_lo, eps_hi]``. Returns ``(eps_opt, history, E_final)``.
    """
    eps = np.asarray(eps0, float).copy()
    history = []
    f0, g0, _ = focus_objective(eps, dx=dx, dy=dy, wl=wl, source=source, target=target, npml=npml)
    if lr is None:
        lr = 0.05 * (eps_hi - eps_lo) / (np.max(np.abs(g0[design_mask])) + 1e-30)
    for _ in range(steps):
        fom, grad, E = focus_objective(eps, dx=dx, dy=dy, wl=wl, source=source, target=target, npml=npml)
        history.append(fom)
        eps[design_mask] = np.clip(eps[design_mask] + lr * grad[design_mask], eps_lo, eps_hi)
    fom, _, E = focus_objective(eps, dx=dx, dy=dy, wl=wl, source=source, target=target, npml=npml)
    history.append(fom)
    return eps, history, E


# --------------------------------------------------------------------------- #
# Mode source + S-parameter extraction (rigorous port characterization)
# --------------------------------------------------------------------------- #
def waveguide_mode(eps_col, dy, wl, mode=0, polarization="te"):
    """Fundamental (or ``mode``-th) 1-D mode of a vertical cross-section column.

    Returns ``(beta, profile)`` power-normalized under the polarization inner
    product (TE: ``sum p^2 dy``; TM: ``sum (1/eps) p^2 dy``).
    """
    from .eme import slab_modes

    betas, fields, _ = slab_modes(np.asarray(eps_col, float), dy, wl, mode + 1, polarization)
    return float(betas[mode]), fields[:, mode]


def mode_source(ny, nx, col, profile, beta, dx, direction=1):
    """Soft line source over two columns that launches ``profile`` forward."""
    b = np.zeros((ny, nx), complex)
    b[:, col] = profile
    b[:, col + direction] = profile * np.exp(-1j * beta * dx * direction)
    return b


def _decompose(E, col, profile, beta, dy, dx, weight=None):
    """Forward/backward modal amplitudes at a monitor plane via two adjacent columns.

    ``weight`` is the modal inner-product weight (``1/eps`` for TM, ones for TE).
    """
    w = np.ones_like(profile) if weight is None else weight
    norm = np.sum(w * profile**2) * dy
    c0 = np.sum(w * E[:, col] * profile) * dy / norm
    c1 = np.sum(w * E[:, col + 1] * profile) * dy / norm
    ep, em = np.exp(1j * beta * dx), np.exp(-1j * beta * dx)
    denom = ep - em
    fwd = (c0 * ep - c1) / denom        # forward (+x) amplitude at plane col
    bwd = (c1 - c0 * em) / denom        # backward (-x) amplitude at plane col
    return fwd, bwd


def waveguide_sparams(
    eps, *, dx, dy, wl,
    src_col, in_mon_col, out_mon_col,
    in_eps_col=None, out_eps_col=None, npml=12, mode=0, polarization="te",
) -> SDict:
    """Rigorous 2-port S-parameters of a planar device via FDFD.

    Injects the input waveguide mode, separates forward/backward at an input
    monitor (giving incident and reflected amplitudes), and the forward amplitude
    at an output monitor (transmitted). Power-normalized by ``sqrt(beta)`` so
    ``|S21|^2`` is the power transmission. Ports ``o1`` (in) and ``o2`` (out).

    Both port reflections are returned: ``("o1","o1")`` from the forward solve
    and ``("o2","o2")`` from a second, right-side-incident solve that reuses the
    already-factored operator (one extra triangular solve, not a refactor).
    Dropping S22 would make the circuit solver treat the output port as
    perfectly matched.

    Examples
    --------
    >>> import numpy as np
    >>> from photonix.em.fdfd import waveguide_sparams
    >>> ny, nx = 80, 120; dx = dy = 0.05
    >>> eps = np.full((ny, nx), 1.444**2)
    >>> eps[ny//2-5:ny//2+5, :] = 2.85**2          # a straight waveguide stripe
    >>> s = waveguide_sparams(eps, dx=dx, dy=dy, wl=1.55,
    ...                       src_col=12, in_mon_col=24, out_mon_col=nx-24)
    >>> 0.9 < abs(s[("o1", "o2")])**2 <= 1.01       # near-unity transmission
    True
    """
    from .geometry import as_real_eps

    eps = as_real_eps(eps, where="waveguide_sparams")
    ny, nx = eps.shape
    bi, mi = waveguide_mode(eps[:, in_eps_col if in_eps_col is not None else in_mon_col], dy, wl, mode, polarization)
    bo, mo = waveguide_mode(eps[:, out_eps_col if out_eps_col is not None else out_mon_col], dy, wl, mode, polarization)
    wi = (1.0 / eps[:, in_mon_col]) if polarization == "tm" else None
    wo = (1.0 / eps[:, out_mon_col]) if polarization == "tm" else None
    sim = FDFD(eps, dx, dy, wl, npml=npml, polarization=polarization).factor()
    E = sim.solve(mode_source(ny, nx, src_col, mi, bi, dx))
    a_inc, a_refl = _decompose(E, in_mon_col, mi, bi, dy, dx, wi)
    a_out, _ = _decompose(E, out_mon_col, mo, bo, dy, dx, wo)
    s21 = (a_out / a_inc) * np.sqrt(bo / bi)
    s11 = a_refl / a_inc
    # S22: right-side incidence. The operator is already LU-factored, so this is
    # one extra triangular solve. The source column mirrors src_col; at the
    # output monitor the incident wave is the backward (-x) component and the
    # device reflection the forward (+x) one.
    E2 = sim.solve(mode_source(ny, nx, nx - 1 - src_col, mo, bo, dx, direction=-1))
    b_fwd, b_inc = _decompose(E2, out_mon_col, mo, bo, dy, dx, wo)
    s22 = b_fwd / b_inc
    return {
        ("o1", "o2"): complex(s21), ("o2", "o1"): complex(s21),
        ("o1", "o1"): complex(s11),
        ("o2", "o2"): complex(s22),
    }
