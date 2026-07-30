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
  under ``sum(w psi_l psi_m) dx``; overlap ``O_{lk} = <psi_l^B | psi_k^A>``.
* **TM** -- field ``Hy`` (continuous), the generalized eigenproblem
  ``A Hy = beta^2 B Hy`` with ``A = d/dx[(1/eps) d/dx] + k0^2`` and
  ``B = diag(1/eps)``. TM power flows as ``~ (beta/eps) |Hy|^2``, so modes are
  orthonormal under the **1/eps-weighted** inner product and the interface uses
  the same vectorial weight: ``O_{lk} = integral (1/eps_B) Hy_l^B Hy_k^A dx``.
  This is what makes the overlap reciprocal and the cascade energy-conserving for
  the discontinuous-D polarization. Validated: transparent interface to 1e-16,
  energy conserved to ~1e-12, and a smooth taper stays adiabatic with reflection
  at the ~1e-5 discretization floor.

Radiation: closed window vs. graded absorber
--------------------------------------------
The transverse window is finite. With ``pml=None`` (the default) it is closed by
Dirichlet walls, so every non-guided basis mode is a **box mode** of that window:
it has real ``beta``, propagates without loss, and re-couples at downstream
interfaces. Power that should radiate away is therefore retained, and the
computed loss of a radiating structure depends on how much of the box basis
happens to be included -- it is not a converged quantity.

Passing the legacy-named ``pml=(thickness_um, strength)`` option applies a
**graded imaginary-permittivity absorber** in the transverse direction, turning
those box modes into leaky modes with ``Im(beta) < 0`` that attenuate as they
propagate. It is intentionally not a stretched-coordinate PML; see
:func:`transverse_pml` for the reason and its reflection caveat.

With the absorber the operator is complex-symmetric rather than Hermitian, so modes are
bi-orthonormal under the **unconjugated** product ``sum(w psi_l psi_m) dx`` with
``w = 1`` for TE and ``w = 1/eps`` for TM. The interface algebra below is
already written with unconjugated products, so it carries over unchanged.

Interface S-matrix (modes orthonormal within a section, amplitudes power-
normalized via ``D = diag(beta)``):

    R_f = (D_A + Oᵀ D_B O)^{-1} (D_A - Oᵀ D_B O)
    T_b = (D_A + Oᵀ D_B O)^{-1} (2 Oᵀ D_B)
    T_f = O (I + R_f)
    R_b = O T_b - I

For identical sections ``O = I`` and the interface is transparent
(R_f = R_b = 0, T = I), as it must be.

Sign convention
---------------
Fields propagate as ``exp(-i beta z)``, so a decaying mode has ``Im(beta) < 0``.
``beta`` is always returned complex on that branch: guided modes are real,
evanescent modes are negative-imaginary (they decay, they do not propagate), and
absorber-leaky modes have both parts. Earlier releases clipped ``beta**2 < 0`` to
``beta = 0``, which made evanescent modes propagate losslessly and -- through the
``sqrt(beta)`` power normalization -- amplified them by ~1e6, so asking for more
modes than the section could propagate produced an S-matrix with singular values
in the thousands.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from photonix.core.types import SDict

__all__ = ["Section", "slab_modes", "eme_smatrix", "EMEResult", "transverse_pml"]


def transverse_pml(n: int, dx: float, k0: float, pml, eps_edge: float = 1.0, m: int = 3):
    """Graded absorbing layer for a 1-D transverse window.

    Returns the complex permittivity *offset* ``d_eps`` (zero in the interior,
    negative-imaginary inside the layer) to be added to ``eps``. Adding loss to
    the cladding wings makes non-guided modes leaky -- they acquire
    ``Im(beta) < 0`` and attenuate as they propagate -- so radiated power leaves
    the simulation instead of bouncing off the window walls.

    Why not a stretched-coordinate PML
    ----------------------------------
    A true SC-PML replaces ``d/dx`` with ``(1/s) d/dx``, which in the *modal*
    eigenproblem puts ``k0^2 eps s`` on the diagonal. Since ``|s|`` reaches
    several units inside the absorber, that term makes the layer behave like a
    high-index medium and spawns a dense band of modes *of the absorber* at
    ``Re(n_eff)`` just below the core index -- precisely where the shift-invert
    target sits. Measured on a 500 nm SOI strip: a spurious band at
    ``n_eff ~ 3.49``, ahead of the true guided mode at 3.272, with ~58 % of its
    energy inside the absorber.

    A graded imaginary permittivity avoids this entirely: ``Re(eps)`` is
    unchanged, so no spurious high-index band appears, the operator keeps exactly
    the same form as the lossless one (merely complex-symmetric), and the guided
    mode -- which is exponentially small out at the layer -- is untouched. It
    reflects a little more than an ideal PML, which the cubic grading keeps small.

    Parameters
    ----------
    n, dx
        Number of transverse points and their spacing (µm).
    k0
        Free-space wavenumber ``2*pi/wl``.
    pml
        ``None`` for a closed (Dirichlet) window, or ``(thickness_um, strength)``.
        ``thickness_um`` is the layer depth on *each* side. ``strength = 1.0``
        grades ``Im(eps)`` to the value that attenuates a normally-incident wave
        by ``e**-3`` across the layer; larger absorbs harder but reflects more.
    eps_edge
        Permittivity at the window edge, used to scale the absorption to the
        local index.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not np.isfinite(dx) or dx <= 0 or not np.isfinite(k0) or k0 <= 0:
        raise ValueError("dx and k0 must be positive and finite")
    if not isinstance(m, (int, np.integer)) or isinstance(m, (bool, np.bool_)) or m < 0:
        raise ValueError("m must be a non-negative integer")
    if pml is None:
        return np.zeros(n, dtype=complex)
    try:
        thickness, strength = pml
    except (TypeError, ValueError) as exc:
        raise ValueError("pml must be None or a (thickness, strength) pair") from exc
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("absorber thickness must be positive and finite")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("absorber strength must be non-negative and finite")
    npml = int(round(thickness / dx))
    if npml <= 0:
        return np.zeros(n, dtype=complex)
    if 2 * npml >= n:
        raise ValueError(
            f"Absorber thickness {thickness} um ({npml} cells per side) does not fit in "
            f"a {n}-point window; widen half_window or thin the absorber."
        )
    # Depth into the layer, normalized to [0, 1] at each edge.
    i = np.arange(n)
    d = np.maximum(npml - i, 0) + np.maximum(i - (n - 1 - npml), 0)
    d = d / npml
    # Im(eps) that gives amplitude decay exp(-3) across the layer at normal
    # incidence: alpha = k0 * Im(eps) / (2 sqrt(Re eps)), so Im(eps) at full
    # depth = 2 sqrt(eps_edge) * 3 / (k0 * thickness).
    eps_i_max = strength * 2.0 * np.sqrt(max(eps_edge, 1e-12)) * 3.0 / (k0 * thickness)
    return -1j * eps_i_max * (d ** m)


#: Modes with |beta| below this are at cutoff: they carry no power and are
#: excluded from the interface algebra rather than rescaled (see ``_interface``).
_BETA_CUTOFF = 1e-9

#: A mode with more than this fraction of its |psi|^2 inside the absorber is an
#: absorber-localized mode, not a mode of the structure. See :func:`_select_physical`.
_ABSORBER_ENERGY_MAX = 0.35


def _select_physical(vals, vecs, npml, n_max, k0, want):
    """Drop absorber-localized modes and keep ``want`` physical modes.

    Even a graded lossy boundary can return modes concentrated primarily inside
    the absorber rather than the physical structure. Selecting by proximity to
    the shift alone can therefore return absorber modes instead of waveguide or
    radiation modes.

    Two physical criteria separate them:

    * a mode of the structure keeps its energy in the *interior*, whereas an
      absorber-localized mode is concentrated in the lossy boundary; and
    * no mode can have ``Re(n_eff)`` above the highest index present.

    Returns ``(vals, vecs)`` filtered and ordered by descending ``Re(beta**2)``.
    """
    frac = np.zeros(vecs.shape[1])
    if npml > 0:
        p = np.abs(vecs) ** 2
        tot = p.sum(axis=0)
        tot[tot == 0] = 1.0
        frac = (p[:npml].sum(axis=0) + p[-npml:].sum(axis=0)) / tot
    neff = np.sqrt(np.asarray(vals, dtype=complex)) / k0
    # `n_max` is a hard physical bound, so the index test is applied strictly --
    # relaxing it is what let the absorber band back in.
    physical = (frac <= _ABSORBER_ENERGY_MAX) & (neff.real <= n_max)
    idx = np.flatnonzero(physical)
    idx = idx[np.argsort(-np.real(np.asarray(vals)[idx]))]
    if idx.size < want:
        raise RuntimeError(
            f"Only {idx.size} of {vecs.shape[1]} computed modes are modes of the "
            f"structure rather than of the absorber, but {want} were requested. "
            "Widen the transverse window, thin the absorber, or lower num_modes."
        )
    return np.asarray(vals)[idx], vecs[:, idx]


def _beta_from_beta2(b2):
    """Propagation constants on the physical branch: ``Re(beta) >= 0``, ``Im(beta) <= 0``.

    With the ``exp(-i beta z)`` convention a *forward* wave needs ``Re(beta) > 0``
    and a *decaying* one needs ``Im(beta) < 0``. Both hold on the same branch:

    * guided (``beta**2 > 0`` real) -> real positive;
    * evanescent (``beta**2 < 0`` real) -> purely negative-imaginary, so
      ``exp(-i beta z) = exp(-|beta| z)`` decays rather than propagating;
    * leaky under the absorber -> positive real part, negative imaginary part.

    ``Re`` is decided first because a mode that is essentially propagating can
    pick up a tiny imaginary part from the complex solve, which is enough to send
    ``numpy.sqrt`` onto the negative-real branch and flip the mode's direction.
    """
    beta = np.sqrt(np.asarray(b2, dtype=complex))
    tol = 1e-12 * np.maximum(np.abs(beta), 1.0)
    flip = (beta.real < -tol) | ((np.abs(beta.real) <= tol) & (beta.imag > 0))
    return np.where(flip, -beta, beta)


def _d_faces(n: int, h: float) -> sp.csr_matrix:
    """Cell-to-face gradient with zero exterior values on both boundaries."""
    rows = np.concatenate(([0], np.arange(1, n), np.arange(1, n), [n]))
    cols = np.concatenate(([0], np.arange(n - 1), np.arange(1, n), [n - 1]))
    data = np.concatenate(([1.0], -np.ones(n - 1), np.ones(n - 1), [-1.0])) / h
    return sp.coo_matrix((data, (rows, cols)), shape=(n + 1, n)).tocsr()


@dataclass
class Section:
    """A z-uniform EME section: a 1-D permittivity profile and a length."""

    eps: np.ndarray   # (nx,) permittivity along x
    length: float     # z length (µm)

    def __post_init__(self) -> None:
        if not np.isfinite(self.length) or self.length < 0:
            raise ValueError("Section length must be non-negative and finite")


def slab_modes(eps: np.ndarray, dx: float, wl: float, num_modes: int,
               polarization: str = "te", pml=None):
    """Modes of a 1-D cross-section: returns ``(betas, fields, weight)``.

    ``betas`` are complex propagation constants ``n_eff k0``, sorted by
    descending ``Re(beta**2)`` (most-confined first) and always on the
    ``Im(beta) <= 0`` branch so that ``exp(-i beta z)`` decays -- see the module
    docstring's sign convention. Guided modes come back real; evanescent modes
    negative-imaginary; absorber-leaky modes with both parts.

    ``fields`` are normalized under the **unconjugated** weighted product
    ``sum(w psi**2) dx = 1``, and the per-point ``weight`` is returned for use in
    the interface overlap:

    ===========  ====================  =========================
    polarization  closed-window weight  absorber weight
    ===========  ====================  =========================
    ``"te"``      ``1``                 ``1``
    ``"tm"``      ``1/eps``             ``1/eps``
    ===========  ====================  =========================

    Parameters
    ----------
    pml
        ``None`` (closed Dirichlet window, real symmetric problem) or
        ``(thickness_um, strength)`` for a graded lossy absorber -- see
        :func:`transverse_pml`. With the absorber the operator is
        complex-symmetric and a non-Hermitian eigensolver is used. The parameter
        retains its historical name for API compatibility; it is not SC-PML.

        **Caveat (C3)**: with ``pml=None`` the non-guided modes are **box modes**
        of the finite Dirichlet window — they have real ``beta``, propagate
        losslessly, and re-couple at downstream interfaces.  Radiated power is
        retained in the simulation rather than lost, which over-estimates
        transmission in multi-section cascades. Use the absorber
        (``pml=(thickness, strength)``) for physically meaningful radiation
        loss; it is the default in the EME-backed components.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-3, 3, 241); dx = float(x[1] - x[0])
    >>> eps = np.where(np.abs(x) < 0.25, 3.4757**2, 1.444**2)
    >>> b, f, w = slab_modes(eps, dx, 1.55, 4, "te")
    >>> bool(np.all(b.imag <= 1e-12))          # decaying branch
    True
    """
    from .geometry import as_real_eps

    if (not isinstance(num_modes, (int, np.integer))
            or isinstance(num_modes, (bool, np.bool_)) or num_modes <= 0):
        raise ValueError("num_modes must be a positive integer")
    if not np.isfinite(dx) or dx <= 0 or not np.isfinite(wl) or wl <= 0:
        raise ValueError("dx and wl must be positive and finite")
    eps = as_real_eps(eps, where="slab_modes/EME")
    if eps.ndim != 1 or eps.size < 3 or not np.all(np.isfinite(eps)):
        raise ValueError("eps must be a finite one-dimensional array with at least 3 points")
    k0 = 2.0 * np.pi / wl
    n = len(eps)
    if polarization not in ("te", "tm"):
        raise ValueError("polarization must be 'te' or 'tm'")

    d_eps = transverse_pml(n, dx, k0, pml, eps_edge=float(eps[0]))
    use_pml = bool(np.any(d_eps != 0.0))

    # Cell-to-face / face-to-cell pair. Including n+1 faces supplies both outer
    # Dirichlet contributions; the previous n-face form silently imposed a
    # Neumann condition on the left and Dirichlet on the right.
    Df = _d_faces(n, dx)
    Db = -Df.T

    # The absorber goes into the *potential* term (the k0^2 diagonal), never into
    # the eigenproblem's weight matrix B. For TE those are the same thing, since
    # eps multiplies k0^2 directly. For TM they are not: B = diag(1/eps), and
    # letting a complex eps into it makes |B| small inside the absorber, which
    # admits a dense band of absorber modes *above* the core index -- measured at
    # 33 of 60 returned modes, swamping the physical spectrum. Keeping B real
    # removes that band entirely while still absorbing.
    weight: np.ndarray
    if polarization == "te":
        # d2psi/dx2 + k0^2 eps psi = beta^2 psi
        A = ((Db @ Df).astype(complex) + sp.diags(k0**2 * (eps.astype(complex) + d_eps))).tocsc()
        B = sp.identity(n, dtype=complex, format="csc")
        weight = np.ones(n, dtype=complex)
    else:
        # d/dx[(1/eps) dHy/dx] + k0^2 (1 + d_eps/eps) Hy = beta^2 (1/eps) Hy
        eps_face = np.empty(n + 1, dtype=float)
        eps_face[0], eps_face[-1] = eps[0], eps[-1]
        eps_face[1:-1] = 0.5 * (eps[:-1] + eps[1:])
        inv_face = (1.0 / eps_face).astype(complex)
        pot = k0**2 * (1.0 + d_eps / eps)
        A = (Db @ sp.diags(inv_face) @ Df + sp.diags(pot)).tocsc()
        weight = (1.0 / eps).astype(complex)
        B = sp.diags(weight).tocsc()

    kk = int(min(num_modes, n - 2))
    sigma = (np.sqrt(eps.max()) * k0) ** 2 * 1.0001
    # ARPACK starts from a *random* vector unless given one, which makes the
    # returned basis -- and therefore every S-matrix built from it -- differ from
    # run to run whenever modes are near-degenerate. Three identical mmi1x2()
    # calls used to return three different transmissions (spread 6.5e-4). A fixed
    # seeded start vector makes the solve reproducible; it has a component along
    # every mode, so convergence is unaffected.
    v0 = np.random.default_rng(0).standard_normal(n)
    if use_pml:
        # Complex-symmetric generalized problem -> non-Hermitian solver. A little
        # oversolving plus _select_physical drops anything that ends up living
        # inside the absorber rather than in the structure.
        npml = int(round(pml[0] / dx))
        kk_solve = int(min(max(2 * kk, kk + 6), n - 2))
        vals, vecs = spla.eigs(A, k=kk_solve, M=B, sigma=sigma, which="LM", v0=v0)
        vals, vecs = _select_physical(vals, vecs, npml, float(np.sqrt(eps.max())), k0, kk)
        vals, vecs = vals[:kk], vecs[:, :kk]
    else:
        # Real symmetric: keep the cheaper, more robust Hermitian path (and the
        # exact numbers the closed-window tests were validated against).
        vals, vecs = spla.eigsh(A.real, k=kk, M=B.real, sigma=sigma, which="LM", v0=v0)
        vals = vals.astype(complex)
        vecs = vecs.astype(complex)
        order = np.argsort(np.real(vals))[::-1]
        vals, vecs = vals[order], vecs[:, order]

    betas = _beta_from_beta2(vals)

    # Bi-orthonormalize under the unconjugated weighted product sum(w psi^2) dx = 1,
    # then fix the residual sign/phase so a real mode stays real and positive-peaked.
    for i in range(vecs.shape[1]):
        nrm = np.sqrt(np.sum(weight * vecs[:, i] ** 2) * dx)
        if nrm == 0:
            continue
        vecs[:, i] = vecs[:, i] / nrm
        peak = vecs[np.argmax(np.abs(vecs[:, i])), i]
        if peak != 0:
            vecs[:, i] = vecs[:, i] * np.sign(peak.real if peak.real != 0 else 1.0)
    if not use_pml:
        vecs = vecs.real.astype(complex)

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
    betaA = np.asarray(betaA, dtype=complex)
    betaB = np.asarray(betaB, dtype=complex)
    DA = np.diag(betaA)
    DB = np.diag(betaB)
    K = O.T @ DB @ O                              # NA x NA
    inv = np.linalg.inv(DA + K)
    Rf = inv @ (DA - K)
    Tb = inv @ (2.0 * O.T @ DB)
    Tf = O @ (np.eye(len(betaA)) + Rf)
    Rb = O @ Tb - np.eye(len(betaB))
    # Similarity transform to power-normalized amplitudes. sqrt is the complex
    # principal branch: beta is real for guided modes, negative-imaginary for
    # evanescent ones and complex with the absorber, so no clamping is needed.
    #
    # This used to floor beta at 1e-12 before the sqrt, which divided the rows
    # and columns of any non-propagating mode by ~1e-6 -- enough to push the
    # cascade's largest singular value into the thousands (see docs/PHYSICS_AUDIT.md,
    # A1). Modes that genuinely sit at beta == 0 are at cutoff and carry no power,
    # so they are dropped instead of rescaled.
    sA = np.sqrt(betaA)
    sB = np.sqrt(betaB)
    keepA = np.abs(betaA) > _BETA_CUTOFF
    keepB = np.abs(betaB) > _BETA_CUTOFF
    sA = np.where(keepA, sA, 1.0)
    sB = np.where(keepB, sB, 1.0)
    Rf = (sA[:, None] * Rf) / sA[None, :]
    Tb = (sA[:, None] * Tb) / sB[None, :]
    Tf = (sB[:, None] * Tf) / sA[None, :]
    Rb = (sB[:, None] * Rb) / sB[None, :]
    # Zero out any exactly-cut-off mode so it cannot carry amplitude either way.
    Rf = Rf * keepA[:, None] * keepA[None, :]
    Tb = Tb * keepA[:, None] * keepB[None, :]
    Tf = Tf * keepB[:, None] * keepA[None, :]
    Rb = Rb * keepB[:, None] * keepB[None, :]
    return Rf, Tf, Tb, Rb


def _prop(beta, length):
    """Propagation through a uniform section: ``exp(-i beta L)`` per mode.

    Complex ``beta`` on the ``Im <= 0`` branch means an evanescent or absorber-leaky
    mode attenuates over the section instead of sailing through unchanged.
    """
    P = np.diag(np.exp(-1j * np.asarray(beta, dtype=complex) * length))
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
        """Scattering dict over the first ``n_in``/``n_out`` modes.

        Port convention: input modes are ``o1 … o{n_in}``, output modes are
        ``o{n_in+1} … o{n_in+n_out}``, following the project-wide ``oN``
        convention. The output-side reflection block ``Rb`` is now included
        (see PHYSICS_AUDIT §D2).
        """
        out: SDict = {}
        for i in range(n_in):
            for j in range(n_out):
                out[(f"o{i+1}", f"o{n_in+j+1}")] = complex(self.Tf[j, i])
                out[(f"o{n_in+j+1}", f"o{i+1}")] = complex(self.Tb[i, j])
            for k in range(n_in):
                out[(f"o{i+1}", f"o{k+1}")] = complex(self.Rf[k, i])
        for j in range(n_out):
            for k in range(n_out):
                out[(f"o{n_in+j+1}", f"o{n_in+k+1}")] = complex(self.Rb[k, j])
        return out


def eme_smatrix(sections: list[Section], dx: float, wl: float, num_modes: int = 6,
                polarization: str = "te", pml=None) -> EMEResult:
    """Cascade EME over ``sections`` and return the total S-matrix.

    Parameters
    ----------
    sections
        The z-uniform slices, in propagation order.
    dx, wl
        Transverse grid spacing and wavelength (µm).
    num_modes
        Size of the modal basis per section. Modes beyond the section's guided
        count are evanescent/leaky and now decay correctly, so raising this is
        safe -- it costs accuracy in the truncated basis, not energy conservation.
    polarization
        ``"te"`` (default) or the vectorial ``"tm"`` formulation.
    pml
        ``None`` for a closed Dirichlet window, or ``(thickness_um, strength)``
        for a graded imaginary-permittivity absorber. **Use an absorber whenever the structure
        radiates** (tapers, MMIs, junctions): without it the non-guided basis
        modes are lossless box modes of the window, they carry radiated power to
        the far end and re-couple, and the resulting loss is a function of the
        window size rather than of the physics. See the module docstring.

    Examples
    --------
    >>> import numpy as np, photonix.em as em
    >>> x = np.linspace(-3, 3, 241); dx = x[1]-x[0]
    >>> eps = np.where(np.abs(x) < 0.25, 3.4757**2, 1.444**2)
    >>> r = em.eme.eme_smatrix([em.eme.Section(eps, 5.0)], dx, 1.55, num_modes=4)
    >>> bool(abs(abs(r.Tf[0, 0]) - 1.0) < 1e-6)     # straight WG: lossless
    True

    A radiating structure needs the absorber to give a window-independent answer:

    >>> secs = [em.eme.Section(np.where(np.abs(x) < w/2, 3.4757**2, 1.444**2), 2.0)
    ...         for w in (0.5, 1.2)]
    >>> r = em.eme.eme_smatrix(secs, dx, 1.55, num_modes=10, pml=(0.8, 1.0))
    >>> bool(abs(r.Tf[0, 0])**2 < 1.0)               # power genuinely leaves
    True
    """
    if not sections:
        raise ValueError("sections must contain at least one EME section")
    modes = [slab_modes(s.eps, dx, wl, num_modes, polarization, pml=pml) for s in sections]
    betas0, _f0, _w0 = modes[0]
    S = _prop(betas0, sections[0].length)
    for i in range(1, len(sections)):
        bA, fA, _wA = modes[i - 1]
        bB, fB, wB = modes[i]
        # The absorber changes the potential, not the modal weight. TM still
        # requires its 1/eps overlap; TE uses the ordinary bilinear product.
        weightB = wB if polarization == "tm" else None
        S = _star(S, _interface(bA, fA, bB, fB, dx, weightB))
        S = _star(S, _prop(bB, sections[i].length))
    Rf, Tf, Tb, Rb = S
    return EMEResult(Rf, Tf, Tb, Rb, modes[0][0], modes[-1][0])
