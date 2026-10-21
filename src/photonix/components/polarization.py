"""Polarization-aware compact models with modal terminals (``"o1:TE0"``, ``"o1:TM0"``).

Every model returns an :data:`~photonix.core.types.SDict` over modal terminals
(see :mod:`photonix.circuit.modal`), is reciprocal (``S = S^T``) and passive
(largest singular value ``<= 1``) by construction, and composes with lifted
single-mode parts in a :class:`~photonix.circuit.ModalNetlist`. The Jones basis
is ``(TE0, TM0)``; for a fibre, ``TE0`` and ``TM0`` stand for the ``x`` and
``y`` polarizations of the fibre axes.

* :func:`polarization_rotator` -- a linear retarder (waveplate),
  ``J = R(-theta) diag(e^{-j Gamma/2}, e^{j Gamma/2}) R(theta)``.
* :func:`polarization_splitter_rotator` -- PSR after Dai & Bowers,
  Opt. Express 19(11):10940 (2011): ``TE0 -> o2:TE0``; ``TM0`` converted to
  ``TE1`` in an adiabatic taper and coupled to ``o3:TE0`` by an asymmetric
  directional coupler.
* :func:`polarization_beam_splitter` -- ``TE0 -> o2``, ``TM0 -> o3`` with
  finite extinction.
* :func:`grating_coupler_2d` -- 2D grating coupler: the two fibre
  polarizations go to two chip waveguides (Gaussian spectrum, PDL, crosstalk).
* :func:`jones_fiber` / :func:`pmd_fiber_jones` -- random birefringent-section
  PMD emulator (Gordon & Kogelnik, PNAS 97(9):4541, 2000) with Maxwellian DGD
  statistics.

These are behavioural (black-box) models with wavelength-independent
polarization parameters, except the grating coupler's spectrum and the
fibre's frequency-dependent birefringence; calibrate them to measurements or
to a full-vector EM simulation before quantitative use.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from photonix.core.backend import xp
from photonix.core.types import SDict

__all__ = [
    "TE0", "TM0", "C_UM_PER_PS",
    "polarization_rotator",
    "polarization_splitter_rotator",
    "polarization_beam_splitter",
    "grating_coupler_2d",
    "pmd_fiber_jones",
    "jones_fiber",
    "jones_sdict",
]

TE0 = "TE0"
TM0 = "TM0"
#: Speed of light in µm/ps.
C_UM_PER_PS = 299.792458
_MAXWELL_MEAN = math.sqrt(8.0 / (3.0 * math.pi))   # <DGD> / sqrt(<DGD^2>) for a Maxwellian


def _amp(loss_db):
    return 10.0 ** (-xp.asarray(loss_db, dtype=float) / 20.0)


def jones_sdict(jones, in_port: str = "o1", out_port: str = "o2",
                modes: Sequence[str] = (TE0, TM0)) -> SDict:
    """Reciprocal two-port modal SDict from a forward Jones matrix ``J[..., out, in]``.

    Forward entries are ``s[(f"{in_port}:{modes[a]}", f"{out_port}:{modes[b]}")] = J[b, a]``;
    the backward ones follow from reciprocity (``J^T``).

    Examples
    --------
    >>> import numpy as np
    >>> s = jones_sdict(np.array([[0, 1], [1, 0]]))
    >>> int(s[("o1:TE0", "o2:TM0")]), int(s[("o2:TM0", "o1:TE0")]), int(s[("o1:TE0", "o2:TE0")])
    (1, 1, 0)
    """
    modes = tuple(modes)
    out: SDict = {}
    for a, ma in enumerate(modes):
        for b, mb in enumerate(modes):
            v = jones[..., b, a] if hasattr(jones, "ndim") and jones.ndim > 2 else jones[b][a]
            out[(f"{in_port}:{ma}", f"{out_port}:{mb}")] = v
            out[(f"{out_port}:{mb}", f"{in_port}:{ma}")] = v
    return out


def polarization_rotator(*, wl=1.55, angle: float = math.pi / 4, retardance: float = math.pi,
                         loss_db: float = 0.0) -> SDict:
    """Linear retarder between ``o1`` and ``o2`` (each with ``TE0``/``TM0``).

    ``J = R(-theta) diag(exp(-j Gamma/2), exp(j Gamma/2)) R(theta)`` with
    ``R(theta) = [[cos, sin], [-sin, cos]]``, fast axis at ``angle`` (radians)
    from TE and retardance ``Gamma``, in this library's ``exp(-j beta L)``
    convention. ``J`` is symmetric, so the element is reciprocal. The defaults
    (``theta = pi/4``, ``Gamma = pi``, a half-wave plate) convert TE0 into TM0
    completely.

    Examples
    --------
    >>> s = polarization_rotator()
    >>> round(abs(complex(s[("o1:TE0", "o2:TM0")])) ** 2, 12)
    1.0
    """
    del wl
    c, s = xp.cos(angle), xp.sin(angle)
    d1, d2 = xp.exp(-0.5j * retardance), xp.exp(0.5j * retardance)
    a = _amp(loss_db)
    jte = a * (c * c * d1 + s * s * d2)
    jtm = a * (s * s * d1 + c * c * d2)
    jx = a * c * s * (d1 - d2)
    return jones_sdict([[jte, jx], [jx, jtm]])


def _sigma_max_2x2(t11, t12, t21, t22):
    """Largest singular value of ``[[t11, t12], [t21, t22]]`` (closed form)."""
    fro = xp.abs(t11) ** 2 + xp.abs(t12) ** 2 + xp.abs(t21) ** 2 + xp.abs(t22) ** 2
    det = xp.abs(t11 * t22 - t12 * t21) ** 2
    disc = xp.sqrt(xp.maximum(fro * fro - 4.0 * det, 0.0))
    return xp.sqrt(0.5 * (fro + disc))


def _passive_scale(sigma):
    return 1.0 / xp.maximum(sigma, 1.0)


def _crosstalk(db):
    return 10.0 ** (xp.asarray(db, dtype=float) / 10.0)


def polarization_splitter_rotator(*, wl=1.55, conversion_loss_db: float = 0.3, te_crosstalk_db: float = -25.0,
                                  tm_crosstalk_db: float = -20.0, loss_db: float = 0.5) -> SDict:
    """Polarization splitter-rotator: ``o1:{TE0, TM0}`` -> ``o2:TE0``, ``o3:TE0``.

    ``TE0`` at ``o1`` exits at ``o2:TE0`` with insertion loss ``loss_db``;
    ``TM0`` is converted (extra ``conversion_loss_db``) and exits at
    ``o3:TE0`` (Dai & Bowers 2011). ``te_crosstalk_db`` is the fraction of the
    TE power reaching ``o3`` and ``tm_crosstalk_db`` that of the TM power
    reaching ``o2``. The crosstalk paths carry a ``-j`` phase (evanescent
    coupling). Unequal crosstalk cannot be realized losslessly, so the
    transfer matrix is divided by ``max(1, sigma_max)``, making the device
    passive for every parameter value.

    Examples
    --------
    >>> import numpy as np
    >>> s = polarization_splitter_rotator(conversion_loss_db=0.0, loss_db=0.0,
    ...                                   te_crosstalk_db=-20, tm_crosstalk_db=-20)
    >>> m = np.array([[complex(s[(i, o)]) for i in ("o1:TE0", "o1:TM0")] for o in ("o2:TE0", "o3:TE0")])
    >>> bool(np.allclose(m.conj().T @ m, np.eye(2)))     # equal crosstalk, no loss: unitary
    True
    """
    del wl
    a = _amp(loss_db)
    b = _amp(loss_db + conversion_loss_db)
    xte, xtm = _crosstalk(te_crosstalk_db), _crosstalk(tm_crosstalk_db)
    t11 = a * xp.sqrt(1.0 - xte)            # TE -> o2:TE0
    t21 = -1j * a * xp.sqrt(xte)            # TE -> o3:TE0
    t12 = -1j * b * xp.sqrt(xtm)            # TM -> o2:TE0
    t22 = b * xp.sqrt(1.0 - xtm)            # TM -> o3:TE0 (rotated)
    k = _passive_scale(_sigma_max_2x2(t11, t12, t21, t22))
    pairs = {("o1:TE0", "o2:TE0"): t11, ("o1:TE0", "o3:TE0"): t21,
             ("o1:TM0", "o2:TE0"): t12, ("o1:TM0", "o3:TE0"): t22}
    out: SDict = {}
    for (i, o), v in pairs.items():
        out[(i, o)] = k * v
        out[(o, i)] = k * v
    return out


def polarization_beam_splitter(*, wl=1.55, te_extinction_db: float = 25.0, tm_extinction_db: float = 25.0,
                               loss_db: float = 0.3) -> SDict:
    """Polarization beam splitter: ``o1:TE0 -> o2:TE0``, ``o1:TM0 -> o3:TM0``.

    A fraction ``10**(-te_extinction_db/10)`` of the TE power leaks to
    ``o3:TE0`` and likewise for TM into ``o2:TM0`` (with a ``-j`` phase).
    Outputs keep their polarization, so the two input columns never overlap
    and the device is passive for every parameter value. Ports ``o2`` and
    ``o3`` carry both modes.

    Examples
    --------
    >>> from photonix.analysis.polarization import polarization_extinction_db
    >>> s = polarization_beam_splitter(te_extinction_db=30.0, tm_extinction_db=30.0)
    >>> round(float(polarization_extinction_db(s, "o1", "o2")), 6)      # 10 log10((1 - 1e-3) / 1e-3)
    29.995655
    """
    del wl
    a = _amp(loss_db)
    xte, xtm = _crosstalk(-xp.asarray(te_extinction_db, dtype=float)), _crosstalk(-xp.asarray(tm_extinction_db,
                                                                                               dtype=float))
    pairs = {("o1:TE0", "o2:TE0"): a * xp.sqrt(1.0 - xte), ("o1:TE0", "o3:TE0"): -1j * a * xp.sqrt(xte),
             ("o1:TM0", "o3:TM0"): a * xp.sqrt(1.0 - xtm), ("o1:TM0", "o2:TM0"): -1j * a * xp.sqrt(xtm)}
    out: SDict = {}
    for (i, o), v in pairs.items():
        out[(i, o)] = v
        out[(o, i)] = v
    return out


def grating_coupler_2d(*, wl=1.55, wl0: float = 1.55, bandwidth: float = 0.035, peak_loss_db: float = 3.0,
                       pdl_db: float = 0.0, crosstalk_db: float = -30.0) -> SDict:
    """2D (polarization-splitting) grating coupler.

    Fibre port ``o1`` carries ``TE0`` (fibre ``x``) and ``TM0`` (fibre
    ``y``); they couple to the chip waveguides ``o2:TE0`` and ``o3:TE0``. The
    spectrum is the Gaussian of :func:`photonix.components.grating_coupler`
    (``bandwidth`` = FWHM); the ``y`` path has ``pdl_db`` extra loss and a
    fraction ``10**(crosstalk_db/10)`` of each polarization reaches the other
    waveguide. The matrix is a unitary crosstalk mixer times per-polarization
    losses, so it is passive with singular values equal to the two path
    transmissions.

    Examples
    --------
    >>> import numpy as np
    >>> from photonix.analysis.polarization import pdl_db
    >>> s = grating_coupler_2d(wl=1.55, pdl_db=0.5)
    >>> j = np.array([[complex(s[(i, o)]) for i in ("o1:TE0", "o1:TM0")] for o in ("o2:TE0", "o3:TE0")])
    >>> round(float(pdl_db(j)), 9)
    0.5
    """
    wl = xp.asarray(wl)
    sigma = bandwidth / 1.6651092223153954
    g = 10.0 ** (-peak_loss_db / 20.0) * xp.exp(-0.5 * ((wl - wl0) / sigma) ** 2)
    py = _amp(pdl_db)
    x = _crosstalk(crosstalk_db)
    d, e = xp.sqrt(1.0 - x), xp.sqrt(x)
    pairs = {("o1:TE0", "o2:TE0"): g * d, ("o1:TE0", "o3:TE0"): -1j * g * e,
             ("o1:TM0", "o2:TE0"): -1j * g * e * py, ("o1:TM0", "o3:TE0"): g * d * py}
    out: SDict = {}
    for (i, o), v in pairs.items():
        v = v.astype(complex) if hasattr(v, "astype") else complex(v)
        out[(i, o)] = v
        out[(o, i)] = v
    return out


# --------------------------------------------------------------------------- #
# PMD fibre emulator
# --------------------------------------------------------------------------- #
def pmd_fiber_jones(wl, *, length_m: float = 1000.0, pmd_ps_per_sqrt_km: float = 0.1, n_sections: int = 50,
                    seed: int | Sequence[int] = 0) -> np.ndarray:
    """Forward Jones matrices of a random-birefringence PMD fibre (NumPy).

    The fibre is ``n_sections`` equal birefringent sections (Gordon &
    Kogelnik 2000). Section ``k`` has a uniformly random axis angle
    ``theta_k``, a random static phase ``phi_k`` and differential group delay
    ``tau = tau_rms / sqrt(n_sections)``, with ``tau_rms =
    pmd * sqrt(L_km) / sqrt(8 / (3 pi))`` so that the mean DGD of the
    (asymptotically Maxwellian) ensemble is ``pmd * sqrt(L_km)``:
    ``J_k(omega) = R(-theta_k) diag(e^{-j(omega tau + phi_k)/2}, e^{j(omega tau + phi_k)/2}) R(theta_k)``
    and ``J = J_N ... J_1``. The common propagation phase is omitted.

    Parameters
    ----------
    wl
        Wavelength(s) in µm (concrete values; the emulator is random, not a
        differentiable model).
    length_m, pmd_ps_per_sqrt_km
        Fibre length (m) and PMD coefficient (mean DGD per sqrt(km), ps).
    n_sections
        Number of birefringent sections.
    seed
        Random seed, or a sequence of seeds for a stacked ensemble.

    Returns
    -------
    numpy.ndarray
        Shape ``(*wl.shape, 2, 2)``, or ``(len(seed), *wl.shape, 2, 2)``.

    Examples
    --------
    >>> import numpy as np
    >>> J = pmd_fiber_jones(np.array([1.55, 1.5501]), seed=3)
    >>> J.shape, bool(np.allclose(J[0] @ J[0].conj().T, np.eye(2)))
    ((2, 2, 2), True)
    """
    if not isinstance(seed, (int, np.integer)):
        return np.stack([pmd_fiber_jones(wl, length_m=length_m, pmd_ps_per_sqrt_km=pmd_ps_per_sqrt_km,
                                         n_sections=n_sections, seed=int(s)) for s in seed])
    if int(n_sections) < 1:
        raise ValueError("n_sections must be >= 1")
    if length_m < 0 or pmd_ps_per_sqrt_km < 0:
        raise ValueError("length_m and pmd_ps_per_sqrt_km must be non-negative")
    n = int(n_sections)
    rng = np.random.default_rng(int(seed))
    theta = rng.uniform(0.0, np.pi, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    tau = pmd_ps_per_sqrt_km * math.sqrt(length_m / 1000.0) / _MAXWELL_MEAN / math.sqrt(n)
    omega = 2.0 * np.pi * C_UM_PER_PS / np.asarray(wl, dtype=float)
    J = np.broadcast_to(np.eye(2, dtype=complex), (*omega.shape, 2, 2)).copy()
    for k in range(n):
        c, s = math.cos(theta[k]), math.sin(theta[k])
        half = 0.5 * (omega * tau + phi[k])
        d1, d2 = np.exp(-1j * half), np.exp(1j * half)
        sec = np.empty((*omega.shape, 2, 2), dtype=complex)
        sec[..., 0, 0] = c * c * d1 + s * s * d2
        sec[..., 1, 1] = s * s * d1 + c * c * d2
        sec[..., 0, 1] = sec[..., 1, 0] = c * s * (d1 - d2)
        J = sec @ J
    return J


def jones_fiber(*, wl=1.55, length_m: float = 1000.0, pmd_ps_per_sqrt_km: float = 0.1, n_sections: int = 50,
                seed: int = 0, loss_db: float = 0.0) -> SDict:
    """PMD fibre as a reciprocal modal two-port (``o1``/``o2``, ``TE0`` = x, ``TM0`` = y).

    Forward transfer :func:`pmd_fiber_jones`, backward its transpose
    (reciprocity of the birefringent medium), times ``10**(-loss_db/20)``.

    Examples
    --------
    >>> import numpy as np
    >>> from photonix.analysis.polarization import dgd
    >>> wl = np.array([1.55, 1.55001])
    >>> s = jones_fiber(wl=wl, n_sections=1, pmd_ps_per_sqrt_km=1.0, length_m=1000.0)
    >>> J = np.stack([[np.asarray(s[(f"o1:{a}", f"o2:{b}")]) for a in ("TE0", "TM0")] for b in ("TE0", "TM0")])
    >>> tau = dgd(wl, np.moveaxis(J, -1, 0))
    >>> bool(np.isclose(tau[0], 1.0 / np.sqrt(8 / (3 * np.pi))))     # one section: tau_rms
    True
    """
    J = pmd_fiber_jones(np.asarray(wl), length_m=length_m, pmd_ps_per_sqrt_km=pmd_ps_per_sqrt_km,
                        n_sections=n_sections, seed=seed)
    a = _amp(loss_db)
    Jx = a * xp.asarray(J)
    return jones_sdict(Jx if Jx.ndim > 2 else [[Jx[0, 0], Jx[0, 1]], [Jx[1, 0], Jx[1, 1]]])
