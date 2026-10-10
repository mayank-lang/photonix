"""Waveguide Bragg gratings from coupled-mode theory (CMT) transfer matrices.

A grating of period ``period`` couples the forward and backward fundamental
modes. Writing the field as ``A(z) exp(-j beta_B z) + B(z) exp(+j beta_B z)``
with the Bragg wavenumber ``beta_B = pi / period``, the slowly varying envelopes
obey::

    dA/dz = -j delta A - j kappa  B
    dB/dz = +j delta B + j kappa* A,      delta = beta(wl) - beta_B - j alpha

A section of length ``dz`` with constant ``kappa`` and ``delta`` therefore has
the exact transfer matrix ``exp(M dz) = cosh(gamma dz) I + sinh(gamma dz)/gamma M``
with ``gamma**2 = |kappa|**2 - delta**2`` (``M**2 = gamma**2 I``). Products of
section matrices describe apodized and phase-shifted gratings; a single section
reproduces the textbook uniform-grating closed form (peak reflectance
``tanh(kappa L)**2``).

Ports ``o1`` (z = 0) and ``o2`` (z = L) with reference planes at the grating
ends; the model is reciprocal and, without loss, unitary.
"""
from __future__ import annotations

import warnings

import numpy as np

from photonix.core.backend import xp
from photonix.core.constants import N_GROUP_SI_STRIP, WL_DEFAULT
from photonix.core.types import SDict
from photonix.core.units import db_per_cm_to_alpha_um

__all__ = ["bragg_grating", "bragg_wavelength", "grating_kappa"]


def _cosh_sinhc(g2):
    """``(cosh(g), sinh(g)/g)`` with ``g = sqrt(g2)``, as entire functions of ``g2``.

    Both are analytic in ``g2``, so evaluating them through ``sqrt`` is only a
    numerical device. Near ``g2 = 0`` (a ``kappa = 0`` section at the Bragg
    wavelength, or an optimisation starting from ``kappa = 0``) the derivative of
    ``sqrt`` is infinite and would turn the gradient into NaN; there the Taylor
    series (error < 1e-16 for ``|g2| < 1e-3``) is used instead.
    """
    small = xp.abs(g2) < 1e-3
    gs = xp.sqrt(xp.where(small, xp.ones_like(g2), g2))
    cosh_series = 1.0 + g2 / 2.0 + g2 * g2 / 24.0 + g2 * g2 * g2 / 720.0
    sinhc_series = 1.0 + g2 / 6.0 + g2 * g2 / 120.0 + g2 * g2 * g2 / 5040.0
    return (xp.where(small, cosh_series, xp.cosh(gs)),
            xp.where(small, sinhc_series, xp.sinh(gs) / gs))


def grating_kappa(dn: float, wl=WL_DEFAULT):
    """Coupling coefficient (1/µm) of a 50 % duty-cycle rectangular grating.

    For a square-wave effective-index modulation of peak-to-peak ``dn`` the
    first Fourier harmonic gives ``kappa = 2 * dn / wl``.

    Examples
    --------
    >>> round(float(grating_kappa(0.03, 1.55)), 6)
    0.03871
    """
    return 2.0 * xp.asarray(dn) / xp.asarray(wl)


def bragg_wavelength(period: float, neff: float = 2.4, ng: float = N_GROUP_SI_STRIP,
                     wl0: float = WL_DEFAULT) -> float:
    """Solve ``wl = 2 * period * n_eff(wl)`` for the linear-dispersion index.

    With ``n_eff(wl) = neff - (ng - neff) (wl - wl0) / wl0`` the condition is
    linear in ``wl`` and has the closed form returned here.

    Examples
    --------
    >>> round(bragg_wavelength(0.3229, neff=2.4, ng=4.2), 4)
    1.55
    """
    # wl = 2P [neff - (ng - neff)(wl - wl0)/wl0]  =>  wl (1 + 2P (ng-neff)/wl0) = 2P ng
    slope = (ng - neff) / wl0
    return float(2.0 * period * ng / (1.0 + 2.0 * period * slope))


def _apodization_profile(kind: str | None, n_sections: int, strength: float) -> np.ndarray:
    z = (np.arange(n_sections) + 0.5) / n_sections - 0.5  # section centres in [-1/2, 1/2]
    if kind is None:
        return np.ones(n_sections)
    if kind == "gaussian":
        # ``strength`` is the Gaussian FWHM as a fraction of the grating length.
        return np.exp(-4.0 * np.log(2.0) * (z / strength) ** 2)
    if kind == "raised_cosine":
        # ``strength`` in (0, 1]: 1 is a full raised cosine (1 at the centre,
        # 0 at the ends), 0 is uniform.
        return 1.0 - strength * 0.5 * (1.0 - np.cos(2.0 * np.pi * z))
    raise ValueError("apodization must be None, 'gaussian' or 'raised_cosine'")


def bragg_grating(
    *,
    wl=WL_DEFAULT,
    period: float = 0.3229,
    n_periods: int = 300,
    dn: float = 0.03,
    kappa=None,
    n_sections: int = 1,
    apodization: str | None = None,
    apodization_strength: float = 0.5,
    phase_shift: float = 0.0,
    neff: float = 2.4,
    ng: float = N_GROUP_SI_STRIP,
    wl0: float = WL_DEFAULT,
    loss_db_cm: float = 0.0,
) -> SDict:
    """Waveguide Bragg grating (uniform, apodized or phase-shifted). Ports o1/o2.

    Parameters
    ----------
    period
        Grating period in µm (``0.3229`` puts the Bragg wavelength at ~1.55 µm
        for the default index model; see :func:`bragg_wavelength`).
    n_periods
        Number of periods; the length is ``period * n_periods``. Use an
        integer so both ends sit at the same grating phase, which makes the
        lossless device exactly unitary and symmetric (``S11 == S22``).
    dn
        Peak-to-peak effective-index modulation of a 50 % duty rectangular
        corrugation; converted with :func:`grating_kappa`.
    kappa
        Optional coupling coefficient in 1/µm (scalar, or one value per
        section, possibly complex) that overrides ``dn``.
    n_sections
        Number of piecewise-uniform transfer-matrix sections. ``1`` is the exact
        uniform grating; use more for apodization.
    apodization, apodization_strength
        ``None`` (uniform), ``"gaussian"`` (``strength`` = FWHM as a fraction
        of the length) or ``"raised_cosine"`` (``strength`` in (0, 1]).
    phase_shift
        Grating phase shift (radians) inserted at the centre, implemented as
        ``kappa -> kappa * exp(j * phase_shift)`` for the second half. ``pi`` is
        the quarter-wave shift that opens a transmission peak at the Bragg
        wavelength. Requires an even ``n_sections`` (``1`` is promoted to ``2``).
    neff, ng, wl0
        Linear-dispersion model of the unperturbed mode.
    loss_db_cm
        Propagation loss.

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> wl_b = px.components.bragg_wavelength(0.3229)
    >>> s = px.components.bragg_grating(wl=wl_b, dn=0.03, n_periods=300)
    >>> kL = float(px.components.grating_kappa(0.03, wl_b)) * 0.3229 * 300
    >>> bool(abs(float(px.power(s[("o1", "o1")])) - np.tanh(kL) ** 2) < 1e-6)
    True
    >>> abs(float(px.power(s[("o1", "o1")]) + px.power(s[("o1", "o2")])) - 1) < 1e-12
    True
    """
    if not isinstance(n_sections, (int, np.integer)) or isinstance(n_sections, bool) or n_sections < 1:
        raise ValueError("n_sections must be a positive integer")
    n_sec = int(n_sections)
    shifted = not (isinstance(phase_shift, (int, float)) and phase_shift == 0.0)
    if shifted and n_sec % 2:
        if n_sec != 1:
            raise ValueError("a central phase shift needs an even n_sections")
        n_sec = 2
    if apodization is not None and n_sec < 8:
        warnings.warn(
            f"apodization={apodization!r} with n_sections={n_sec} samples the profile very coarsely; "
            "use n_sections >= 20 for a meaningful apodized response",
            stacklevel=2,
        )

    wl = xp.asarray(wl)
    length = period * n_periods
    dz = length / n_sec
    beta_b = xp.pi / period
    n = neff - (ng - neff) * (wl - wl0) / wl0
    alpha = db_per_cm_to_alpha_um(loss_db_cm)
    delta = 2.0 * xp.pi * n / wl - beta_b - 1j * alpha

    if kappa is None:
        k_base = grating_kappa(dn, wl)
        profile = _apodization_profile(apodization, n_sec, apodization_strength)
        kappas = [k_base * float(profile[i]) for i in range(n_sec)]
    else:
        k_arr = xp.asarray(kappa)
        if k_arr.ndim == 0:
            profile = _apodization_profile(apodization, n_sec, apodization_strength)
            kappas = [k_arr * float(profile[i]) for i in range(n_sec)]
        else:
            if k_arr.shape[0] != n_sec:
                raise ValueError("a kappa array needs exactly one value per section")
            kappas = [k_arr[i] for i in range(n_sec)]
    if shifted:
        rot = xp.exp(1j * xp.asarray(phase_shift))
        kappas = [k * rot if i >= n_sec // 2 else k for i, k in enumerate(kappas)]

    # Accumulate T = T_N ... T_1 elementwise over the wavelength batch.
    t11 = t22 = xp.ones_like(delta)
    t12 = t21 = xp.zeros_like(delta)
    for k in kappas:
        k = xp.asarray(k, dtype=complex)
        # cosh(gamma dz) and sinh(gamma dz)/gamma with gamma^2 = |kappa|^2 - delta^2
        ch, shc = _cosh_sinhc((k * xp.conj(k) - delta * delta) * dz * dz)
        sh = dz * shc
        a11 = ch - 1j * delta * sh
        a12 = -1j * k * sh
        a21 = 1j * xp.conj(k) * sh
        a22 = ch + 1j * delta * sh
        t11, t12, t21, t22 = (a11 * t11 + a12 * t21, a11 * t12 + a12 * t22,
                              a21 * t11 + a22 * t21, a21 * t12 + a22 * t22)

    carrier = xp.exp(-1j * beta_b * length)
    r1 = -t21 / t22
    t = carrier / t22
    r2 = t12 * carrier * carrier / t22
    r1 = xp.asarray(r1, dtype=complex)
    r2 = xp.asarray(r2, dtype=complex)
    t = xp.asarray(t, dtype=complex)
    return {
        ("o1", "o2"): t, ("o2", "o1"): t,
        ("o1", "o1"): r1, ("o2", "o2"): r2,
    }
