"""Grating couplers and simple lumped elements (phase shifter, attenuator, ...).

All models are pure, differentiable functions returning an
:data:`~photonix.core.types.SDict`. Wavelengths/lengths in micrometers.
"""
from __future__ import annotations

from photonix.core.backend import xp
from photonix.core.types import SDict

__all__ = ["grating_coupler", "phase_shifter", "attenuator", "terminator"]


def grating_coupler(
    *,
    wl=1.55,
    wl0: float = 1.55,
    bandwidth: float = 0.035,
    peak_loss_db: float = 3.0,
) -> SDict:
    """Fiber-to-chip grating coupler with a Gaussian spectral response.

    Two ports: ``o1`` (in-plane waveguide) and ``o2`` (out-of-plane fiber). The
    power transmission is Gaussian in wavelength::

        T(wl) = T_peak * exp(-((wl - wl0)/sigma)**2),  sigma = bandwidth/1.6651

    so ``bandwidth`` is the full width at half maximum (µm). Reciprocal.

    Examples
    --------
    >>> import photonix as px
    >>> import numpy as np
    >>> wl = px.linspace(1.5, 1.6, 101)
    >>> s = px.components.grating_coupler(wl=wl, wl0=1.55)
    >>> i = int(np.argmax(px.to_numpy(px.power(s[("o1","o2")]))))
    >>> abs(float(wl[i]) - 1.55) < 1e-3
    True
    """
    wl = xp.asarray(wl)
    sigma = bandwidth / 1.6651092223153954  # FWHM -> Gaussian sigma
    t_peak = 10.0 ** (-peak_loss_db / 20.0)
    amp = t_peak * xp.exp(-0.5 * ((wl - wl0) / sigma) ** 2)
    amp = amp.astype(complex)
    return {("o1", "o2"): amp, ("o2", "o1"): amp}


def phase_shifter(
    *,
    wl=1.55,
    length: float = 100.0,
    dn_dv: float = 0.0,
    voltage: float = 0.0,
    phase0: float = 0.0,
    loss_db_cm: float = 0.0,
) -> SDict:
    """Tunable phase shifter (e.g. thermo-optic or carrier-based).

    Applies ``phi = phase0 + 2*pi/wl * (dn_dv * voltage) * length`` plus optional
    propagation loss. Ports ``o1``/``o2``.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.phase_shifter(length=100.0, dn_dv=1e-3, voltage=1.0)
    >>> set(s) == {("o1","o2"), ("o2","o1")}
    True
    """
    wl = xp.asarray(wl)
    from photonix.core.units import db_per_cm_to_alpha_um

    dn = dn_dv * voltage
    phi = phase0 + 2.0 * xp.pi / wl * dn * length
    alpha = db_per_cm_to_alpha_um(loss_db_cm)
    t = (xp.exp(-alpha * length) * xp.exp(-1j * phi)).astype(complex)
    return {("o1", "o2"): t, ("o2", "o1"): t}


def attenuator(*, wl=1.55, loss_db: float = 3.0) -> SDict:
    """Reciprocal 2-port attenuator with ``loss_db`` power loss. Ports o1/o2."""
    wl = xp.asarray(wl)
    amp = (10.0 ** (-loss_db / 20.0)) * xp.ones_like(xp.real(wl))
    a = amp.astype(complex)
    return {("o1", "o2"): a, ("o2", "o1"): a}


def terminator(*, wl=1.55, reflection: float = 0.0) -> SDict:
    """Single-port terminator with (amplitude) ``reflection`` on ``o1``.

    A perfect terminator (``reflection=0``) absorbs all incident light.
    """
    wl = xp.asarray(wl)
    r = (xp.asarray(reflection) * xp.ones_like(xp.real(wl))).astype(complex)
    return {("o1", "o1"): r}
