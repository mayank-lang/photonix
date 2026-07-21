"""Analytic Mach-Zehnder interferometer reference model.

The circuit-assembled MZI (two couplers + two arms) lives in
:mod:`photonix.circuit`; this closed form is an exact reference for validation
and fast sweeps. Ports ``in0`` -> ``out0`` (bar) and ``out1`` (cross).
"""
from __future__ import annotations

from photonix.core.backend import xp
from photonix.core.types import SDict

__all__ = ["mzi"]


def mzi(
    *,
    wl=1.55,
    delta_length: float = 20.0,
    neff: float = 2.4,
    ng: float = 4.2,
    wl0: float = 1.55,
    coupling: float = 0.5,
    loss_db_cm: float = 0.0,
) -> SDict:
    """Balanced/imbalanced MZI with two ``coupling`` splitters.

    The relative phase between the arms is ``dphi = 2*pi/wl * n_eff(wl) *
    delta_length``. With two 50/50 couplers the bar transmission is
    ``|sin(dphi/2)|**2`` and the cross transmission ``|cos(dphi/2)|**2`` (up to a
    common phase), giving the familiar MZI fringes.

    Parameters
    ----------
    delta_length : float
        Path-length imbalance between the two arms (µm).
    coupling : float
        Power cross-coupling of each splitter (0.5 = 50/50).
    loss_db_cm : float
        Propagation loss (dB/cm) applied to the extra ``|delta_length|`` of
        whichever arm is longer (the differential loss that reduces fringe
        contrast); the response is normalized to the shorter arm. The common
        arm length is not part of this closed form; for absolute per-arm loss
        use the circuit-assembled :func:`photonix.circuit.mzi`.

    Returns
    -------
    SDict
        Ports ``in0`` (input), ``out0`` (bar), ``out1`` (cross).

    Examples
    --------
    >>> import photonix as px
    >>> import numpy as np
    >>> wl = px.linspace(1.5, 1.6, 1001)
    >>> s = px.components.mzi(wl=wl, delta_length=40.0)
    >>> T = px.to_numpy(px.power(s[("in0","out0")]))
    >>> bool(T.max() > 0.9 and T.min() < 0.1)   # full-contrast fringes
    True
    """
    from .waveguide import neff_linear

    wl = xp.asarray(wl)
    n = neff_linear(wl, neff, ng, wl0)
    # Two arms differ by delta_length; common length cancels in interference.
    dphi = 2.0 * xp.pi / wl * n * delta_length

    c = xp.asarray(coupling)
    t = xp.sqrt(1.0 - c)          # through amplitude of each coupler
    k = xp.sqrt(c)                # cross amplitude magnitude

    # Transfer through two couplers with arm phases (0 and dphi). Using the
    # -j cross convention, the bar/cross field amplitudes are:
    #
    # Loss: the analytic model only knows the arm *imbalance* delta_length, so
    # propagation loss is applied to the extra length of whichever arm is
    # longer -- i.e. the response is normalized to the shorter arm's
    # transmission. This keeps |S| <= 1 and makes the observables symmetric
    # under delta_length -> -delta_length, matching the circuit-assembled
    # px.circuit.mzi exactly (up to the common shorter-arm factor, which is not
    # a parameter of this closed form). xp.maximum (not Python max/abs) keeps
    # the expression JAX-traceable.
    from photonix.core.units import db_per_cm_to_alpha_um

    alpha = db_per_cm_to_alpha_um(loss_db_cm)
    a_top = xp.exp(-alpha * xp.maximum(delta_length, 0.0))   # top longer (dL > 0)
    a_bot = xp.exp(-alpha * xp.maximum(-delta_length, 0.0))  # bottom longer (dL < 0)
    z = a_top * xp.exp(-1j * dphi)
    bar = (t * t) * z - (k * k) * a_bot          # in0 -> out0
    cross = -1j * t * k * (z + a_bot)            # in0 -> out1

    bar = xp.asarray(bar, dtype=complex)
    cross = xp.asarray(cross, dtype=complex)
    return {
        ("in0", "out0"): bar, ("out0", "in0"): bar,
        ("in0", "out1"): cross, ("out1", "in0"): cross,
    }
