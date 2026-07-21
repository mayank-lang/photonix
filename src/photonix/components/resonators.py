"""Ring/racetrack resonator building blocks and analytic references.

The circuit-assembled ring lives in :mod:`photonix.circuit`; the closed-form
expressions here are exact references used for validation and quick studies.
"""
from __future__ import annotations

from photonix.core.backend import xp
from photonix.core.types import SDict

__all__ = ["ring_coupler", "all_pass_ring", "add_drop_ring"]


def ring_coupler(*, wl=1.55, coupling: float = 0.1) -> SDict:
    """Point coupler between a bus and a ring (a 2x2 directional coupler).

    Thin wrapper over :func:`photonix.components.directional_coupler` kept under a
    descriptive name for resonator circuits.
    """
    from .couplers import directional_coupler

    return directional_coupler(wl=wl, coupling=coupling)


def _round_trip(wl, neff, ng, wl0, circumference, loss_db_cm):
    from photonix.core.units import db_per_cm_to_alpha_um

    from .waveguide import neff_linear

    wl = xp.asarray(wl)
    n = neff_linear(wl, neff, ng, wl0)
    phi = 2.0 * xp.pi / wl * n * circumference
    alpha = db_per_cm_to_alpha_um(loss_db_cm)
    a = xp.exp(-alpha * circumference)  # round-trip amplitude
    return a, phi


def all_pass_ring(
    *,
    wl=1.55,
    coupling: float = 0.1,
    radius: float = 10.0,
    neff: float = 2.4,
    ng: float = 4.2,
    wl0: float = 1.55,
    loss_db_cm: float = 2.0,
) -> SDict:
    """Analytic all-pass (single-bus) ring resonator. Ports ``o1`` -> ``o2``.

    Through-port transfer function::

        H = (t - a e^{-j phi}) / (1 - t a e^{-j phi})

    with self-coupling ``t = sqrt(1 - coupling)``, round-trip amplitude ``a`` and
    phase ``phi``. On resonance the through port shows a dip (notch).

    Examples
    --------
    >>> import photonix as px
    >>> import numpy as np
    >>> wl = px.linspace(1.54, 1.56, 2001)
    >>> s = px.components.all_pass_ring(wl=wl, coupling=0.2, radius=10.0)
    >>> T = px.to_numpy(px.power(s[("o1","o2")]))
    >>> bool(T.min() < 0.95)   # a visible resonant dip
    True
    """
    circumference = 2.0 * xp.pi * radius
    a, phi = _round_trip(wl, neff, ng, wl0, circumference, loss_db_cm)
    t = xp.sqrt(1.0 - xp.asarray(coupling))
    z = xp.exp(-1j * phi)
    H = (t - a * z) / (1.0 - t * a * z)
    return {("o1", "o2"): H, ("o2", "o1"): H}


def add_drop_ring(
    *,
    wl=1.55,
    coupling1: float = 0.1,
    coupling2: float = 0.1,
    radius: float = 10.0,
    neff: float = 2.4,
    ng: float = 4.2,
    wl0: float = 1.55,
    loss_db_cm: float = 2.0,
) -> SDict:
    """Analytic add-drop ring with two bus couplers.

    Ports: ``i1`` (input) / ``t1`` (through) on bus 1, ``d2`` (drop) on bus 2.
    Through and drop transfer functions follow the standard add-drop equations.

    Examples
    --------
    >>> import photonix as px
    >>> import numpy as np
    >>> wl = px.linspace(1.54, 1.56, 4001)
    >>> s = px.components.add_drop_ring(wl=wl, coupling1=0.2, coupling2=0.2)
    >>> D = px.to_numpy(px.power(s[("i1","d2")]))
    >>> bool(D.max() > 0.2)   # power appears at the drop port on resonance
    True
    """
    circumference = 2.0 * xp.pi * radius
    a, phi = _round_trip(wl, neff, ng, wl0, circumference, loss_db_cm)
    t1 = xp.sqrt(1.0 - xp.asarray(coupling1))
    t2 = xp.sqrt(1.0 - xp.asarray(coupling2))
    k1 = xp.sqrt(xp.asarray(coupling1))
    k2 = xp.sqrt(xp.asarray(coupling2))
    z = xp.exp(-1j * phi)
    denom = 1.0 - t1 * t2 * a * z
    through = (t1 - t2 * a * z) / denom
    drop = (-k1 * k2 * xp.sqrt(a) * xp.exp(-1j * phi / 2.0)) / denom
    return {
        ("i1", "t1"): through, ("t1", "i1"): through,
        ("i1", "d2"): drop, ("d2", "i1"): drop,
    }
