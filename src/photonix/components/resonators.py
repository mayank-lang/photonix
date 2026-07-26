"""Ring/racetrack resonator building blocks and analytic references.

The circuit-assembled ring lives in :mod:`photonix.circuit`; the closed-form
expressions here are exact references used for validation and quick studies.
"""
from __future__ import annotations

from photonix.core.backend import xp
from photonix.core.types import AliasedSDict, SDict

__all__ = ["ring_coupler", "all_pass_ring", "add_drop_ring", "ADD_DROP_PORT_ALIASES"]

#: Legacy add-drop port names accepted on lookup -> canonical ``oN`` names.
ADD_DROP_PORT_ALIASES = {"i1": "o1", "t1": "o2", "d2": "o3"}


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
    """Analytic four-port add-drop ring with two bus couplers.

    Ports
    -----
    ``o1``
        Input (bus 1).
    ``o2``
        Through (bus 1).
    ``o3``
        Drop (bus 2).
    ``o4``
        Add (bus 2).

    The legacy three-port names ``i1``/``t1``/``d2`` remain valid *lookup*
    aliases for ``o1``/``o2``/``o3``.

    Physics
    -------
    Light entering ``o1`` circulates one way around the ring; light entering the
    add port ``o4`` circulates the *other* way. The two senses of circulation are
    decoupled, so ``o1 -> o4`` and ``o2 -> o3`` vanish while the add branch
    reuses the same denominator with the two couplers exchanged::

        D          = 1 - t1 t2 a e^{-j phi}
        o1 -> o2   = (t1 - t2 a e^{-j phi}) / D          (through)
        o1 -> o3   = -k1 k2 sqrt(a) e^{-j phi/2} / D     (drop)
        o4 -> o3   = (t2 - t1 a e^{-j phi}) / D          (through, bus 2)
        o4 -> o2   = o1 -> o3                            (reciprocity)

    Earlier releases returned only the three ``i1``/``t1``/``d2`` terminals,
    which left the add port undefined and made the model unusable as a real
    four-port in a circuit.

    Examples
    --------
    >>> import photonix as px
    >>> import numpy as np
    >>> wl = px.linspace(1.54, 1.56, 4001)
    >>> s = px.components.add_drop_ring(wl=wl, coupling1=0.2, coupling2=0.2)
    >>> D = px.to_numpy(px.power(s[("o1","o3")]))
    >>> bool(D.max() > 0.2)   # power appears at the drop port on resonance
    True
    >>> sorted(px.core.ports_of(s))            # a genuine four-port
    ['o1', 'o2', 'o3', 'o4']
    >>> bool(np.allclose(px.to_numpy(s[("i1","d2")]), px.to_numpy(s[("o1","o3")])))
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
    through = xp.asarray((t1 - t2 * a * z) / denom, dtype=complex)
    through_add = xp.asarray((t2 - t1 * a * z) / denom, dtype=complex)
    drop = xp.asarray((-k1 * k2 * xp.sqrt(a) * xp.exp(-1j * phi / 2.0)) / denom, dtype=complex)
    return AliasedSDict(
        {
            # Bus 1 forward: in -> through, in -> drop.
            ("o1", "o2"): through, ("o2", "o1"): through,
            ("o1", "o3"): drop, ("o3", "o1"): drop,
            # Bus 2 (add port), counter-circulating: add -> drop, add -> through.
            ("o4", "o3"): through_add, ("o3", "o4"): through_add,
            ("o4", "o2"): drop, ("o2", "o4"): drop,
        },
        aliases=ADD_DROP_PORT_ALIASES,
    )
