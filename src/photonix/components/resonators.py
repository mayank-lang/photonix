"""Ring/racetrack resonator building blocks and analytic references.

The circuit-assembled ring lives in :mod:`photonix.circuit`; the closed-form
expressions here are exact references used for validation and quick studies.
"""
from __future__ import annotations

from photonix.core.backend import xp
from photonix.core.types import AliasedSDict, SDict

__all__ = [
    "ring_coupler", "all_pass_ring", "add_drop_ring", "ADD_DROP_PORT_ALIASES",
    "ring_single", "ring_double",
]

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
    c = xp.asarray(coupling)
    t = xp.sqrt(1.0 - c)
    z = xp.exp(-1j * phi)
    u = a * z
    # Algebraically this is (t-u)/(1-t*u), but the feedback form below has two
    # important numerical/physical advantages.  c/(1+t) evaluates 1-t without
    # catastrophic cancellation for weak coupling, and an exactly uncoupled
    # lossless ring has the correct H=1 limit instead of the removable 0/0 that
    # occurs at a round-trip resonance.
    one_minus_t = c / (1.0 + t)
    denom = (1.0 - u) + u * one_minus_t
    removable_singularity = (c == 0.0) & (xp.abs(denom) == 0.0)
    denom = xp.where(removable_singularity, xp.ones_like(denom), denom)
    H = t - c * u / denom
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
    c1 = xp.asarray(coupling1)
    c2 = xp.asarray(coupling2)
    t1 = xp.sqrt(1.0 - c1)
    t2 = xp.sqrt(1.0 - c2)
    k1 = xp.sqrt(c1)
    k2 = xp.sqrt(c2)
    z = xp.exp(-1j * phi)
    u = a * z
    # Stable feedback denominator.  Computing 1-t1*t2 directly loses the
    # linewidth when either power coupling is tiny; c/(1+t) is the accurate
    # identity 1-sqrt(1-c).  With both couplers exactly off, the ring is
    # disconnected and each bus must transmit with unit amplitude, including
    # at wavelengths where the inaccessible ring is resonant.
    one_minus_t1 = c1 / (1.0 + t1)
    one_minus_t2 = c2 / (1.0 + t2)
    one_minus_t1t2 = one_minus_t1 + t1 * one_minus_t2
    denom = (1.0 - u) + u * one_minus_t1t2
    uncoupled = (c1 == 0.0) & (c2 == 0.0)
    removable_singularity = uncoupled & (xp.abs(denom) == 0.0)
    denom = xp.where(removable_singularity, xp.ones_like(denom), denom)
    # These feedback forms are exactly equivalent to
    # (t1-t2*u)/denom and (t2-t1*u)/denom, respectively, while exposing
    # the removable zero-coupling limit explicitly.
    through = xp.asarray(t1 - c1 * t2 * u / denom, dtype=complex)
    through_add = xp.asarray(t2 - c2 * t1 * u / denom, dtype=complex)
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


def _ring_parts(wl, radius, gap, length_x, neff, ng, wl0, loss_db_cm, splitting):
    """Bus-ring coupler S-dict and the uncoupled ring-path transmission."""
    from photonix.core.units import db_per_cm_to_alpha_um

    from .couplers import evanescent_coupler
    from .waveguide import neff_linear

    cpl = evanescent_coupler(wl=wl, gap=gap, length=length_x, bend_radius=radius, curved_arms=1,
                             splitting=splitting, neff=neff, ng=ng, wl0=wl0, loss_db_cm=loss_db_cm)
    wl = xp.asarray(wl)
    n = neff_linear(wl, neff, ng, wl0)
    alpha = db_per_cm_to_alpha_um(loss_db_cm)
    return cpl, n, alpha


def ring_single(
    *,
    wl=1.55,
    radius: float = 10.0,
    gap: float = 0.2,
    length_x: float = 0.0,
    neff: float = 2.4,
    ng: float = 4.2,
    wl0: float = 1.55,
    loss_db_cm: float = 2.0,
    splitting=None,
) -> SDict:
    """All-pass ring/racetrack with geometry-derived coupling. Ports ``o1`` -> ``o2``.

    The bus-ring coupler is :func:`~photonix.components.evanescent_coupler`
    with ``length = length_x`` and the ring's curvature (``bend_radius =
    radius``, one curved arm), so coupling follows from ``gap``. With the
    coupler's through/cross amplitudes ``t``/``k`` and the remaining ring path
    ``A = a exp(-j phi)`` (length ``2 pi R + length_x``; the coupler holds the
    other straight), the through response is the exact feedback result::

        H = t + k^2 A / (1 - t A)

    which reduces to ``(t - A)/(1 - t A)`` for a real ``t``.

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> wl = px.linspace(1.54, 1.56, 2001)
    >>> T = px.to_numpy(px.power(px.components.ring_single(wl=wl, gap=0.15)[("o1", "o2")]))
    >>> bool(T.min() < 0.9 and T.max() <= 1.0)
    True
    """
    cpl, n, alpha = _ring_parts(wl, radius, gap, length_x, neff, ng, wl0, loss_db_cm, splitting)
    wl = xp.asarray(wl)
    l_ring = 2.0 * xp.pi * radius + length_x
    a_path = xp.exp(-alpha * l_ring) * xp.exp(-1j * 2.0 * xp.pi / wl * n * l_ring)
    t = cpl[("o1", "o4")]
    k = cpl[("o1", "o3")]
    h = xp.asarray(t + k * k * a_path / (1.0 - t * a_path), dtype=complex)
    return {("o1", "o2"): h, ("o2", "o1"): h}


def ring_double(
    *,
    wl=1.55,
    radius: float = 10.0,
    gap: float = 0.2,
    gap2: float | None = None,
    length_x: float = 0.0,
    neff: float = 2.4,
    ng: float = 4.2,
    wl0: float = 1.55,
    loss_db_cm: float = 2.0,
    splitting=None,
) -> SDict:
    """Add-drop ring/racetrack with geometry-derived couplers.

    Ports as in :func:`add_drop_ring`: ``o1`` input, ``o2`` through, ``o3``
    drop, ``o4`` add. Each bus coupler is an
    :func:`~photonix.components.evanescent_coupler` (``gap`` and ``gap2``),
    and the two half-rings between them are ``pi R`` long. The network is
    solved exactly with :func:`photonix.circuit.evaluate_circuit`.

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> wl = px.linspace(1.54, 1.56, 2001)
    >>> s = px.components.ring_double(wl=wl, gap=0.2)
    >>> D = px.to_numpy(px.power(s[("o1", "o3")]))
    >>> bool(D.max() > 0.5)
    True
    """
    from photonix.circuit.solver import evaluate_circuit

    from .waveguide import straight

    g2 = gap if gap2 is None else gap2
    c1, _n, _a = _ring_parts(wl, radius, gap, length_x, neff, ng, wl0, loss_db_cm, splitting)
    c2, _n, _a = _ring_parts(wl, radius, g2, length_x, neff, ng, wl0, loss_db_cm, splitting)
    half = straight(wl=wl, length=xp.pi * radius, neff=neff, ng=ng, loss_db_cm=loss_db_cm, wl0=wl0)
    # Coupler ports: o1/o4 bus, o2/o3 ring side (bar o1->o4, cross o1->o3).
    # Lower coupler: bus left->right; its ring-side o3 (right) feeds the right
    # half-ring up to the upper coupler, whose bus runs right->left.
    instances = {"c1": c1, "c2": c2, "right": half, "left": half}
    connections = {
        ("c1", "o3"): ("right", "o1"),
        ("right", "o2"): ("c2", "o2"),
        ("c2", "o3"): ("left", "o1"),
        ("left", "o2"): ("c1", "o2"),
    }
    ports = {"o1": ("c1", "o1"), "o2": ("c1", "o4"), "o4": ("c2", "o1"), "o3": ("c2", "o4")}
    return evaluate_circuit(instances, connections, ports)
