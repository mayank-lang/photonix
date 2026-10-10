"""Waveguide component models (straight and bend).

All models are pure functions returning an :data:`~photonix.core.types.SDict`,
differentiable and ``jit``-able. Lengths and wavelengths are in micrometers.

Ports
-----
``o1`` (input) and ``o2`` (output).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from photonix.core.backend import xp
from photonix.core.constants import N_GROUP_SI_STRIP, WL_DEFAULT
from photonix.core.types import SDict
from photonix.core.units import db_per_cm_to_alpha_um

__all__ = [
    "straight", "bend", "bend_from_solver", "neff_linear",
    "bend_euler", "euler_bend_length", "taper", "waveguide_route",
]

IndexLike: TypeAlias = float | Callable[..., object]


def _eval_index(value, wl):
    """Evaluate a refractive index given as a number or a callable of ``wl``."""
    if callable(value):
        return xp.asarray(value(wl))
    return xp.asarray(value)


def neff_linear(wl, neff: float, ng: float, wl0: float):
    """First-order dispersive effective index.

    ``n_eff(wl) = neff - (ng - neff) * (wl - wl0) / wl0``

    This is the standard linearization that makes the *group* index ``ng`` come
    out correct at ``wl0`` while keeping a single analytic expression.
    """
    wl = xp.asarray(wl)
    return neff - (ng - neff) * (wl - wl0) / wl0


def straight(
    *,
    wl=WL_DEFAULT,
    length: float = 10.0,
    neff: IndexLike = 2.4,
    ng: float = N_GROUP_SI_STRIP,
    loss_db_cm: float = 0.0,
    wl0: float = WL_DEFAULT,
) -> SDict:
    """Straight waveguide of physical ``length`` (µm).

    The transmission amplitude is ``exp(-alpha*L) * exp(-j*beta*L)`` with
    ``beta = 2*pi/wl * n_eff(wl)`` and ``alpha`` derived from ``loss_db_cm``.

    Parameters
    ----------
    wl : float or array
        Wavelength(s) in µm.
    length : float
        Waveguide length in µm.
    neff : float or callable
        Effective index at ``wl0`` (a number) or a callable ``neff(wl)``. If a
        callable is given, ``ng``/``wl0`` are ignored for the phase.
    ng : float
        Group index at ``wl0`` (used only when ``neff`` is a number).
    loss_db_cm : float
        Propagation loss in dB/cm.
    wl0 : float
        Reference wavelength in µm for the linear dispersion model.

    Returns
    -------
    SDict
        ``{("o1","o2"): t, ("o2","o1"): t}`` with complex transmission ``t``.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.straight(wl=1.55, length=100.0)
    >>> bool(abs(px.power(s[("o1","o2")]) - 1.0) < 1e-9)
    True
    """
    wl = xp.asarray(wl)
    if callable(neff):
        n = _eval_index(neff, wl)
    else:
        # No float(): neff/ng/wl0 may be JAX tracers (grad/jit over waveguide parameters).
        n = neff_linear(wl, neff, ng, wl0)
    beta = 2.0 * xp.pi / wl * n
    alpha = db_per_cm_to_alpha_um(loss_db_cm)
    t = xp.exp(-alpha * length) * xp.exp(-1j * beta * length)
    return {("o1", "o2"): t, ("o2", "o1"): t}


def bend(
    *,
    wl=WL_DEFAULT,
    radius: float = 5.0,
    angle: float = 90.0,
    neff: IndexLike = 2.4,
    ng: float = N_GROUP_SI_STRIP,
    loss_db_cm: float = 0.0,
    excess_loss_db: float = 0.0,
    wl0: float = WL_DEFAULT,
) -> SDict:
    """Circular waveguide bend.

    Models a bend as a straight section of arc length ``radius*angle`` plus a
    lumped ``excess_loss_db`` (bend/transition loss). Ports ``o1``/``o2``.

    .. note::

       ``excess_loss_db`` **defaults to 0** — a lossless bend of any radius.
       For physically accurate bend loss, use :func:`bend_from_solver` which
       calls :func:`~photonix.em.fde_vector.bend_loss_fullvector` to compute
       the real radiation loss (see PHYSICS_AUDIT §C7).

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.bend(wl=1.55, radius=5.0, angle=90.0)
    >>> set(s) == {("o1","o2"), ("o2","o1")}
    True
    """
    wl = xp.asarray(wl)
    arc = abs(radius * angle * xp.pi / 180.0)
    base = straight(wl=wl, length=arc, neff=neff, ng=ng, loss_db_cm=loss_db_cm, wl0=wl0)
    excess = 10.0 ** (-excess_loss_db / 20.0)
    return {k: excess * v for k, v in base.items()}


def bend_from_solver(
    *,
    wl: float = WL_DEFAULT,
    radius: float = 5.0,
    angle: float = 90.0,
    width: float = 0.5,
    thickness: float = 0.22,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    ng: float = N_GROUP_SI_STRIP,
    loss_db_cm: float = 0.0,
    wl0: float = WL_DEFAULT,
    **solver_kwargs,
) -> SDict:
    """Waveguide bend with rigorous bend loss from the full-vector solver.

    Calls :func:`~photonix.em.fde_vector.bend_loss_fullvector` to compute the
    radiation loss and effective index shift, then passes them into :func:`bend`.
    This bridges the compact model to the EM solver (see PHYSICS_AUDIT §C7).

    Parameters
    ----------
    width, thickness, n_core, n_clad
        Waveguide cross-section parameters passed to the full-vector solver.
    **solver_kwargs
        Forwarded to ``bend_loss_fullvector`` (e.g. ``resolution``, ``pml``).

    Returns
    -------
    SDict
        Same as :func:`bend`, with ``excess_loss_db`` and ``neff`` computed
        from the rigorous solver.

    Notes
    -----
    The loss is scaled linearly with swept angle magnitude
    (``loss_per_90 * abs(angle)/90``). This is correct for distributed
    radiation loss but does not capture
    straight-to-bend junction (transition) loss, which is angle-independent.
    For short bends where junction loss dominates, consider a full FDTD
    simulation instead.

    Examples
    --------
    >>> import photonix as px                                    # doctest: +SKIP
    >>> s = px.components.bend_from_solver(radius=5.0, angle=90.0)  # doctest: +SKIP
    >>> set(s) == {("o1","o2"), ("o2","o1")}                     # doctest: +SKIP
    True
    """
    from photonix.em.fde_vector import bend_loss_fullvector

    result = bend_loss_fullvector(
        width=width, thickness=thickness, bend_radius=radius,
        wl=wl, n_core=n_core, n_clad=n_clad, **solver_kwargs,
    )
    return bend(
        wl=wl, radius=radius, angle=angle,
        neff=float(result.n_eff.real), ng=ng,
        loss_db_cm=loss_db_cm,
        # Bend handedness changes the layout direction, not the radiated power.
        # A signed angle used here previously turned a clockwise bend's positive
        # solver loss into optical gain.
        excess_loss_db=result.loss_db_per_90deg * (abs(angle) / 90.0),
        wl0=wl0,
    )


def euler_bend_length(radius: float, angle: float, p: float = 0.5):
    """Path length of an Euler (clothoid) bend in µm.

    The bend turns ``angle`` degrees with a minimum radius of curvature
    ``radius``. A fraction ``p`` of the turn is made in two linear-curvature
    (clothoid) ramps, the remaining ``1 - p`` on a circular arc of ``radius``.
    Each ramp grows the curvature from 0 to ``1/radius`` over a length
    ``L_c = p * theta * radius``, so the closed form is
    ``L = radius * |theta| * (1 + p)``. ``p = 0`` is a circular bend and
    ``p = 1`` a full Euler bend twice the length of the same-radius arc.

    Examples
    --------
    >>> import math
    >>> abs(float(euler_bend_length(5.0, 90.0, 0.0)) - 5.0 * math.pi / 2) < 1e-12
    True
    >>> abs(float(euler_bend_length(5.0, 90.0, 1.0)) - 5.0 * math.pi) < 1e-12
    True
    """
    return radius * xp.abs(xp.asarray(angle)) * xp.pi / 180.0 * (1.0 + p)


def bend_euler(
    *,
    wl=WL_DEFAULT,
    radius: float = 5.0,
    angle: float = 90.0,
    p: float = 0.5,
    neff: IndexLike = 2.4,
    ng: float = N_GROUP_SI_STRIP,
    loss_db_cm: float = 0.0,
    excess_loss_db: float = 0.0,
    wl0: float = WL_DEFAULT,
) -> SDict:
    """Euler (clothoid) waveguide bend. Ports ``o1``/``o2``.

    The optical path is :func:`euler_bend_length` long. ``radius`` is the
    *minimum* radius of curvature (which sets the radiation loss); compared with
    a circular bend of the same minimum radius, the curvature ramps suppress
    the straight-to-bend mode-mismatch loss at the cost of a longer path. As for
    :func:`bend`, ``excess_loss_db`` is a lumped loss that defaults to zero.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.bend_euler(radius=5.0, angle=90.0, p=0.5)
    >>> bool(abs(px.power(s[("o1", "o2")]) - 1.0) < 1e-12)
    True
    """
    length = euler_bend_length(radius, angle, p)
    base = straight(wl=wl, length=length, neff=neff, ng=ng, loss_db_cm=loss_db_cm, wl0=wl0)
    excess = 10.0 ** (-excess_loss_db / 20.0)
    return {k: excess * v for k, v in base.items()}


# 5-point Gauss-Legendre nodes/weights on [0, 1]: exact for polynomials of
# degree <= 9, i.e. far beyond the curvature of a smooth n_eff(width) fit.
_GL_NODES = (0.04691007703066800, 0.23076534494715845, 0.5,
             0.76923465505284155, 0.95308992296933200)
_GL_WEIGHTS = (0.11846344252809454, 0.23931433524968324, 0.28444444444444444,
               0.23931433524968324, 0.11846344252809454)


def taper(
    *,
    wl=WL_DEFAULT,
    length: float = 10.0,
    width1: float = 0.5,
    width2: float = 1.0,
    neff: IndexLike = 2.4,
    ng: float = N_GROUP_SI_STRIP,
    loss_db_cm: float = 0.0,
    excess_loss_db: float = 0.0,
    wl0: float = WL_DEFAULT,
) -> SDict:
    """Compact adiabatic linear taper from ``width1`` (o1) to ``width2`` (o2).

    An adiabatic taper converts the fundamental mode without reflection, so its
    transmission is ``exp(-j * 2*pi/wl * integral(n_eff(w(z)) dz))`` times loss.
    ``neff`` may be

    * a number -- the phase uses the linear-dispersion model of :func:`straight`
      and the widths only document the geometry; or
    * a callable ``neff(wl, width)`` -- e.g. a fit to mode-solver results --
      integrated along the linear width profile with 5-point Gauss-Legendre
      quadrature.

    Non-adiabatic conversion and reflection are not modelled; use the EME-backed
    :func:`photonix.em.components.taper` to quantify them.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.taper(length=10.0, neff=lambda wl, w: 2.0 + 0.4 * w)
    >>> import cmath, math
    >>> # mean index of 2.0 + 0.4 * w over w in [0.5, 1.0] is 2.3
    >>> phi = -cmath.phase(complex(s[("o1", "o2")])) % (2 * math.pi)
    >>> abs(phi - (2 * math.pi / 1.55 * 2.3 * 10.0) % (2 * math.pi)) < 1e-9
    True
    """
    wl = xp.asarray(wl)
    if callable(neff):
        n_mean = 0.0
        for node, weight in zip(_GL_NODES, _GL_WEIGHTS, strict=True):
            n_mean = n_mean + weight * xp.asarray(neff(wl, width1 + node * (width2 - width1)))
        base = straight(wl=wl, length=length, neff=lambda _wl: n_mean, loss_db_cm=loss_db_cm)
    else:
        base = straight(wl=wl, length=length, neff=neff, ng=ng, loss_db_cm=loss_db_cm, wl0=wl0)
    excess = 10.0 ** (-excess_loss_db / 20.0)
    return {k: excess * v for k, v in base.items()}


def waveguide_route(
    *,
    wl=WL_DEFAULT,
    length: float = 10.0,
    n_bends: int = 0,
    bend_loss_db: float = 0.0,
    neff: IndexLike = 2.4,
    ng: float = N_GROUP_SI_STRIP,
    loss_db_cm: float = 0.0,
    wl0: float = WL_DEFAULT,
) -> SDict:
    """A routed waveguide: total centre-line ``length`` (µm) with ``n_bends`` bends.

    This is the model behind :func:`photonix.layout.route` and
    :func:`photonix.layout.route_manhattan` cells, whose layout records the
    drawn path length and bend count. The phase and distributed loss follow
    :func:`straight` over the full path; each bend adds a lumped
    ``bend_loss_db``.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.waveguide_route(length=100.0, n_bends=4, bend_loss_db=0.05)
    >>> bool(abs(float(px.power(s[("o1", "o2")])) - 10 ** (-0.02)) < 1e-12)
    True
    """
    base = straight(wl=wl, length=length, neff=neff, ng=ng, loss_db_cm=loss_db_cm, wl0=wl0)
    excess = 10.0 ** (-(n_bends * bend_loss_db) / 20.0)
    return {k: excess * v for k, v in base.items()}
