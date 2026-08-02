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

__all__ = ["straight", "bend", "bend_from_solver", "neff_linear"]

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
        n = neff_linear(wl, float(neff), float(ng), float(wl0))
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
