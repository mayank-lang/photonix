"""Closed-form dispersion formulas evaluated with the backend array module.

Every function here is a pure, differentiable expression in the wavelength
``wl`` (micrometres) and, where it applies, the temperature ``T`` (kelvin). They
are the building blocks of :class:`photonix.materials.OpticalModel`; the
literature coefficients themselves live in :mod:`photonix.materials.library`.

Sign convention: functions that return a complex permittivity or index use the
*physics* sign ``N = n + i k`` with ``k >= 0`` for loss (``Im eps >= 0``), i.e.
the convention of refractiveindex.info, Palik and Meep. See
:func:`photonix.materials.to_engineering_sign` for the ``n - j k`` form the
``photonix.em`` field solvers use.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from photonix.core.backend import xp
from photonix.core.constants import C0, H_PLANCK, Q_E

__all__ = [
    "HC_EV_UM",
    "refractiveindex_formula",
    "lorentz_drude_epsilon",
    "li1980_silicon",
    "li1980_germanium",
]

#: ``h c`` in eV * um, so that photon energy [eV] = HC_EV_UM / wl [um] (CODATA 2018 exact constants).
HC_EV_UM: float = H_PLANCK * C0 / Q_E * 1e6


def _pairs(coeffs: Sequence[float], start: int) -> list[tuple[float, float]]:
    c = list(coeffs)
    return [(c[i], c[i + 1]) for i in range(start, len(c) - 1, 2)]


def refractiveindex_formula(number: int, coeffs: Sequence[float], wl: Any) -> Any:
    """Evaluate dispersion formula ``number`` (1-9) of the refractiveindex.info database.

    ``coeffs`` are ``C1, C2, ...`` exactly as listed on the database pages (and in
    its YAML files); ``wl`` is the vacuum wavelength in micrometres. The formulas
    are (``l`` = wavelength):

    1. Sellmeier: ``n^2 - 1 = C1 + sum C_i l^2 / (l^2 - C_{i+1}^2)``
    2. Sellmeier-2: ``n^2 - 1 = C1 + sum C_i l^2 / (l^2 - C_{i+1})``
    3. Polynomial: ``n^2 = C1 + sum C_i l^C_{i+1}``
    4. RefractiveIndex.INFO: ``n^2 = C1 + C2 l^C3/(l^2 - C4^C5) + C6 l^C7/(l^2 - C8^C9) + sum C_i l^C_{i+1}``
    5. Cauchy: ``n = C1 + sum C_i l^C_{i+1}``
    6. Gases: ``n - 1 = C1 + sum C_i / (C_{i+1} - l^-2)``
    7. Herzberger: ``n = C1 + C2/(l^2-0.028) + C3/(l^2-0.028)^2 + C4 l^2 + C5 l^4 + C6 l^6``
    8. Retro: ``(n^2 - 1)/(n^2 + 2) = C1 + C2 l^2/(l^2 - C3) + C4 l^2``
    9. Exotic: ``n^2 = C1 + C2/(l^2 - C3) + C4 (l - C5) / ((l - C5)^2 + C6)``

    Missing trailing coefficients are treated as zero, as on the database site.
    The result is the real refractive index.

    Examples
    --------
    Formula 1 with Malitson's fused-silica coefficients reproduces
    :func:`photonix.em.silica`:

    >>> import photonix.em as em
    >>> c = [0, 0.6961663, 0.0684043, 0.4079426, 0.1162414, 0.8974794, 9.896161]
    >>> abs(float(refractiveindex_formula(1, c, 1.55)) - float(em.silica(1.55))) < 1e-12
    True
    """
    lam = xp.asarray(wl, dtype=float)
    c = [float(v) for v in coeffs]
    l2 = lam * lam
    if number == 1:
        n2 = 1.0 + c[0] + sum(b * l2 / (l2 - cc * cc) for b, cc in _pairs(c, 1))
        return xp.sqrt(n2)
    if number == 2:
        n2 = 1.0 + c[0] + sum(b * l2 / (l2 - cc) for b, cc in _pairs(c, 1))
        return xp.sqrt(n2)
    if number == 3:
        n2 = c[0] + sum(a * lam ** p for a, p in _pairs(c, 1))
        return xp.sqrt(n2)
    if number == 4:
        c = (c + [0.0] * 17)[:17]
        n2 = c[0] + c[1] * lam ** c[2] / (l2 - c[3] ** c[4]) + c[5] * lam ** c[6] / (l2 - c[7] ** c[8])
        n2 = n2 + sum(a * lam ** p for a, p in _pairs(c, 9))
        return xp.sqrt(n2)
    if number == 5:
        return c[0] + sum(a * lam ** p for a, p in _pairs(c, 1)) + 0.0 * lam
    if number == 6:
        return 1.0 + c[0] + sum(a / (b - 1.0 / l2) for a, b in _pairs(c, 1))
    if number == 7:
        c = (c + [0.0] * 6)[:6]
        q = 1.0 / (l2 - 0.028)
        return c[0] + c[1] * q + c[2] * q * q + c[3] * l2 + c[4] * l2 * l2 + c[5] * l2 * l2 * l2
    if number == 8:
        c = (c + [0.0] * 4)[:4]
        a = c[0] + c[1] * l2 / (l2 - c[2]) + c[3] * l2
        return xp.sqrt((1.0 + 2.0 * a) / (1.0 - a))
    if number == 9:
        c = (c + [0.0] * 6)[:6]
        d = lam - c[4]
        n2 = c[0] + c[1] / (l2 - c[2]) + c[3] * d / (d * d + c[5])
        return xp.sqrt(n2)
    raise ValueError(f"refractiveindex.info formula number must be 1-9, got {number!r}")


def lorentz_drude_epsilon(wl: Any, plasma_ev: float, drude: tuple[float, float],
                          oscillators: Sequence[tuple[float, float, float]]) -> Any:
    """Lorentz-Drude permittivity in the parametrisation of Rakic et al. (1998).

    ``eps(w) = 1 - f0 wp^2 / (w (w + i G0)) + sum_j f_j wp^2 / (w_j^2 - w^2 - i w G_j)``

    with photon energy ``w = HC_EV_UM / wl`` and every parameter in eV. ``drude``
    is ``(f0, G0)``; ``oscillators`` is a sequence of ``(f_j, G_j, w_j)``. The
    result uses the physics sign (``Im eps >= 0`` is loss), the convention of the
    source.

    Reference: A. D. Rakic, A. B. Djurisic, J. M. Elazar and M. L. Majewski,
    "Optical properties of metallic films for vertical-cavity optoelectronic
    devices", Appl. Opt. 37, 5271-5283 (1998), doi:10.1364/AO.37.005271, Eq. (3).

    Examples
    --------
    A free-electron (Drude-only) metal is negative-real below its plasma energy:

    >>> eps = lorentz_drude_epsilon(1.55, 9.0, (1.0, 0.0), [])
    >>> float(eps.real) < 0.0
    True
    """
    w = HC_EV_UM / xp.asarray(wl, dtype=float)
    f0, g0 = drude
    wp2 = plasma_ev * plasma_ev
    eps = 1.0 - f0 * wp2 / (w * (w + 1j * g0))
    for fj, gj, wj in oscillators:
        eps = eps + fj * wp2 / (wj * wj - w * w - 1j * w * gj)
    return eps


def li1980_silicon(wl: Any, T: Any) -> Any:
    """Refractive index of crystalline silicon versus wavelength and temperature, Li (1980).

    H. H. Li, "Refractive index of silicon and germanium and its wavelength and
    temperature derivatives", J. Phys. Chem. Ref. Data 9, 561-658 (1980),
    doi:10.1063/1.555624::

        n^2(l, T) = eps(T) + L(T)/l^2 * (A0 + A1 T + A2 T^2)
        eps(T)    = 11.4445 + 2.7739e-4 T + 1.7050e-6 T^2 - 8.1347e-10 T^3
        L(T)      = exp(-3 dL(T)/L_293)           (lattice thermal expansion)
        A0, A1, A2 = 0.8948, 4.3977e-4, 7.3835e-8

    with ``dL/L_293`` [%] ``= -0.021 - 4.149e-5 T - 4.620e-8 T^2 + 1.482e-9 T^3``
    for ``T <= 293 K`` and ``-0.071 + 1.887e-4 T + 1.934e-7 T^2 - 4.544e-11 T^3``
    above. ``wl`` in micrometres, ``T`` in kelvin.

    Evidence: reproduces Li's tabulated recommended values (as transcribed in
    the refractiveindex.info database, "Li-100K" ... "Li-750K") to <= 6e-5 for
    100-350 K and <= 2e-4 up to 750 K over 1.2-14 um (``tests/test_materials_db.py``);
    the enforced range of :data:`photonix.materials.library.SI_LI1980` is that
    verified window. Its ``dn/dT`` at 1.55 um and 300 K is 1.84e-4 /K.

    Examples
    --------
    >>> round(float(li1980_silicon(1.55, 293.0)), 4)       # Li's table: 3.4757
    3.4757
    """
    lam = xp.asarray(wl, dtype=float)
    t = xp.asarray(T, dtype=float)
    eps = 11.4445 + 2.7739e-4 * t + 1.7050e-6 * t**2 - 8.1347e-10 * t**3
    low = -0.021 - 4.149e-5 * t - 4.620e-8 * t**2 + 1.482e-9 * t**3
    high = -0.071 + 1.887e-4 * t + 1.934e-7 * t**2 - 4.544e-11 * t**3
    dl = xp.where(t <= 293.0, low, high) * 1e-2
    lt = xp.exp(-3.0 * dl)
    n2 = eps + lt / (lam * lam) * (0.8948 + 4.3977e-4 * t + 7.3835e-8 * t**2)
    return xp.sqrt(n2)


def li1980_germanium(wl: Any, T: Any) -> Any:
    """Refractive index of crystalline germanium versus wavelength and temperature, Li (1980).

    Same reference as :func:`li1980_silicon`::

        n^2(l, T) = eps(T) + L(T)/l^2 * (A0 + A1 T + A2 T^2)
        eps(T)    = 15.2892 + 1.4549e-3 T + 3.5078e-6 T^2 - 1.2071e-9 T^3
        dL/L_293 [%] = -0.089 + 2.626e-4 u + 1.463e-6 u^2 - 2.221e-9 u^3,  u = T - 100
        A0, A1, A2 = 2.5381, 1.8260e-3, 2.8888e-6

    Evidence: reproduces Li's tabulated values ("Li-100K" ... "Li-550K") to
    <= 7e-5 for 100-500 K and <= 2e-4 at 550 K over 1.9-18 um, apart from isolated
    single-digit table outliers. Germanium absorbs strongly below ~1.9 um, where
    this real-index formula does not apply.

    Examples
    --------
    >>> round(float(li1980_germanium(2.0, 293.0)), 4)     # Li's table: 4.1008
    4.1008
    """
    lam = xp.asarray(wl, dtype=float)
    t = xp.asarray(T, dtype=float)
    eps = 15.2892 + 1.4549e-3 * t + 3.5078e-6 * t**2 - 1.2071e-9 * t**3
    u = t - 100.0
    dl = (-0.089 + 2.626e-4 * u + 1.463e-6 * u**2 - 2.221e-9 * u**3) * 1e-2
    lt = xp.exp(-3.0 * dl)
    n2 = eps + lt / (lam * lam) * (2.5381 + 1.8260e-3 * t + 2.8888e-6 * t**2)
    return xp.sqrt(n2)
