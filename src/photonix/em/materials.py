"""Dispersive refractive-index models (Sellmeier equations).

Each material is a differentiable callable ``n(wl_um)`` returning the refractive
index at the given wavelength(s) in micrometers. Coefficients are standard
literature Sellmeier fits valid around the telecom band.
"""
from __future__ import annotations

from photonix.core.backend import xp

__all__ = ["silica", "silicon", "silicon_nitride", "Material", "constant"]


def silica(wl):
    """Fused silica (SiO2) index, Malitson 1965 Sellmeier (valid 0.21-6.7 µm).

    >>> abs(float(silica(1.55)) - 1.444) < 2e-3
    True
    """
    l2 = xp.asarray(wl) ** 2
    n2 = 1.0 + 0.6961663 * l2 / (l2 - 0.0684043**2) \
        + 0.4079426 * l2 / (l2 - 0.1162414**2) \
        + 0.8974794 * l2 / (l2 - 9.896161**2)
    return xp.sqrt(n2)


def silicon(wl):
    """Crystalline silicon index, Salzberg & Villa / Li Sellmeier (1.36-11 µm).

    >>> abs(float(silicon(1.55)) - 3.4757) < 3e-3
    True
    """
    l2 = xp.asarray(wl) ** 2
    n2 = 1.0 + 10.6684293 * l2 / (l2 - 0.301516485**2) \
        + 0.0030434748 * l2 / (l2 - 1.13475115**2) \
        + 1.54133408 * l2 / (l2 - 1104.0**2)
    return xp.sqrt(n2)


def silicon_nitride(wl):
    """Stoichiometric Si3N4 index, Luke 2015 Sellmeier (0.31-5.5 µm).

    >>> abs(float(silicon_nitride(1.55)) - 1.996) < 3e-3
    True
    """
    l2 = xp.asarray(wl) ** 2
    n2 = 1.0 + 3.0249 * l2 / (l2 - 0.1353406**2) + 40314.0 * l2 / (l2 - 1239.842**2)
    return xp.sqrt(n2)


def constant(value: float):
    """Return a callable giving a wavelength-independent index ``value``."""

    def n(wl):
        return xp.asarray(value) * xp.ones_like(xp.asarray(wl, dtype=float))

    return n


class Material:
    """Named material wrapping an index callable ``n(wl)``.

    Examples
    --------
    >>> Si = Material("Si", silicon)
    >>> abs(float(Si.index(1.55)) - 3.4757) < 3e-3
    True
    """

    def __init__(self, name: str, index_fn):
        self.name = name
        self._fn = index_fn

    def index(self, wl):
        return self._fn(wl)

    def __call__(self, wl):
        return self._fn(wl)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Material({self.name!r})"


#: Convenient pre-built materials.
SI = Material("Si", silicon)
SIO2 = Material("SiO2", silica)
SIN = Material("Si3N4", silicon_nitride)
