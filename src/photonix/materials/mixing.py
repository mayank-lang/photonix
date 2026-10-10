"""Effective-medium mixing of two isotropic materials (Lorentz-Lorenz / Clausius-Mossotti).

For a homogeneous mixture of two polarisable constituents with volume fraction
``f`` of ``b`` (H. A. Lorentz, 1880; L. Lorenz, 1880; see M. Born and E. Wolf,
*Principles of Optics*, 7th ed., Sec. 2.3)::

    (eps - 1) / (eps + 2) = (1 - f) (eps_a - 1) / (eps_a + 2) + f (eps_b - 1) / (eps_b + 2)

This is the usual first model of silicon oxynitride, SiO_xN_y, as a molecular
mixture of SiO2 and Si3N4. It is exact at ``f = 0`` and ``f = 1`` and linear in
the polarisability density, but it is *not* calibrated to any deposition
process: hydrogen content, porosity and non-stoichiometry move deposited films
by several 1e-2 in index. Treat the composition as a fitting parameter matched
to a measured index, not as a prediction of film chemistry.
"""
from __future__ import annotations

from typing import Any

from photonix.core.backend import xp

from .model import MaterialSpec, OpticalModel, Reference, _intersect

__all__ = ["lorentz_lorenz_mix", "sion"]

_LL_REF = Reference("Lorentz-Lorenz (Clausius-Mossotti) mixing rule; M. Born and E. Wolf, Principles of Optics, "
                    "7th ed. (Cambridge University Press, 1999), Sec. 2.3",
                    notes="effective-medium model, not calibrated to deposited films")


def _t_overlap(a: tuple[float, float] | None, b: tuple[float, float] | None) -> tuple[float, float] | None:
    if a is None or b is None:
        return a or b
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if lo > hi:
        raise ValueError(f"temperature windows {a} and {b} do not overlap")
    return (lo, hi)


def lorentz_lorenz_mix(a: MaterialSpec, b: MaterialSpec, fraction_b: float, *, name: str | None = None) -> MaterialSpec:
    """Lorentz-Lorenz mixture of isotropic materials ``a`` and ``b`` (volume fraction ``fraction_b`` of ``b``).

    The wavelength and temperature windows are the overlaps of the two
    constituents'. The mixture carries no thermal or electrical data.

    Examples
    --------
    >>> from photonix.materials import get_material, lorentz_lorenz_mix
    >>> sio2, sin = get_material("SiO2"), get_material("Si3N4")
    >>> mix = lorentz_lorenz_mix(sio2, sin, 0.5)
    >>> 1.444 < float(mix.index(1.55).real) < 1.997
    True
    >>> float(lorentz_lorenz_mix(sio2, sin, 0.0).index(1.55).real) == float(sio2.index(1.55).real)
    True
    """
    f = float(fraction_b)
    if not 0.0 <= f <= 1.0:
        raise ValueError("fraction_b must be in [0, 1]")
    if not (a.isotropic and b.isotropic):
        raise ValueError("lorentz_lorenz_mix needs isotropic constituents")
    assert isinstance(a.optical, OpticalModel) and isinstance(b.optical, OpticalModel)
    ma, mb = a.optical, b.optical

    def evaluate(wl: Any, T: Any) -> Any:
        na = ma.evaluate(wl, T) + 0j
        nb = mb.evaluate(wl, T) + 0j
        if f == 0.0:
            return na
        if f == 1.0:
            return nb
        ea, eb = na * na, nb * nb
        q = (1.0 - f) * (ea - 1.0) / (ea + 2.0) + f * (eb - 1.0) / (eb + 2.0)
        return xp.sqrt((1.0 + 2.0 * q) / (1.0 - q))

    t_ref = ma.reference_temperature_k if ma.reference_temperature_k is not None else mb.reference_temperature_k
    optical = OpticalModel("lorentz_lorenz_mix", evaluate, ma.references + mb.references + (_LL_REF,),
                           _intersect(ma.wl_range_um, mb.wl_range_um),
                           _t_overlap(ma.temperature_range_k, mb.temperature_range_k), t_ref,
                           ma.temperature_dependent or mb.temperature_dependent,
                           {"a": a.name, "b": b.name, "fraction_b": f}, f"{a.name}:{b.name} LL mix")
    return MaterialSpec(name or f"{a.name}({1 - f:.3g})+{b.name}({f:.3g})", optical, tags=("mixture",),
                        description=f"Lorentz-Lorenz mixture, {f:.3g} volume fraction {b.name} in {a.name}.")


def sion(nitride_fraction: float) -> MaterialSpec:
    """Silicon oxynitride as a Lorentz-Lorenz mixture of the library ``SiO2`` and ``Si3N4``.

    Examples
    --------
    >>> from photonix.materials import sion
    >>> round(float(sion(1.0).index(1.55).real), 4)        # pure Luke (2015) Si3N4
    1.9963
    """
    from .library import SI3N4, SIO2

    return lorentz_lorenz_mix(SIO2, SI3N4, nitride_fraction, name=f"SiON(x={float(nitride_fraction):.3g})")
