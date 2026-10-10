"""Free-carrier (plasma-dispersion) index and absorption changes in crystalline silicon.

Two empirical fits are provided, both returning ``(dn, dalpha)`` where ``dn``
is the change of the real index and ``dalpha`` the change of the *power*
absorption coefficient in 1/cm, for excess electron/hole densities in cm^-3:

* :func:`soref_bennett_1987` -- R. A. Soref and B. R. Bennett, "Electrooptical
  effects in silicon", IEEE J. Quantum Electron. 23, 123-129 (1987),
  doi:10.1109/JQE.1987.1073206, 1.55 um fits::

      dn     = -(8.8e-22 dNe + 8.5e-18 dNh^0.8)
      dalpha =   8.5e-18 dNe + 6.0e-18 dNh

  scaled by the Drude ``(wl / 1.55)^2`` law. The same paper's separate 1.3 um
  fits (``6.2e-22``, ``6.0e-18``, ``6.0e-18``, ``4.0e-18``) agree with that
  scaling to <= 5 % (checked in the tests), which sets the 1.3-1.6 um window.
* :func:`nedeljkovic_2011` -- M. Nedeljkovic, R. Soref and G. Z. Mashanovich,
  "Free-carrier electrorefraction and electroabsorption modulation predictions
  for silicon over the 1-14 um infrared wavelength range", IEEE Photonics J. 3,
  1171-1180 (2011), doi:10.1109/JPHOT.2011.2171930, 1.55 um fits::

      dn     = -5.4e-22 dNe^1.011 - 1.53e-18 dNh^0.838
      dalpha =  8.88e-21 dNe^1.167 + 5.84e-20 dNh^1.109

  The paper fits each wavelength separately; only the 1.55 um coefficients
  are implemented, with ``(wl / 1.55)^2`` scaling inside a 1.5-1.6 um window.

The fits come from absorption data on doped bulk silicon at room temperature;
they are not temperature dependent. :func:`doped_si` wraps either fit into a
:class:`~photonix.materials.MaterialSpec`.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from photonix.core.backend import xp

from .model import MaterialRangeError, MaterialSpec, OpticalModel, Reference, _intersect

__all__ = [
    "soref_bennett_1987",
    "nedeljkovic_2011",
    "FREE_CARRIER_MODELS",
    "doped_si",
    "alpha_to_k",
]

_SB_REF = Reference("R. A. Soref and B. R. Bennett, Electrooptical effects in silicon, IEEE J. Quantum Electron. "
                    "23, 123-129 (1987)", doi="10.1109/JQE.1987.1073206", wl_range_um=(1.3, 1.6),
                    notes="1.55 um fits with Drude (wl/1.55)^2 scaling; room temperature")
_NED_REF = Reference("M. Nedeljkovic, R. Soref and G. Z. Mashanovich, Free-carrier electrorefraction and "
                     "electroabsorption modulation predictions for silicon over the 1-14 um infrared wavelength "
                     "range, IEEE Photonics J. 3, 1171-1180 (2011)", doi="10.1109/JPHOT.2011.2171930",
                     wl_range_um=(1.5, 1.6), notes="1.55 um fits with (wl/1.55)^2 scaling; room temperature")


def _densities(delta_ne: Any, delta_nh: Any) -> tuple[Any, Any]:
    dne = xp.asarray(delta_ne, dtype=float)
    dnh = xp.asarray(delta_nh, dtype=float)
    return dne, dnh


def soref_bennett_1987(delta_ne: Any = 0.0, delta_nh: Any = 0.0, wl: Any = 1.55) -> tuple[Any, Any]:
    """Soref & Bennett (1987) free-carrier ``(dn, dalpha [1/cm])`` in silicon.

    Parameters
    ----------
    delta_ne, delta_nh
        Excess electron and hole densities, cm^-3 (non-negative).
    wl
        Wavelength, um (fit window 1.3-1.6 um).

    Examples
    --------
    >>> dn, da = soref_bennett_1987(1e18, 1e18)
    >>> round(float(dn), 6), round(float(da), 2)
    (-0.003015, 14.5)
    """
    dne, dnh = _densities(delta_ne, delta_nh)
    scale = (xp.asarray(wl, dtype=float) / 1.55) ** 2
    dn = -(8.8e-22 * dne + 8.5e-18 * dnh ** 0.8) * scale
    dalpha = (8.5e-18 * dne + 6.0e-18 * dnh) * scale
    return dn, dalpha


def nedeljkovic_2011(delta_ne: Any = 0.0, delta_nh: Any = 0.0, wl: Any = 1.55) -> tuple[Any, Any]:
    """Nedeljkovic, Soref & Mashanovich (2011) free-carrier ``(dn, dalpha [1/cm])`` at 1.55 um.

    Examples
    --------
    >>> dn, da = nedeljkovic_2011(1e18, 1e18)
    >>> round(float(dn), 6), round(float(da), 2)
    (-0.002708, 14.35)
    """
    dne, dnh = _densities(delta_ne, delta_nh)
    scale = (xp.asarray(wl, dtype=float) / 1.55) ** 2
    dn = -(5.4e-22 * dne ** 1.011 + 1.53e-18 * dnh ** 0.838) * scale
    dalpha = (8.88e-21 * dne ** 1.167 + 5.84e-20 * dnh ** 1.109) * scale
    return dn, dalpha


#: Registered free-carrier fits: name -> (function, reference).
FREE_CARRIER_MODELS: dict[str, tuple[Callable[..., tuple[Any, Any]], Reference]] = {
    "soref_bennett1987": (soref_bennett_1987, _SB_REF),
    "nedeljkovic2011": (nedeljkovic_2011, _NED_REF),
}


def alpha_to_k(alpha_per_cm: Any, wl: Any) -> Any:
    """Extinction coefficient ``k = alpha wl / (4 pi)`` from a power absorption coefficient in 1/cm.

    Examples
    --------
    >>> round(float(alpha_to_k(100.0, 1.55)), 7)       # 100 /cm (434 dB/cm) at 1.55 um
    0.0012335
    """
    return xp.asarray(alpha_per_cm, dtype=float) * xp.asarray(wl, dtype=float) * 1e-4 / (4.0 * xp.pi)


def doped_si(electrons_cm3: float = 0.0, holes_cm3: float = 0.0, *, model: str = "nedeljkovic2011",
             base: MaterialSpec | None = None, name: str | None = None) -> MaterialSpec:
    """Silicon with free carriers: ``N = N_Si + dn + i alpha wl / (4 pi)``.

    Parameters
    ----------
    electrons_cm3, holes_cm3
        Free electron and hole densities relative to intrinsic silicon, cm^-3
        (for a doped region, the ionised donor/acceptor density).
    model
        ``"nedeljkovic2011"`` (default; window 1.5-1.6 um) or
        ``"soref_bennett1987"`` (window 1.3-1.6 um).
    base
        Undoped silicon model; defaults to :data:`~photonix.materials.library.SI_LI1980`,
        which covers the O band.
    name
        Registry-style name; defaults to e.g. ``"Si(n=1e+18,p=0)"``.

    The enforced wavelength range is the overlap of the base model and the fit
    window; the temperature window is limited to 273-373 K because the
    free-carrier coefficients are room-temperature fits.

    Examples
    --------
    >>> from photonix.materials import doped_si, get_material
    >>> p_si = doped_si(holes_cm3=1e18)
    >>> dn = float(p_si.index(1.55).real - get_material("Si_Li1980").index(1.55).real)
    >>> round(dn, 6)                                     # -1.53e-18 * (1e18)^0.838
    -0.001856
    >>> alpha = float(4 * 3.141592653589793 * p_si.index(1.55).imag / 1.55e-4)   # power absorption, 1/cm
    >>> round(alpha, 2)                                  # 5.84e-20 * (1e18)^1.109
    5.35
    """
    if model not in FREE_CARRIER_MODELS:
        raise ValueError(f"unknown free-carrier model {model!r}; choose from {sorted(FREE_CARRIER_MODELS)}")
    if electrons_cm3 < 0 or holes_cm3 < 0:
        raise ValueError("carrier densities must be non-negative")
    fn, ref = FREE_CARRIER_MODELS[model]
    if base is None:
        from .library import SI_LI1980

        base = SI_LI1980
    if not base.isotropic:
        raise ValueError("base must be an isotropic silicon model")
    assert isinstance(base.optical, OpticalModel)
    base_model = base.optical
    ne, nh = float(electrons_cm3), float(holes_cm3)

    def evaluate(wl: Any, T: Any) -> Any:
        w = xp.asarray(wl, dtype=float)
        dn, dalpha = fn(ne, nh, w)
        return base_model.evaluate(w, T) + dn + 1j * alpha_to_k(dalpha, w)

    t_range = (273.15, 373.15)
    if base_model.temperature_range_k is not None:
        lo = max(t_range[0], base_model.temperature_range_k[0])
        hi = min(t_range[1], base_model.temperature_range_k[1])
        if lo >= hi:
            raise MaterialRangeError("base temperature window does not overlap 273-373 K")
        t_range = (lo, hi)
    optical = OpticalModel(f"{base_model.kind}+free_carrier({model})", evaluate, base_model.references + (ref,),
                           _intersect(base_model.wl_range_um, ref.wl_range_um), t_range,
                           base_model.reference_temperature_k, base_model.temperature_dependent,
                           {"electrons_cm3": ne, "holes_cm3": nh, "model": model},
                           f"doped Si ({model})")
    label = name or f"Si(n={ne:.3g},p={nh:.3g})"
    return base.replace(name=label, optical=optical, tags=tuple(base.tags) + ("doped",),
                        description=f"Si with {ne:.3g} cm^-3 electrons and {nh:.3g} cm^-3 holes ({model}).")

