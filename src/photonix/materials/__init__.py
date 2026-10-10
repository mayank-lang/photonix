"""photonix.materials -- cited, range-checked, differentiable material models.

A :class:`MaterialSpec` bundles a complex, possibly temperature-dependent and
anisotropic optical model with thermal, electrical and electro-optic data,
each with its literature source. Look materials up by name::

    >>> from photonix.materials import get_material
    >>> si = get_material("Si")
    >>> 3.47 < float(si.index(1.55).real) < 3.48
    True
    >>> round(float(si.dn_dt(1.55, 300.0)), 6)            # Li (1980) thermo-optic coefficient
    0.000184

The material protocol shared with other photonix packages is
``index(wl, temperature=None) -> n + i k`` (``k >= 0`` is loss),
``epsilon(wl, temperature=None)``, the optional attributes
``thermal_conductivity`` [W/(m K)], ``heat_capacity`` [J/(kg K)],
``density`` [kg/m^3], ``relative_permittivity_static``, and ``dn_dt(wl, T)``;
see :mod:`photonix.materials.model` for the sign convention and the range
checks. :func:`photonix.em.silicon`, :func:`~photonix.em.silica` and
:func:`~photonix.em.silicon_nitride` stay the canonical Sellmeier
implementations; the ``Si``/``SiO2``/``Si3N4`` entries wrap them.

Submodules: :mod:`~photonix.materials.model` (data model),
:mod:`~photonix.materials.library` (curated entries),
:mod:`~photonix.materials.formulas` (dispersion formulas),
:mod:`~photonix.materials.io` (refractiveindex.info YAML / CSV readers),
:mod:`~photonix.materials.free_carrier` (doped silicon),
:mod:`~photonix.materials.fit` (passive Lorentz-Drude fits, Meep export),
:mod:`~photonix.materials.kk` (Kramers-Kronig consistency),
:mod:`~photonix.materials.mixing` (Lorentz-Lorenz mixtures such as SiON).
"""
from __future__ import annotations

from . import fit, formulas, free_carrier, io, kk, library, mixing, model, registry
from .fit import DrudeTerm, LorentzFit, LorentzPole, fit_lorentz
from .formulas import li1980_germanium, li1980_silicon, lorentz_drude_epsilon, refractiveindex_formula
from .free_carrier import doped_si, nedeljkovic_2011, soref_bennett_1987
from .io import load_nk_csv, load_refractiveindex_yaml, parse_refractiveindex_yaml
from .kk import kramers_kronig_n
from .library import (
    AG,
    AIR,
    AL,
    AL2O3,
    AU,
    CR,
    CU,
    GAAS,
    GE,
    GE_LI1980,
    INP,
    LINBO3,
    NI,
    PT,
    SI,
    SI3N4,
    SI_GREEN2008,
    SI_LI1980,
    SIO2,
    SU8,
    TI,
    TIN,
    TIO2,
    VACUUM,
    W,
)
from .mixing import lorentz_lorenz_mix, sion
from .model import (
    ROOM_TEMPERATURE_K,
    MaterialLike,
    MaterialRangeError,
    MaterialRangeWarning,
    MaterialSpec,
    MissingPropertyError,
    OpticalModel,
    Reference,
    Uniaxial,
    as_material,
    callable_model,
    constant_model,
    tabulated_model,
    to_engineering_sign,
    with_linear_thermo_optic,
    with_temperature_shift,
)
from .registry import get_material, list_materials, register, unregister

__all__ = [
    # submodules
    "model", "library", "formulas", "io", "free_carrier", "fit", "kk", "mixing", "registry",
    # data model
    "Reference", "OpticalModel", "Uniaxial", "MaterialSpec", "MaterialLike",
    "MaterialRangeWarning", "MaterialRangeError", "MissingPropertyError", "ROOM_TEMPERATURE_K",
    "to_engineering_sign", "as_material", "constant_model", "callable_model", "tabulated_model",
    "with_linear_thermo_optic", "with_temperature_shift",
    # registry
    "register", "unregister", "get_material", "list_materials",
    # library entries
    "SI", "SI_LI1980", "SI_GREEN2008", "SIO2", "SI3N4", "AIR", "VACUUM", "GE", "GE_LI1980", "INP", "GAAS",
    "LINBO3", "AL2O3", "TIO2", "SU8", "AL", "CU", "AU", "AG", "W", "TI", "CR", "PT", "NI", "TIN",
    # formulas
    "refractiveindex_formula", "lorentz_drude_epsilon", "li1980_silicon", "li1980_germanium",
    # io
    "load_refractiveindex_yaml", "load_nk_csv", "parse_refractiveindex_yaml",
    # free carriers
    "doped_si", "soref_bennett_1987", "nedeljkovic_2011",
    # mixing
    "lorentz_lorenz_mix", "sion",
    # fitting / causality
    "fit_lorentz", "LorentzFit", "LorentzPole", "DrudeTerm", "kramers_kronig_n",
]
