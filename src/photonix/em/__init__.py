"""photonix.em -- rigorous, differentiable electromagnetic solvers.

Delivered:
* FDE scalar mode solver with a differentiable ``n_eff`` adjoint;
* polarization-resolved 1-D slab solver (TE & TM) validated to <0.1% vs analytic;
* semivectorial (quasi-TE/TM) 2-D FDE solver with a non-symmetric adjoint;
* full-vector (Yee-grid) 2-D FDE solver: hybrid Ex/Ey modes + polarization
  fraction, validated on the SOI strip (TE0~2.45) with a differentiable adjoint;
* stretched-coordinate PML + conformal-map bend loss (complex n_eff);
* vectorial EME: TE and TM (1/eps-weighted power overlap) mode matching;
* EME<->FDFD cross-solver benchmark (TE and TM) agreeing to ~1%;
* TM (Hz) FDFD solver, independently validating the vectorial TM EME;
* EIM 2-D channel-mode estimator (exact in the slab limit, few-% on strips);
* cross-section geometry with subpixel permittivity averaging, and Sellmeier
  dispersive material models (:mod:`photonix.em.geometry`,
  :mod:`photonix.em.materials`) -- these are the canonical implementations;
  :mod:`photonix.modes` is a thin compatibility facade over them;
* an optional Meep/MPB backend (:mod:`photonix.em.meep`) to which all FDTD needs
  are delegated -- MPB cross-section modes and 2-D Meep FDTD S-parameters, both in
  native photonix types (``VectorModeData`` / ``SDict``). Its pure specifications
  are import-safe; only calls that construct Meep objects or run solvers require
  the optional dependency.

EME, FDFD, and varFDTD follow (see ``docs/DESIGN_EM_SOLVERS.md``).
"""
from __future__ import annotations

from . import (
    components,
    eim,
    eme,
    fabrication,
    fde,
    fde_vector,
    fdfd,
    geometry,
    inverse,
    materials,
    operators,
    slab,
    spectrum,
)
from .fde import ModeData, group_index, n_eff, n_eff_eps, solve_modes
from .fde_vector import (
    BendMode,
    VectorModeData,
    bend_loss_fullvector,
    fullvector_transverse_fields,
    n_eff_eps_fullvector,
    n_eff_eps_vector,
    n_eff_fullvector,
    n_eff_vector,
    power_overlap,
    solve_modes_fullvector,
    solve_modes_vector,
)
from .geometry import CrossSection, rectangular_waveguide
from .materials import Material, silica, silicon, silicon_nitride
from .slab import slab_neff, slab_neff_analytic
from .spectrum import sweep

__all__ = [
    "fde",
    "fde_vector",
    "operators",
    "geometry",
    "slab",
    "eim",
    "eme",
    "fdfd",
    "meep",
    "fabrication",
    "components",
    "materials",
    "spectrum",
    "inverse",
    "sweep",
    "CrossSection",
    "rectangular_waveguide",
    "Material",
    "silicon",
    "silica",
    "silicon_nitride",
    "ModeData",
    "solve_modes",
    "n_eff",
    "n_eff_eps",
    "group_index",
    "slab_neff",
    "slab_neff_analytic",
    "VectorModeData",
    "solve_modes_vector",
    "n_eff_vector",
    "n_eff_eps_vector",
    "solve_modes_fullvector",
    "n_eff_fullvector",
    "n_eff_eps_fullvector",
    "bend_loss_fullvector",
    "BendMode",
    "fullvector_transverse_fields",
    "power_overlap",
]


def __getattr__(name: str):
    """Lazily expose the optional Meep backend on attribute access.

    Keeping ``meep`` out of the eager imports above keeps the main EM import light.
    The subpackage itself is import-safe; solver calls raise a helpful ImportError
    when Meep is not installed. We use
    ``importlib.import_module`` rather than ``from . import meep`` so that a failed
    Meep import propagates ImportError directly instead of recursing back into this
    ``__getattr__`` through the import machinery's ``hasattr`` probe.
    """
    if name == "meep":
        import importlib

        return importlib.import_module(f"{__name__}.meep")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
