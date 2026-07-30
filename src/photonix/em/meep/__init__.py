"""photonix.em.meep -- optional Meep/MPB backend with import-safe specifications.

photonix is a differentiable, frequency-domain PIC design library; it deliberately
does **not** ship its own time-domain Maxwell (FDTD) solver. Instead, any workflow
that needs FDTD -- broadband S-parameters of a non-adiabatic device, radiation from
an abrupt junction, a full 3-D scattering check -- is delegated to `MIT Meep
<https://meep.readthedocs.io>`_ and its bundled mode solver MPB through this
extension. The bridge speaks photonix's native types in both directions:

* :func:`solve_modes` / :func:`n_eff` -- cross-section modes via MPB, returned as a
  :class:`photonix.em.fde_vector.VectorModeData` (a drop-in cross-check for the
  in-house full-vector FDE solver);
* :func:`waveguide_sparams` -- 2-D FDTD S-parameters, returned as a
  :data:`photonix.core.types.SDict` with the same ``o1``/``o2`` ports and column
  indexing as :func:`photonix.em.fdfd.waveguide_sparams`;
* :func:`to_material_grid` / :func:`material_grid_weights` -- the permittivity-grid
  translation underneath both.

Import contract
---------------
This subpackage is safe to import without Meep so pure grids, layout specifications,
and availability diagnostics remain useful everywhere. Functions that construct
Meep objects or run MPB/FDTD raise :class:`ImportError` with an install hint at the
point of use.
"""
from __future__ import annotations

from ._guard import (
    HAS_MEEP,
    HAS_MPB,
    k_from_n_eff,
    meep_frequency,
    n_eff_from_k,
    require_meep,
    require_mpb,
)
from .fdtd import build_simulation, parity_for, waveguide_dataset, waveguide_sparams, waveguide_spectrum
from .geometry import (
    DeviceGrid,
    build_block,
    build_pixel_block,
    cell_size,
    col_to_x,
    row_to_y,
)
from .layout import (
    LayerSpec,
    MeepLayout,
    MeepPortRegion,
    PreparedLayout,
    PreparedPolygon,
    PreparedPort,
    build_layout_geometry,
    build_layout_simulation,
    port_region,
    port_regions,
    prepare_layout,
)
from .materials import (
    epsilon_lookup,
    index_grid,
    material_grid_weights,
    medium,
    to_material_grid,
    to_medium,
)
from .modes import n_eff, solve_modes
from .multiport import (
    ModalTerminal,
    MultiportPlan,
    PortModeSpec,
    modal_terminal,
    plan_multiport,
    simulate_multiport_sparameters,
)

__all__ = [
    # availability / units
    "HAS_MEEP",
    "HAS_MPB",
    "require_meep",
    "require_mpb",
    "meep_frequency",
    "n_eff_from_k",
    "k_from_n_eff",
    # translation
    "material_grid_weights",
    "index_grid",
    "epsilon_lookup",
    "to_material_grid",
    "medium",
    "to_medium",
    "DeviceGrid",
    "cell_size",
    "col_to_x",
    "row_to_y",
    "build_block",
    "build_pixel_block",
    # solvers
    "solve_modes",
    "n_eff",
    "waveguide_sparams",
    "waveguide_spectrum",
    "waveguide_dataset",
    "build_simulation",
    "parity_for",
    # native layout bridge
    "LayerSpec",
    "PreparedPolygon",
    "PreparedPort",
    "PreparedLayout",
    "MeepLayout",
    "MeepPortRegion",
    "prepare_layout",
    "build_layout_geometry",
    "build_layout_simulation",
    "port_region",
    "port_regions",
    # generalized modal S matrices
    "PortModeSpec",
    "ModalTerminal",
    "MultiportPlan",
    "modal_terminal",
    "plan_multiport",
    "simulate_multiport_sparameters",
]
