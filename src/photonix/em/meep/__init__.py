"""photonix.em.meep -- Meep/MPB backend (requires Meep at import time).

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
This subpackage **requires Meep**: importing it (``from photonix.em import meep``)
raises :class:`ImportError` with an install hint when Meep is absent. The rest of
photonix does not import it eagerly, so ``import photonix.em`` still works without
Meep -- only code that actually reaches for the Meep backend pays the requirement.
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

# Hard requirement of *this* backend: fail loudly, at import, with an install hint.
require_meep()

from .fdtd import build_simulation, parity_for, waveguide_sparams  # noqa: E402
from .geometry import (  # noqa: E402
    DeviceGrid,
    build_block,
    cell_size,
    col_to_x,
    row_to_y,
)
from .materials import (  # noqa: E402
    epsilon_lookup,
    index_grid,
    material_grid_weights,
    medium,
    to_material_grid,
)
from .modes import n_eff, solve_modes  # noqa: E402

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
    "DeviceGrid",
    "cell_size",
    "col_to_x",
    "row_to_y",
    "build_block",
    # solvers
    "solve_modes",
    "n_eff",
    "waveguide_sparams",
    "build_simulation",
    "parity_for",
]
