"""photonix — differentiable, GPU-accelerated photonic integrated circuit design.

photonix unifies the photonic design stack that is currently spread across many
separate tools (layout, mode solving, circuit simulation, optimization) into a
single, rigorously validated, **differentiable** library built on JAX.

Quick start
-----------
>>> import photonix as px
>>> wl = px.linspace(1.5, 1.6, 201)              # wavelength sweep [um]
>>> mzi = px.circuit.mzi(delta_length=20.0)       # build an MZI circuit
>>> S = mzi(wl=wl)                                 # differentiable S-parameters
>>> T = px.power(S[("o1", "o4")])                  # bar transmission spectrum

Design principles
-----------------
* **Differentiable-first** — gradients flow through every model and the circuit
  solver, enabling adjoint/inverse design out of the box.
* **Accuracy-first** — models are validated against analytic limits and the
  numerical core defaults to 64-bit precision.
* **Composable** — components are pure functions returning scattering
  dictionaries; circuits are built by connecting named ports.

The public API is intentionally small and re-exports the most-used names from
each subpackage. Subpackages (:mod:`photonix.core`, :mod:`photonix.circuit`,
:mod:`photonix.components`, :mod:`photonix.modes`, :mod:`photonix.layout`,
:mod:`photonix.pdk`, :mod:`photonix.viz`, :mod:`photonix.optim`,
:mod:`photonix.multiphysics`, :mod:`photonix.interop`,
:mod:`photonix.diagnostics`) can also be imported directly.
"""
from __future__ import annotations

__version__ = "0.1.0"

# --- Core (always available) ------------------------------------------------ #
from . import core  # noqa: E402
from .core import (  # noqa: E402
    HAS_JAX,
    SParameterDataset,
    SParameterDiagnostics,
    analyze_sparameters,
    as_sdense,
    as_sdict,
    backend_name,
    differentiate_samples,
    grad,
    group_delay,
    group_delay_dispersion,
    insertion_loss_db,
    jit,
    power,
    project_passive,
    to_numpy,
    touchstone_capabilities,
    use_x64,
    value_and_grad,
    vmap,
    xp,
)
from .core.constants import WL_C_BAND, WL_DEFAULT  # noqa: E402

# Enable 64-bit precision by default for numerical accuracy (photonics needs it).
use_x64(True)

# Reproducibility helpers inspect the now-configured numerical backend.
from .diagnostics import runtime_info, show_config  # noqa: E402

# Convenience re-exports of common array constructors on the active backend.
linspace = xp.linspace
asarray = xp.asarray
array = xp.asarray

__all__ = [
    "__version__",
    "core",
    "HAS_JAX", "xp", "jit", "grad", "value_and_grad", "vmap", "use_x64",
    "backend_name", "power", "insertion_loss_db", "as_sdict", "as_sdense", "to_numpy",
    "analyze_sparameters", "project_passive", "SParameterDiagnostics",
    "differentiate_samples", "group_delay", "group_delay_dispersion",
    "linspace", "asarray", "array", "WL_C_BAND", "WL_DEFAULT", "SParameterDataset",
    "touchstone_capabilities",
    "runtime_info", "show_config",
]

# --- Subpackages ------------------------------------------------------------ #
# Optional third-party dependencies are imported at the feature boundary:
# matplotlib inside plotting calls, gdstk inside GDSII/OASIS calls, and KLayout
# only through an external batch invocation. The
# subpackages themselves therefore import on a minimal NumPy/SciPy installation.
# Import every namespace eagerly so an internal ImportError is never mistaken
# for a missing optional dependency and silently hidden.
from . import circuit, components, em, interop, layout, modes, multiphysics, optim, pdk, viz  # noqa: E402

__all__ += [
    "components", "circuit", "modes", "em", "interop", "layout", "pdk", "viz", "optim", "multiphysics",
]

# Kept for compatibility with 0.1.x callers. Optional *features* now raise their
# dependency error when called, so no complete subpackage can be unavailable.
UNAVAILABLE: dict[str, str] = {}
__all__.append("UNAVAILABLE")


def __dir__() -> list[str]:
    """Restrict tab-completion / ``dir()`` to the documented public API.

    Without this, ``annotations`` (bound as a side effect of
    ``from __future__ import annotations``) and the private module machinery
    show up alongside the real API.
    """
    return sorted(__all__)
