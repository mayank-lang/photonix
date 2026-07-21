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
>>> T = px.power(S[("in0", "out0")])               # transmission spectrum

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
:mod:`photonix.pdk`, :mod:`photonix.viz`, :mod:`photonix.optim`) can also be
imported directly.
"""
from __future__ import annotations

__version__ = "0.1.0"

# --- Core (always available) ------------------------------------------------ #
from . import core  # noqa: E402
from .core import (  # noqa: E402
    HAS_JAX,
    as_sdense,
    as_sdict,
    backend_name,
    grad,
    insertion_loss_db,
    jit,
    power,
    to_numpy,
    use_x64,
    value_and_grad,
    vmap,
    xp,
)
from .core.constants import WL_C_BAND, WL_DEFAULT  # noqa: E402

# Enable 64-bit precision by default for numerical accuracy (photonics needs it).
use_x64(True)

# Convenience re-exports of common array constructors on the active backend.
linspace = xp.linspace
asarray = xp.asarray
array = xp.asarray

__all__ = [
    "__version__",
    "core",
    "HAS_JAX", "xp", "jit", "grad", "value_and_grad", "vmap", "use_x64",
    "backend_name", "power", "insertion_loss_db", "as_sdict", "as_sdense", "to_numpy",
    "linspace", "asarray", "array", "WL_C_BAND", "WL_DEFAULT",
]

# --- Optional subpackages (filled in by feature modules) -------------------- #
# Each is imported defensively so that a partially-installed / in-development
# tree still imports cleanly. As modules land, their names become available.
for _name in ("components", "circuit", "modes", "layout", "pdk", "viz", "optim", "em"):
    try:  # pragma: no cover - availability depends on what is installed
        _mod = __import__(f"photonix.{_name}", fromlist=[_name])
        globals()[_name] = _mod
        __all__.append(_name)
    except Exception:  # noqa: BLE001
        pass
del _name
