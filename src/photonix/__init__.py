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

# --- Subpackages ------------------------------------------------------------ #
# `viz` and `layout` genuinely depend on optional third-party packages
# (matplotlib, gdstk), so a missing *dependency* degrades gracefully. Everything
# else is pure photonix + numpy/scipy and must import, or something is broken.
#
# This used to be a blanket `except Exception: pass` over every subpackage, which
# meant a typo or a real bug anywhere in the tree made the subpackage silently
# vanish from `photonix.*` -- surfacing much later as a confusing AttributeError
# instead of the actual traceback. Failures are now scoped and never silent.
from . import circuit, components, em, modes, optim, pdk  # noqa: E402

__all__ += ["components", "circuit", "modes", "em", "pdk", "optim"]

#: Subpackages that could not be imported, mapped to the reason why.
#: Empty on a complete install; inspect it if an optional feature is missing.
UNAVAILABLE: dict[str, str] = {}

for _name, _extra in (("layout", "layout"), ("viz", "viz")):
    try:
        globals()[_name] = __import__(f"photonix.{_name}", fromlist=[_name])
        __all__.append(_name)
    except ImportError as _exc:  # pragma: no cover - depends on what is installed
        # Only a *missing optional dependency* is tolerated. Record it rather
        # than discarding it, and keep the original message.
        UNAVAILABLE[_name] = (
            f"{_exc}. Install the optional dependency with: pip install 'photonix[{_extra}]'"
        )
del _name, _extra

__all__.append("UNAVAILABLE")


def __dir__() -> list[str]:
    """Restrict tab-completion / ``dir()`` to the documented public API.

    Without this, ``annotations`` (bound as a side effect of
    ``from __future__ import annotations``) and the private module machinery
    show up alongside the real API.
    """
    return sorted(__all__)
