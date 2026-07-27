"""photonix component model library.

Physics-based, differentiable, parametric models. Every model is a pure function
``f(*, wl=1.55, **params) -> SDict`` returning a scattering dictionary, so it is
``jit``-able and ``grad``-able and composes directly in :mod:`photonix.circuit`.
"""
from __future__ import annotations

from .couplers import coupler, directional_coupler, mmi1x2, mmi2x2
from .gratings import attenuator, grating_coupler, phase_shifter, terminator
from .mzi import mzi
from .resonators import add_drop_ring, all_pass_ring, ring_coupler
from .waveguide import bend, bend_from_solver, neff_linear, straight

#: Registry of model name -> callable, used by PDKs and the circuit solver.
MODELS = {
    "straight": straight,
    "bend": bend,
    "directional_coupler": directional_coupler,
    "coupler": coupler,
    "mmi1x2": mmi1x2,
    "mmi2x2": mmi2x2,
    "grating_coupler": grating_coupler,
    "phase_shifter": phase_shifter,
    "attenuator": attenuator,
    "terminator": terminator,
    "ring_coupler": ring_coupler,
    "all_pass_ring": all_pass_ring,
    "add_drop_ring": add_drop_ring,
    "mzi": mzi,
}

__all__ = [
    "straight", "bend", "bend_from_solver", "neff_linear",
    "directional_coupler", "coupler", "mmi1x2", "mmi2x2",
    "grating_coupler", "phase_shifter", "attenuator", "terminator",
    "ring_coupler", "all_pass_ring", "add_drop_ring",
    "mzi", "MODELS",
]
