"""Filter synthesis: from a target response to component parameters.

* :mod:`~photonix.synthesis.prototypes` -- Butterworth/Chebyshev low-pass
  prototypes and the periodic all-pole targets of resonator filters.
* :mod:`~photonix.synthesis.crow` -- exact coupled-resonator (CROW) synthesis.
* :mod:`~photonix.synthesis.lattice` -- exact FIR lattice (cascaded-MZI)
  synthesis by polynomial peeling, maximally flat and equiripple targets, and
  minimum-phase spectral factorization.
* :mod:`~photonix.synthesis.gaps` -- designed couplings to physical gaps.
"""
from __future__ import annotations

from .crow import CrowDesign, crow_closed_form, crow_drop_power, synthesize_crow
from .gaps import coupling_to_gap
from .lattice import (
    CrossBar,
    LatticeDesign,
    complementary_bar,
    equiripple_lowpass,
    interleaver,
    lattice_coefficients,
    lattice_response,
    maxflat_halfband,
    spectral_factor,
    synthesize_lattice,
)
from .prototypes import chebyshev_epsilon, lowpass_prototype, periodic_target

__all__ = [
    "lowpass_prototype", "chebyshev_epsilon", "periodic_target",
    "CrowDesign", "crow_closed_form", "crow_drop_power", "synthesize_crow",
    "coupling_to_gap",
    "CrossBar", "LatticeDesign", "lattice_response", "lattice_coefficients", "spectral_factor",
    "complementary_bar", "maxflat_halfband", "equiripple_lowpass", "synthesize_lattice", "interleaver",
]
