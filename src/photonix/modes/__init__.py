"""photonix waveguide mode solver subpackage.

A scalar FDFD eigenmode solver plus dispersive material models. Results (n_eff,
n_g, field profiles) feed directly into component models, e.g.::

    from photonix.modes import n_eff
    neff_fn = lambda wl: n_eff(wl=float(wl), width=0.5, thickness=0.22)
    s = photonix.components.straight(wl=1.55, length=100.0, neff=neff_fn)
"""
from __future__ import annotations

from . import materials
from .geometry import CrossSection, rectangular_waveguide
from .materials import Material, silica, silicon, silicon_nitride
from .solver import ModeResult, group_index, n_eff, overlap, solve_modes

__all__ = [
    "materials", "Material", "silicon", "silica", "silicon_nitride",
    "CrossSection", "rectangular_waveguide",
    "ModeResult", "solve_modes", "n_eff", "group_index", "overlap",
]
