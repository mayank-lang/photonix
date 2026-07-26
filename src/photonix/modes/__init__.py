"""Waveguide mode solving -- a compatibility facade over :mod:`photonix.em`.

.. deprecated:: 0.2
   New code should import from :mod:`photonix.em` directly. This subpackage is
   kept so that existing ``photonix.modes`` code keeps working, and it now
   *forwards* to :mod:`photonix.em.fde` rather than carrying a second,
   independent solver that returned different numbers under the same names (see
   :mod:`photonix.modes.solver` for the details).

What lives where
----------------
``solve_modes`` / ``n_eff`` / ``group_index`` / ``CrossSection`` /
``rectangular_waveguide``
    Re-exported from :mod:`photonix.em`. Identical objects, not copies.
``overlap``
    Scalar field overlap helper, kept here.
``materials``, ``Material``, ``silicon``, ``silica``, ``silicon_nitride``
    Dispersive index models. These have no :mod:`photonix.em` counterpart and
    are re-exported *from* :mod:`photonix.em` as well, so ``em`` is
    self-sufficient.

Typical use::

    from photonix.modes import n_eff
    neff_fn = lambda wl: n_eff(wl=float(wl), width=0.5, thickness=0.22)
    s = photonix.components.straight(wl=1.55, length=100.0, neff=neff_fn)

Examples
--------
>>> import photonix.modes as modes, photonix.em as em
>>> modes.n_eff is em.n_eff          # one implementation, two import paths
True
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
