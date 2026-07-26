"""Waveguide cross-section geometry (compatibility re-export).

.. deprecated:: 0.2
   The canonical implementation now lives in :mod:`photonix.em.geometry`. This
   module re-exports it unchanged so ``from photonix.modes import CrossSection,
   rectangular_waveguide`` keeps working.

Historically this module built the permittivity grid by *staircasing* the core
boundary onto the grid. :mod:`photonix.em.geometry` instead uses **subpixel
permittivity averaging**, which restores the O(h^2) convergence the solver
assumes; the staircased version biased ``n_eff`` high by ~1% on an SOI strip.
Keeping two grid builders meant :func:`photonix.modes.n_eff` and
:func:`photonix.em.n_eff` returned different numbers under the same name, so the
old one was retired rather than kept as a silent second answer.

Examples
--------
>>> from photonix.modes import rectangular_waveguide
>>> cs = rectangular_waveguide(width=0.5, thickness=0.22, resolution=30)
>>> bool(cs.eps.max() > cs.eps.min())
True
"""
from __future__ import annotations

from photonix.em.geometry import CrossSection, rectangular_waveguide

__all__ = ["CrossSection", "rectangular_waveguide"]
