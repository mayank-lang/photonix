"""Dispersive refractive-index models (compatibility re-export).

.. deprecated:: 0.2
   The canonical module is now :mod:`photonix.em.materials`, so that
   :mod:`photonix.em` is self-sufficient and the dependency runs
   ``modes -> em`` in one direction only. This module re-exports it unchanged.

Examples
--------
>>> from photonix.modes.materials import silicon
>>> bool(abs(float(silicon(1.55)) - 3.4757) < 3e-3)
True
"""
from __future__ import annotations

from photonix.em.materials import (
    SI,
    SIN,
    SIO2,
    Material,
    constant,
    silica,
    silicon,
    silicon_nitride,
)

__all__ = [
    "silica", "silicon", "silicon_nitride", "Material", "constant",
    "SI", "SIO2", "SIN",
]
