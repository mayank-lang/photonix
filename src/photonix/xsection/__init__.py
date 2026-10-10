"""Polygonal multi-material cross-sections shared by optical, thermal and electrical solvers.

One :class:`XSection` -- painted polygonal :class:`Region` objects plus a
background -- is rasterized *exactly* onto a :class:`TensorGrid` and evaluated
into whatever coefficients a solver needs: permittivity (arithmetic or Kottke
tensor smoothing) for the FDE mode solvers, conductivities and capacities for
heat or conduction solvers, tag masks for heaters, contacts and doping.  Using
the same object for all of them keeps the optical and thermal models from
silently diverging.

* :mod:`~photonix.xsection.shapes` -- regions: polygons, rectangles, bands,
  trapezoids (sidewall angles), ribs, disks.
* :mod:`~photonix.xsection.raster` -- grids, the exact sweep rasterizer,
  :class:`Coverage`, :class:`XSection`.
* :mod:`~photonix.xsection.properties` -- duck-typed material/property look-up.
* :mod:`~photonix.xsection.library` -- strip, rib, coupler, heater-over-waveguide.
* :mod:`~photonix.xsection.from_stack` -- adapter from :class:`photonix.pdk.LayerStack`.
* :mod:`~photonix.xsection.modes` -- full-vector modes with staggered tensor
  smoothing (second-order at interfaces) and a differentiable ``n_eff``.
"""
from __future__ import annotations

from .from_stack import from_layer_stack
from .library import coupler, heater_over_waveguide, rib_waveguide, strip
from .modes import XSectionModes, n_eff, n_eff_eps_tensor, solve_modes, yee_permittivity
from .properties import resolve_index, resolve_property
from .raster import Coverage, TensorGrid, XSection, graded_grid, uniform_grid
from .shapes import BAND_HALF_WIDTH, MaterialSpec, Region, band, disk, polygon, polygon_area, rect, rib, trapezoid

__all__ = [
    "MaterialSpec",
    "Region",
    "polygon",
    "rect",
    "band",
    "trapezoid",
    "rib",
    "disk",
    "polygon_area",
    "BAND_HALF_WIDTH",
    "resolve_index",
    "resolve_property",
    "TensorGrid",
    "uniform_grid",
    "graded_grid",
    "Coverage",
    "XSection",
    "strip",
    "rib_waveguide",
    "coupler",
    "heater_over_waveguide",
    "from_layer_stack",
    "XSectionModes",
    "yee_permittivity",
    "solve_modes",
    "n_eff",
    "n_eff_eps_tensor",
]
