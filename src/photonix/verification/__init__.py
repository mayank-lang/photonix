"""Layout verification: LVS-lite connectivity, a DRC rule deck, density analysis.

Photonix's own geometric verification, for iterating on a design before the
foundry's qualified decks (which :func:`photonix.layout.run_drc` and
:func:`photonix.layout.run_lvs` hand to KLayout). Nothing here is sign-off.

* :mod:`~photonix.verification.connectivity` -- ports on geometry, opens,
  near-opens and optical shorts (:class:`LvsIssue`).
* :mod:`~photonix.verification.extract_params` -- lengths, gaps and radii
  re-extracted from polygons and compared with ``model_settings``.
* :mod:`~photonix.verification.lvs` -- layout-versus-schematic netlist
  comparison and the combined :func:`run_lvs_lite`.
* :mod:`~photonix.verification.drc_rules` -- a :class:`RuleDeck` of
  enclosure, separation, extension, angle, grid, max-width, hole-area and
  density rules with waivers, JSON, and a generated KLayout DRC script.
* :mod:`~photonix.verification.rdb` -- KLayout ``.lyrdb`` marker databases.
* :mod:`~photonix.verification.density` -- exact sliding-window density and
  the evanescent keep-out distance for dummy fill
  (:func:`photonix.layout.fill` places the fill).

All functions need the optional ``gdstk`` extra (``pip install photonix[layout]``).
"""
from __future__ import annotations

from . import connectivity, density, drc_rules, extract_params, lvs, rdb
from .connectivity import ISSUE_KINDS, LvsIssue, check_optical_connectivity, check_ports_on_geometry
from .density import DensityMap, check_density, density_map, evanescent_keepout
from .drc_rules import (
    METRICS,
    RULE_KINDS,
    DeckReport,
    Rule,
    RuleDeck,
    Waiver,
    apply_waivers,
    check_enclosure,
    check_extension,
    check_hole_area,
    check_inside,
    check_max_width,
    check_min_angle,
    check_not_overlap,
    check_offgrid_vertices,
    check_separation,
    check_space,
    compare_with_klayout,
)
from .extract_params import check_parameters, extract_parameters
from .lvs import LvsReport, compare_netlists, run_lvs_lite
from .rdb import read_lyrdb, write_lyrdb

__all__ = [
    "connectivity", "extract_params", "lvs",
    "LvsIssue", "ISSUE_KINDS", "LvsReport",
    "check_ports_on_geometry", "check_optical_connectivity",
    "extract_parameters", "check_parameters",
    "compare_netlists", "run_lvs_lite",
    "drc_rules", "rdb", "density",
    "RULE_KINDS", "METRICS", "Rule", "RuleDeck", "DeckReport", "Waiver", "apply_waivers",
    "check_space", "check_enclosure", "check_separation", "check_extension", "check_not_overlap",
    "check_inside", "check_min_angle", "check_offgrid_vertices", "check_max_width", "check_hole_area",
    "compare_with_klayout", "write_lyrdb", "read_lyrdb",
    "DensityMap", "density_map", "check_density", "evanescent_keepout",
]
