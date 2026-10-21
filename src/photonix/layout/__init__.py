"""Layout cells, routing, lossless GDSII/OASIS I/O, text and labels, extraction, DRC pre-checks and KLayout runs."""
from __future__ import annotations

from . import components, library
from .cell import ELECTRICAL, GEOMETRY_ONLY, OPTICAL, Cell, Port, Reference
from .drc import DrcReport, DrcViolation, check_min_area, check_min_space, check_min_width, run_drc_precheck
from .extract import extract_netlist
from .extract_hierarchy import extract_hierarchical_netlist
from .gds import gdstk_available, read_gds, read_oas, write_gds, write_oas
from .gds_io import (
    LayoutArtifact,
    StaleSidecarWarning,
    infer_ports,
    layout_units,
    list_top_cells,
    read_layout,
    sidecar_path_for,
    snap_to_grid,
    write_layout,
)
from .klayout import (
    KLayoutResult,
    KLayoutRunError,
    find_klayout,
    klayout_available,
    run_drc,
    run_klayout_deck,
    run_lvs,
)
from .labels import Label, LabeledCell, text
from .pcell import canonical_settings, pcell, pcell_info, settings_hash
from .placement import DieCell, FloorplanIssue, align, check_floorplan, distribute, grid, pack, reference_bbox
from .routing import RoutingError, route, route_manhattan
from .routing_astar import route_astar
from .routing_auto import default_bend_penalty, route_auto
from .routing_bundle import fanout, route_bundle
from .routing_length import route_length_matched
from .routing_sbend import route_sbend, sbend_arc_length, sbend_min_radius, sbend_min_span
from .schematic import ROUTING_STRATEGIES, Placement, RouteSpec, SchematicSpec, build_layout, schematic_from_layout

__all__ = [
    "Cell", "Port", "Reference", "OPTICAL", "ELECTRICAL", "GEOMETRY_ONLY", "components",
    "route", "route_manhattan", "RoutingError",
    "route_auto", "route_sbend", "route_bundle", "fanout", "route_astar", "route_length_matched",
    "sbend_arc_length", "sbend_min_radius", "sbend_min_span", "default_bend_penalty",
    "gdstk_available", "write_gds", "read_gds", "write_oas", "read_oas",
    "read_layout", "write_layout", "LayoutArtifact", "list_top_cells", "layout_units", "infer_ports",
    "snap_to_grid", "sidecar_path_for", "StaleSidecarWarning",
    "Label", "LabeledCell", "text",
    "grid", "pack", "align", "distribute", "reference_bbox", "DieCell", "FloorplanIssue", "check_floorplan",
    "library",
    "extract_netlist", "extract_hierarchical_netlist",
    "pcell", "pcell_info", "canonical_settings", "settings_hash",
    "Placement", "RouteSpec", "SchematicSpec", "build_layout", "schematic_from_layout", "ROUTING_STRATEGIES",
    "DrcReport", "DrcViolation", "check_min_width", "check_min_space", "check_min_area",
    "run_drc_precheck",
    "KLayoutResult", "KLayoutRunError", "find_klayout",
    "klayout_available", "run_klayout_deck", "run_drc", "run_lvs",
]
