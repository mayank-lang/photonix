"""Layout cells, routing, GDSII/OASIS I/O, extraction, DRC pre-checks, and optional KLayout runs."""
from __future__ import annotations

from . import components
from .cell import ELECTRICAL, GEOMETRY_ONLY, OPTICAL, Cell, Port, Reference
from .drc import DrcReport, DrcViolation, check_min_area, check_min_space, check_min_width, run_drc_precheck
from .extract import extract_netlist
from .gds import gdstk_available, read_gds, read_oas, write_gds, write_oas
from .klayout import (
    KLayoutResult,
    KLayoutRunError,
    find_klayout,
    klayout_available,
    run_drc,
    run_klayout_deck,
    run_lvs,
)
from .routing import RoutingError, route, route_manhattan
from .routing_astar import route_astar
from .routing_auto import default_bend_penalty, route_auto
from .routing_bundle import fanout, route_bundle
from .routing_length import route_length_matched
from .routing_sbend import route_sbend, sbend_arc_length, sbend_min_radius, sbend_min_span

__all__ = [
    "Cell", "Port", "Reference", "OPTICAL", "ELECTRICAL", "GEOMETRY_ONLY", "components",
    "route", "route_manhattan", "RoutingError",
    "route_auto", "route_sbend", "route_bundle", "fanout", "route_astar", "route_length_matched",
    "sbend_arc_length", "sbend_min_radius", "sbend_min_span", "default_bend_penalty",
    "gdstk_available", "write_gds", "read_gds", "write_oas", "read_oas",
    "extract_netlist",
    "DrcReport", "DrcViolation", "check_min_width", "check_min_space", "check_min_area",
    "run_drc_precheck",
    "KLayoutResult", "KLayoutRunError", "find_klayout",
    "klayout_available", "run_klayout_deck", "run_drc", "run_lvs",
]
