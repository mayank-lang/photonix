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

__all__ = [
    "Cell", "Port", "Reference", "OPTICAL", "ELECTRICAL", "GEOMETRY_ONLY", "components",
    "route", "route_manhattan", "RoutingError",
    "gdstk_available", "write_gds", "read_gds", "write_oas", "read_oas",
    "extract_netlist",
    "DrcReport", "DrcViolation", "check_min_width", "check_min_space", "check_min_area",
    "run_drc_precheck",
    "KLayoutResult", "KLayoutRunError", "find_klayout",
    "klayout_available", "run_klayout_deck", "run_drc", "run_lvs",
]
