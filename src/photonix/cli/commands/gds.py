"""``photonix gds inspect|precheck``: summarize a GDSII file and run the geometric DRC pre-check."""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .._common import CliError, ExitCode, add_json_flag, is_version, require, schema_id, sort_ports

__all__ = ["register", "run_inspect", "render_inspect", "run_precheck", "render_precheck", "load_rules",
           "read_cell"]

INSPECT_SCHEMA = schema_id("gds-inspect")
PRECHECK_SCHEMA = schema_id("gds-precheck")
RULES_FORMAT = "photonix.drc_rules"


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("gds", help="inspect a GDSII file or run the DRC pre-check (needs gdstk)",
                       description="GDSII tools. Both actions need the optional gdstk package "
                                   "(pip install 'photonix[layout]') and exit 3 without it.")
    actions = p.add_subparsers(dest="action", metavar="ACTION", required=True)

    i = actions.add_parser("inspect", help="cells, layers, polygon counts and bounding boxes",
                           description="Summarize every cell of a GDSII library and the flattened layers of "
                                       "its top cell. Coordinates are reported in um.")
    i.add_argument("file", help="GDSII file")
    i.add_argument("--cell", default=None, help="cell whose flattened layers to report (default: the top cell)")
    add_json_flag(i)
    i.set_defaults(handler=run_inspect, render=render_inspect)

    c = actions.add_parser(
        "precheck", help="minimum width/space/area pre-check; exit 1 on violations",
        description="Flatten one cell and run photonix.layout.run_drc_precheck with a PDK's design rules and/or "
                    "a rules JSON file. A fast pre-check, not foundry sign-off (see docs/CLI.md).")
    c.add_argument("file", help="GDSII file")
    c.add_argument("--pdk", default=None, help="use a PDK's design rules (e.g. demo)")
    c.add_argument("--rules", default=None,
                   help='rules JSON: {"1/0": {"min_width": 0.15, "min_space": 0.15, "min_area": 0.05}} '
                        "(keys LAYER/DATATYPE or LAYER); overrides --pdk rules per layer and rule")
    c.add_argument("--cell", default=None, help="cell to check (default: the top cell)")
    c.add_argument("--max-violations", type=int, default=100,
                   help="violations listed in full (the counts are always complete; default 100)")
    add_json_flag(c)
    c.set_defaults(handler=run_precheck, render=render_precheck)


# --------------------------------------------------------------------------- #
# GDS reading
# --------------------------------------------------------------------------- #
def _library(path: str):
    gdstk = require("gdstk", "reading GDSII")
    source = Path(path)
    if not source.is_file():
        raise CliError(f"{source}: no such file")
    try:
        lib = gdstk.read_gds(str(source))
    except (OSError, RuntimeError, ValueError) as exc:
        raise CliError(f"{source}: not a readable GDSII file ({exc})") from exc
    if not lib.cells:
        raise CliError(f"{source}: the library contains no cells")
    return lib


def _select(lib, name: str | None):
    if name is not None:
        for cell in lib.cells:
            if cell.name == name:
                return cell
        raise CliError(f"no cell named {name!r}; cells: {', '.join(sorted(c.name for c in lib.cells))}")
    tops = lib.top_level()
    if len(tops) == 1:
        return tops[0]
    names = ", ".join(sorted(c.name for c in tops)) or "(none: the hierarchy is cyclic)"
    raise CliError(f"the library has {len(tops)} top cells ({names}); choose one with --cell")


def _scale(lib) -> float:
    """Factor from library user units to um."""
    return float(lib.unit) / 1e-6


def _bbox(points: list[np.ndarray]) -> list[list[float]] | None:
    if not points:
        return None
    allp = np.concatenate(points, axis=0)
    return [[float(allp[:, 0].min()), float(allp[:, 1].min())], [float(allp[:, 0].max()), float(allp[:, 1].max())]]


def read_cell(path: str, cell: str | None = None):
    """A flattened :class:`~photonix.layout.Cell` (geometry only, um) of one GDS cell."""
    from photonix.layout import GEOMETRY_ONLY, Cell

    lib = _library(path)
    gcell = _select(lib, cell)
    scale = _scale(lib)
    out = Cell(gcell.name, model=GEOMETRY_ONLY)
    for poly in gcell.get_polygons():
        out.add_polygon(np.asarray(poly.points) * scale, (int(poly.layer), int(poly.datatype)))
    return out, lib


def run_inspect(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    lib = _library(args.file)
    scale = _scale(lib)
    cells = []
    for c in sorted(lib.cells, key=lambda c: c.name):
        bb = c.bounding_box()
        cells.append({
            "name": c.name,
            "polygons": len(c.polygons),
            "paths": len(c.paths),
            "references": len(c.references),
            "labels": len(c.labels),
            "bbox_um": None if bb is None else (np.asarray(bb, dtype=float) * scale).tolist(),
        })
    top = _select(lib, args.cell)
    layers: dict[tuple[int, int], dict[str, Any]] = {}
    pts_all = []
    for poly in top.get_polygons():
        key = (int(poly.layer), int(poly.datatype))
        entry = layers.setdefault(key, {"layer": key[0], "datatype": key[1], "polygons": 0, "drawn_area_um2": 0.0})
        entry["polygons"] += 1
        entry["drawn_area_um2"] += float(poly.area()) * scale**2
        pts_all.append(np.asarray(poly.points) * scale)
    labels = sort_ports({str(lbl.text) for lbl in top.get_labels()})
    payload = {
        "schema": INSPECT_SCHEMA,
        "file": str(Path(args.file)),
        "library": lib.name,
        "unit_m": float(lib.unit),
        "precision_m": float(lib.precision),
        "top_cells": sorted(c.name for c in lib.top_level()),
        "cells": cells,
        "cell": top.name,
        "bbox_um": _bbox(pts_all),
        "layers": [layers[k] for k in sorted(layers)],
        "labels": labels,
    }
    return payload, ExitCode.OK


def render_inspect(payload: dict[str, Any]) -> str:
    lines = [f"{payload['file']}: library {payload['library']!r}, unit {payload['unit_m']:g} m, "
             f"precision {payload['precision_m']:g} m, {len(payload['cells'])} cells, "
             f"top: {', '.join(payload['top_cells'])}"]
    lines.append("  cell                       polygons  paths  refs  labels")
    for c in payload["cells"]:
        lines.append(f"  {c['name']:<26} {c['polygons']:>8}  {c['paths']:>5}  {c['references']:>4}  {c['labels']:>6}")
    bb = payload["bbox_um"]
    box = "empty" if bb is None else f"({bb[0][0]:g}, {bb[0][1]:g}) - ({bb[1][0]:g}, {bb[1][1]:g}) um"
    lines.append(f"cell {payload['cell']!r} flattened: bbox {box}")
    for la in payload["layers"]:
        lines.append(f"  layer {la['layer']}/{la['datatype']}: {la['polygons']} polygons, "
                     f"drawn area {la['drawn_area_um2']:.4g} um^2")
    if payload["labels"]:
        lines.append(f"  labels: {' '.join(payload['labels'])}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# DRC pre-check
# --------------------------------------------------------------------------- #
_RULE_NAMES = ("min_width", "min_space", "min_area")


def _layer_key(text: str) -> tuple[int, int]:
    parts = str(text).replace(":", "/").split("/")
    try:
        if len(parts) == 1:
            return (int(parts[0]), 0)
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    except ValueError:
        pass
    raise CliError(f"rules: layer key {text!r} must be LAYER or LAYER/DATATYPE")


def load_rules(path: str) -> dict[tuple[int, int], dict[str, float]]:
    """Parse a rules JSON file: ``{"1/0": {"min_width": w, "min_space": s, "min_area": a}}``.

    The mapping may also be wrapped as ``{"format": "photonix.drc_rules",
    "version": 1, "rules": {...}}``.
    """
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CliError(f"{source}: no such file") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CliError(f"{source}: not valid JSON ({exc})") from exc
    if isinstance(data, Mapping) and "rules" in data:
        if data.get("format", RULES_FORMAT) != RULES_FORMAT or not is_version(data.get("version", 1)):
            raise CliError(f"{source}: expected format {RULES_FORMAT!r} version 1")
        data = data["rules"]
    if not isinstance(data, Mapping) or not data:
        raise CliError(f"{source}: expected a non-empty object mapping layers to rules")
    rules: dict[tuple[int, int], dict[str, float]] = {}
    for key, values in data.items():
        if not isinstance(values, Mapping):
            raise CliError(f"{source}: rules for layer {key!r} must be an object")
        layer = _layer_key(key)
        for name, value in values.items():
            if name not in _RULE_NAMES:
                raise CliError(f"{source}: unknown rule {name!r} on layer {key!r}; use {', '.join(_RULE_NAMES)}")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or value <= 0:
                raise CliError(f"{source}: {name} on layer {key!r} must be a positive number")
            rules.setdefault(layer, {})[name] = float(value)
    return rules


def run_precheck(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    require("gdstk", "the DRC pre-check")
    if args.pdk is None and args.rules is None:
        raise CliError("give the design rules with --pdk NAME and/or --rules FILE")
    if args.max_violations < 0:
        raise CliError("--max-violations must be non-negative")
    rules: dict[tuple[int, int], dict[str, float]] = {}
    sources = []
    if args.pdk is not None:
        from .mode import load_pdk

        pdk = load_pdk(args.pdk)
        for name, layer_rules in pdk.design_rules.items():
            rules.setdefault(pdk.layers[name].tuple, {}).update(layer_rules)
        sources.append(f"pdk:{pdk.name}")
    if args.rules is not None:
        for layer, layer_rules in load_rules(args.rules).items():
            rules.setdefault(layer, {}).update(layer_rules)
        sources.append(f"rules:{args.rules}")
    if not rules:
        raise CliError("no design rules to check")
    cell, _lib = read_cell(args.file, args.cell)

    from photonix.layout import run_drc_precheck

    report = run_drc_precheck(cell, rules)
    violations = [{"rule": v.rule, "layer": list(v.layer), "value": v.value, "bbox_um": [list(c) for c in v.bbox],
                   "area_um2": v.area} for v in report.violations]
    summary = [{"rule": rule, "layer": list(layer), "count": n}
               for (rule, layer), n in sorted(report.summary().items())]
    payload = {
        "schema": PRECHECK_SCHEMA,
        "file": str(Path(args.file)),
        "cell": cell.name,
        "rule_sources": sources,
        "rules": [{"layer": list(layer), "rule": name, "value": value}
                  for layer, values in sorted(rules.items()) for name, value in sorted(values.items())],
        "layers_present": [list(layer) for layer in sorted(cell.layers)],
        "unchecked_layers": [list(layer) for layer in sorted(cell.layers - set(rules))],
        "clean": report.clean,
        "violation_count": len(violations),
        "summary": summary,
        "violations": violations[: args.max_violations],
        "truncated": len(violations) > args.max_violations,
    }
    return payload, ExitCode.OK if report.clean else ExitCode.CHECK_FAILED


def render_precheck(payload: dict[str, Any]) -> str:
    lines = [f"{payload['file']} cell {payload['cell']!r}: DRC pre-check with {', '.join(payload['rule_sources'])}"]
    for r in payload["rules"]:
        lines.append(f"  rule {r['rule']:<9} layer {r['layer'][0]}/{r['layer'][1]}: {r['value']:g}")
    if payload["unchecked_layers"]:
        lines.append("  layers without rules (not checked): "
                     + ", ".join(f"{la[0]}/{la[1]}" for la in payload["unchecked_layers"]))
    if payload["clean"]:
        lines.append("clean")
        return "\n".join(lines)
    lines.append(f"{payload['violation_count']} violation(s)")
    for s in payload["summary"]:
        lines.append(f"  {s['rule']:<9} layer {s['layer'][0]}/{s['layer'][1]}: {s['count']}")
    for v in payload["violations"]:
        (x0, y0), (x1, y1) = v["bbox_um"]
        lines.append(f"    {v['rule']} {v['value']:g} at ({x0:.4g}, {y0:.4g})-({x1:.4g}, {y1:.4g}) um")
    if payload["truncated"]:
        lines.append(f"    ... ({payload['violation_count'] - len(payload['violations'])} more; see --max-violations)")
    return "\n".join(lines)
