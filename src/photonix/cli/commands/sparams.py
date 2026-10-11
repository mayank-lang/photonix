"""``photonix sparams check|convert``: gate S-parameter files on physicality and convert formats."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .._common import CliError, ExitCode, add_json_flag, load_dataset, physicality, schema_id, write_dataset

__all__ = ["register", "run_check", "render_check", "run_convert", "render_convert"]

CHECK_SCHEMA = schema_id("sparams-check")
CONVERT_SCHEMA = schema_id("sparams-convert")


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("sparams", help="check (passivity, reciprocity) or convert S-parameter files",
                       description="Work with S-parameter files: Touchstone 1.0 single-ended S RI (.sNp), "
                                   "photonix dataset NPZ (.npz) and photonix.sparameters JSON (.json).")
    actions = p.add_subparsers(dest="action", metavar="ACTION", required=True)

    c = actions.add_parser(
        "check", help="exit 1 unless the data is passive (and reciprocal)",
        description="Measure passivity (largest singular value <= 1), reciprocity (S == S^T) and losslessness "
                    "(S^H S == I) at every wavelength with photonix.analyze_sparameters. Exit 1 when a gated "
                    "check fails. Assumes power-normalized waves with equal real reference impedances.")
    c.add_argument("file", help=".sNp, .npz or .json S-parameter file")
    c.add_argument("--atol", type=float, default=1e-6,
                   help="absolute tolerance for every check (default 1e-6; measured data often needs 1e-4)")
    c.add_argument("--passivity-atol", type=float, default=None, help="override --atol for passivity")
    c.add_argument("--reciprocity-atol", type=float, default=None, help="override --atol for reciprocity")
    c.add_argument("--lossless-atol", type=float, default=None, help="override --atol for losslessness")
    c.add_argument("--no-reciprocity", action="store_true",
                   help="report but do not gate reciprocity (non-reciprocal devices, e.g. isolators)")
    c.add_argument("--lossless", action="store_true", help="also gate losslessness (unitarity)")
    add_json_flag(c)
    c.set_defaults(handler=run_check, render=render_check)

    v = actions.add_parser("convert", help="convert between .sNp, .npz, .json and .csv",
                           description="Read one S-parameter file and write it in the format its suffix names. "
                                       "Port names and metadata survive every conversion except to CSV.")
    v.add_argument("input", help="source file (.sNp, .npz or .json)")
    v.add_argument("output", help="destination file (.sNp, .npz, .json or .csv)")
    v.add_argument("--frequency-unit", default="ghz", choices=("hz", "khz", "mhz", "ghz"),
                   help="Touchstone frequency unit (default ghz)")
    add_json_flag(v)
    v.set_defaults(handler=run_convert, render=render_convert)


def _tol(value: float | None, default: float, name: str) -> float:
    tol = default if value is None else value
    if not tol >= 0:
        raise CliError(f"{name} must be non-negative")
    return float(tol)


def run_check(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    ds = load_dataset(args.file)
    atol = _tol(args.atol, 1e-6, "--atol")
    phys = physicality(ds, passivity_atol=_tol(args.passivity_atol, atol, "--passivity-atol"),
                       reciprocity_atol=_tol(args.reciprocity_atol, atol, "--reciprocity-atol"),
                       lossless_atol=_tol(args.lossless_atol, atol, "--lossless-atol"))
    checks = [
        {"name": "passivity", "passed": phys["passive"], "gated": True,
         "value": phys["worst_passivity_violation"], "atol": phys["passivity_atol"],
         "at_wl_um": phys["max_singular_value_wl_um"],
         "measure": "max(0, largest singular value - 1)"},
        {"name": "reciprocity", "passed": phys["reciprocal"], "gated": not args.no_reciprocity,
         "value": phys["worst_reciprocity_error"], "atol": phys["reciprocity_atol"],
         "at_wl_um": phys["worst_reciprocity_wl_um"], "measure": "max |S_ij - S_ji|"},
        {"name": "losslessness", "passed": phys["lossless"], "gated": bool(args.lossless),
         "value": phys["worst_unitarity_error"], "atol": phys["lossless_atol"],
         "at_wl_um": phys["worst_unitarity_wl_um"], "measure": "spectral norm of S^H S - I"},
    ]
    ok = all(c["passed"] for c in checks if c["gated"])
    payload = {
        "schema": CHECK_SCHEMA,
        "file": str(Path(args.file)),
        "ports": list(ds.ports),
        "n_wavelengths": int(ds.wavelengths.size),
        "wl_min_um": float(ds.wavelengths[0]),
        "wl_max_um": float(ds.wavelengths[-1]),
        "max_singular_value": phys["max_singular_value"],
        "checks": checks,
        "ok": ok,
    }
    return payload, ExitCode.OK if ok else ExitCode.CHECK_FAILED


def render_check(payload: dict[str, Any]) -> str:
    lines = [f"{payload['file']}: {len(payload['ports'])} ports, {payload['n_wavelengths']} wavelengths "
             f"{payload['wl_min_um']:g}-{payload['wl_max_um']:g} um"]
    for c in payload["checks"]:
        if c["gated"]:
            status = "pass" if c["passed"] else "FAIL"
        else:
            status = ("yes" if c["passed"] else "no") + " (reported, not gated)"
        lines.append(f"  {c['name']:<13} {status}: {c['measure']} = {c['value']:.3e} "
                     f"(atol {c['atol']:.1e}, worst at {c['at_wl_um']:.6g} um)")
    lines.append("OK" if payload["ok"] else "FAILED")
    return "\n".join(lines)


def run_convert(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    ds = load_dataset(args.input)
    fmt = write_dataset(ds, args.output, frequency_unit=args.frequency_unit)
    return {"schema": CONVERT_SCHEMA, "input": str(Path(args.input)), "output": str(Path(args.output)),
            "format": fmt, "ports": list(ds.ports), "n_wavelengths": int(ds.wavelengths.size)}, ExitCode.OK


def render_convert(payload: dict[str, Any]) -> str:
    return (f"wrote {payload['output']} ({payload['format']}): {len(payload['ports'])} ports, "
            f"{payload['n_wavelengths']} wavelengths from {payload['input']}")
