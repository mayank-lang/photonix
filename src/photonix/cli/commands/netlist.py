"""``photonix netlist simulate FILE``: simulate a photonix Netlist JSON over wavelength."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .._common import CliError, ExitCode, parse_params, parse_wavelengths, schema_id
from .sweep import add_output_options, evaluate, render_spectrum, spectrum_payload

__all__ = ["register", "run_simulate", "render_simulate"]

SIMULATE_SCHEMA = schema_id("netlist-simulate")


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("netlist", help="simulate a Netlist JSON file (Netlist.to_json format)",
                       description="Netlist tools for the photonix.netlist JSON format written by "
                                   "photonix.circuit.Netlist.to_json.")
    actions = p.add_subparsers(dest="action", metavar="ACTION", required=True)
    s = actions.add_parser(
        "simulate", help="compile with circuit_from_netlist and sweep wavelength",
        description="Load a netlist, resolve its models against photonix.components.MODELS (or a PDK's "
                    "registry with --pdk), solve the circuit at every wavelength and report or save the "
                    "external-port S-parameters.")
    s.add_argument("file", help="netlist JSON file (format photonix.netlist, version 1)")
    s.add_argument("-s", "--set", action="append", default=[], metavar="INSTANCE.NAME=VALUE",
                   help="override one instance setting at call time; repeatable")
    add_output_options(s)
    s.set_defaults(handler=run_simulate, render=render_simulate)


def _overrides(items: list[str], instances: dict[str, str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        target, sep, _ = item.partition("=")
        inst, dot, name = target.partition(".")
        if not sep or not dot or not inst or not name:
            raise CliError(f"--set {item!r}: expected INSTANCE.NAME=VALUE")
        if inst not in instances:
            raise CliError(f"--set {item!r}: no instance {inst!r}; instances: {', '.join(sorted(instances))}")
        out.setdefault(inst, {}).update(parse_params([f"{name}={item.split('=', 1)[1]}"]))
    return out


def run_simulate(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    import json

    from photonix.circuit import Netlist, circuit_from_netlist

    from .sweep import model_registry

    source = Path(args.file)
    if not source.is_file():
        raise CliError(f"{source}: no such file")
    try:
        netlist = Netlist.from_json(source)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise CliError(f"{source}: not a valid photonix netlist ({exc})") from exc
    models = model_registry(args)
    missing = sorted({m for m in netlist.instances.values() if m not in models})
    if missing:
        raise CliError(f"{source}: unknown model(s) {', '.join(missing)}; run `photonix models` for the registry")
    try:
        circuit = circuit_from_netlist(netlist, models)
    except (ValueError, TypeError, KeyError) as exc:
        raise CliError(f"{source}: {exc}") from exc
    overrides = _overrides(args.set, netlist.instances)
    wavelengths = parse_wavelengths(args.wl)
    ds, caught = evaluate(circuit, wavelengths, overrides, label=f"netlist {source.name!r}")
    extra = {"netlist": str(source), "name": netlist.name, "instances": len(netlist.instances),
             "connections": len(netlist.connections), "pdk": args.pdk, "cross_section": args.xs,
             "settings": overrides}
    return spectrum_payload(ds, args, schema=SIMULATE_SCHEMA, extra=extra, warnings_=caught), ExitCode.OK


def render_simulate(payload: dict[str, Any]) -> str:
    name = f" {payload['name']!r}" if payload["name"] else ""
    title = (f"netlist{name} from {payload['netlist']}: {payload['instances']} instances, "
             f"{payload['connections']} connections")
    return render_spectrum(payload, title)
