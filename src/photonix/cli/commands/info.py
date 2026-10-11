"""``photonix info``: the numerical runtime and optional integrations (the 0.1 behaviour)."""
from __future__ import annotations

import argparse
from typing import Any

from .._common import ExitCode, add_json_flag, schema_id

__all__ = ["register", "run", "render"]

SCHEMA = schema_id("info")


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("info", help="show the backend, precision, devices and optional dependencies",
                       description="Show the numerical runtime: backend, precision, devices, optional Python "
                                   "packages and KLayout discovery (a reproducibility manifest).")
    add_json_flag(p)
    p.set_defaults(handler=run, render=render)


def run(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    from photonix.diagnostics import runtime_info

    return {"schema": SCHEMA, **runtime_info().as_dict()}, ExitCode.OK


def render(payload: dict[str, Any]) -> str:
    from photonix.diagnostics import DependencyStatus, RuntimeInfo, format_runtime_info

    fields = {k: v for k, v in payload.items() if k != "schema"}
    fields["optional_dependencies"] = {
        name: DependencyStatus(**status) for name, status in fields["optional_dependencies"].items()
    }
    return format_runtime_info(RuntimeInfo(**fields))
