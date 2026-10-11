"""``photonix models [NAME]``: the compact-model registry with ports and parameter defaults."""
from __future__ import annotations

import argparse
import difflib
import inspect
import warnings
from collections.abc import Callable, Mapping
from typing import Any, cast

from .._common import CliError, ExitCode, add_json_flag, jsonable, schema_id, sort_ports

__all__ = ["register", "run", "render", "describe_model", "resolve_model"]

SCHEMA = schema_id("models")


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("models", help="list registered compact models, their ports and parameter defaults",
                       description="List photonix.components.MODELS (the registry the circuit solver and "
                                   "layout extraction use). Ports come from one evaluation at the default "
                                   "parameters and wl = 1.55 um; parameter defaults from the signature.")
    p.add_argument("name", nargs="?", help="show one model in detail (with its docstring)")
    p.add_argument("--no-ports", action="store_true", help="skip the default evaluation that discovers ports")
    add_json_flag(p)
    p.set_defaults(handler=run, render=render)


def registry(pdk: str | None = None) -> Mapping[str, Callable]:
    """``photonix.components.MODELS`` or a named PDK's bound model registry."""
    if pdk is None:
        from photonix.components import MODELS

        return cast(Mapping[str, Callable], MODELS)
    from .mode import load_pdk

    return load_pdk(pdk).models()


def resolve_model(name: str, models: Mapping[str, Callable]) -> Callable:
    """Look up a model, suggesting close names when it is unknown."""
    if name in models:
        return models[name]
    close = difflib.get_close_matches(name, list(models), n=3)
    hint = f"; did you mean {', '.join(close)}?" if close else "; run `photonix models` for the list"
    raise CliError(f"unknown model {name!r}{hint}")


def _annotation(p: inspect.Parameter) -> str | None:
    if p.annotation is inspect.Parameter.empty:
        return None
    return p.annotation if isinstance(p.annotation, str) else getattr(p.annotation, "__name__", repr(p.annotation))


def describe_model(name: str, fn: Callable, *, ports: bool = True, doc: bool = False) -> dict[str, Any]:
    """Registry entry of one model: function, summary, parameters, ports."""
    target = inspect.unwrap(getattr(fn, "func", fn))
    params = []
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - every registry entry is a Python function
        signature = None
    if signature is not None:
        for pname, p in signature.parameters.items():
            if pname == "wl" or p.kind is not p.KEYWORD_ONLY:
                continue
            default = None if p.default is inspect.Parameter.empty else jsonable(p.default)
            params.append({"name": pname, "default": default, "required": p.default is inspect.Parameter.empty,
                           "annotation": _annotation(p)})
    text = inspect.getdoc(target) or ""
    entry: dict[str, Any] = {
        "name": name,
        "function": f"{getattr(target, '__module__', '?')}.{getattr(target, '__qualname__', name)}",
        "summary": text.strip().splitlines()[0] if text.strip() else "",
        "parameters": params,
        "waveguide_parameters": list(getattr(fn, "waveguide_parameters", ()) or ()),
        "ports": None,
        "warnings": [],
        "error": None,
    }
    if ports:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                entry["ports"] = sort_ports({p for pair in fn(wl=1.55) for p in pair})
            except Exception as exc:  # report, never crash the listing
                entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["warnings"] = sorted({str(w.message) for w in caught})
    if doc:
        entry["doc"] = text
    return entry


def run(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    models = registry()
    if args.name is not None:
        entries = [describe_model(args.name, resolve_model(args.name, models), ports=not args.no_ports, doc=True)]
    else:
        entries = [describe_model(n, fn, ports=not args.no_ports) for n, fn in sorted(models.items())]
    return {"schema": SCHEMA, "registry": "photonix.components.MODELS", "count": len(entries),
            "models": entries}, ExitCode.OK


def render(payload: dict[str, Any]) -> str:
    entries = payload["models"]
    if len(entries) == 1 and "doc" in entries[0]:
        e = entries[0]
        lines = [f"{e['name']}  ({e['function']})"]
        if e["ports"] is not None:
            lines.append(f"  ports: {' '.join(e['ports'])}")
        if e["error"]:
            lines.append(f"  default evaluation failed: {e['error']}")
        lines.append("  parameters:")
        lines += [f"    {p['name']} = {p['default']!r}" + (f"  [{p['annotation']}]" if p["annotation"] else "")
                  for p in e["parameters"]] or ["    (none)"]
        if e["waveguide_parameters"]:
            lines.append("  waveguide parameters (bound by a PDK cross-section): "
                         + ", ".join(e["waveguide_parameters"]))
        lines += ["", *(f"  {line}" if line else "" for line in e["doc"].splitlines())]
        return "\n".join(lines)
    width = max((len(e["name"]) for e in entries), default=4)
    lines = [f"{payload['count']} models in {payload['registry']}"]
    for e in entries:
        ports = "" if e["ports"] is None else f"[{len(e['ports'])} ports] "
        lines.append(f"  {e['name']:<{width}}  {ports}{e['summary']}")
    lines.append("Run `photonix models NAME` for parameters and documentation.")
    return "\n".join(lines)
