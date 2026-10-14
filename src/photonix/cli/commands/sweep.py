"""``photonix sweep MODEL``: evaluate a compact model over wavelength and save the S-parameters."""
from __future__ import annotations

import argparse
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from .._common import (
    CliError,
    ExitCode,
    add_json_flag,
    check_output,
    dataset_document,
    jsonable,
    parse_params,
    parse_wavelengths,
    physicality,
    schema_id,
    sort_ports,
    write_dataset,
)

__all__ = ["register", "run", "render", "evaluate", "spectrum_payload", "render_spectrum", "add_output_options",
           "model_registry"]

SCHEMA = schema_id("sweep")


def add_output_options(p: argparse.ArgumentParser) -> None:
    """Options shared by every command that produces a spectrum."""
    p.add_argument("--wl", default="1.5:1.6:201",
                   help="wavelengths in um: START:STOP:COUNT, a comma list, or one value (default 1.5:1.6:201)")
    p.add_argument("-o", "--output", default=None,
                   help="write the S-parameters; the suffix picks the format: .sNp (Touchstone 1.0 RI), .npz, "
                        ".json (photonix.sparameters) or .csv (real/imaginary columns)")
    p.add_argument("--force", action="store_true", help="overwrite an existing --output file")
    p.add_argument("--frequency-unit", default="ghz", choices=("hz", "khz", "mhz", "ghz"),
                   help="Touchstone frequency unit (default ghz)")
    p.add_argument("--pdk", default=None, help="use a PDK's model registry and cross-section (e.g. demo)")
    p.add_argument("--xs", default=None, help="PDK cross-section whose waveguide settings are bound")
    add_json_flag(p)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("sweep", help="evaluate a registered model over wavelength and save its S-parameters",
                       description="Evaluate a model from photonix.components.MODELS (or a PDK's registry) at "
                                   "the requested wavelengths and report or save its S-parameters together with "
                                   "a passivity/reciprocity summary.")
    p.add_argument("model", help="model name (see `photonix models`)")
    p.add_argument("-p", "--param", action="append", default=[], metavar="NAME=VALUE",
                   help="model parameter; repeatable. Values are JSON literals (25, 2.45, true, null, [1,2]) "
                        "or plain strings")
    add_output_options(p)
    p.set_defaults(handler=run, render=render)


def evaluate(fn: Callable, wavelengths: np.ndarray, settings: Mapping[str, Any], *, label: str):
    """Evaluate a model into an :class:`~photonix.SParameterDataset` with naturally ordered ports."""
    from photonix.core import SParameterDataset

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            sdict = fn(wl=wavelengths, **settings)
        except (TypeError, ValueError, KeyError) as exc:
            raise CliError(f"{label} rejected the settings: {exc}") from exc
    ports = sort_ports({p for pair in sdict for p in pair})
    values = {key: np.asarray(value) for key, value in sdict.items()}
    ds = SParameterDataset.from_sdict(wavelengths, values, ports=ports,
                                      metadata={"source": label, "settings": jsonable(dict(settings))})
    return ds, sorted({str(w.message) for w in caught})


def _paths(ds) -> list[dict[str, Any]]:
    out = []
    for j, p_in in enumerate(ds.ports):
        for i, p_out in enumerate(ds.ports):
            power = np.abs(ds.s[:, i, j]) ** 2
            if float(power.max()) <= 1e-30:
                continue
            db = 10.0 * np.log10(np.maximum(power, 1e-300))
            out.append({"in": p_in, "out": p_out, "min_db": float(db.min()), "max_db": float(db.max()),
                        "mean_power": float(power.mean())})
    return out


def spectrum_payload(ds, args: argparse.Namespace, *, schema: str, extra: Mapping[str, Any],
                     warnings_: list[str]) -> dict[str, Any]:
    """Common payload of ``sweep`` and ``netlist simulate``; writes ``--output`` if given."""
    output = None
    if args.output is not None:
        fmt = write_dataset(ds, args.output, frequency_unit=args.frequency_unit)
        output = {"path": str(Path(args.output)), "format": fmt}
    payload: dict[str, Any] = {
        "schema": schema,
        **extra,
        "ports": list(ds.ports),
        "n_wavelengths": int(ds.wavelengths.size),
        "wl_min_um": float(ds.wavelengths[0]),
        "wl_max_um": float(ds.wavelengths[-1]),
        "paths": _paths(ds),
        "physicality": physicality(ds),
        "warnings": warnings_,
        "output": output,
        "data": None if output is not None else dataset_document(ds),
    }
    return payload


def model_registry(args: argparse.Namespace) -> Mapping[str, Callable]:
    if args.pdk is None:
        if args.xs is not None:
            raise CliError("--xs needs --pdk")
        from photonix.components import MODELS

        return cast(Mapping[str, Callable], MODELS)
    from .mode import load_pdk

    pdk = load_pdk(args.pdk)
    try:
        return pdk.models(args.xs)
    except KeyError as exc:
        raise CliError(f"{exc.args[0]}; available: {', '.join(sorted(pdk.cross_sections))}") from exc


def run(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    from .models import resolve_model

    if args.output is not None:
        check_output(args.output, force=args.force)
    fn = resolve_model(args.model, model_registry(args))
    settings = parse_params(args.param, allowed=fn)
    wavelengths = parse_wavelengths(args.wl)
    ds, caught = evaluate(fn, wavelengths, settings, label=f"model {args.model!r}")
    extra = {"model": args.model, "pdk": args.pdk, "cross_section": args.xs, "settings": jsonable(dict(settings))}
    return spectrum_payload(ds, args, schema=SCHEMA, extra=extra, warnings_=caught), ExitCode.OK


def render_spectrum(payload: dict[str, Any], title: str) -> str:
    """Text rendering shared by ``sweep`` and ``netlist simulate``."""
    phys = payload["physicality"]
    lines = [title,
             f"  {len(payload['ports'])} ports ({', '.join(payload['ports'])}), {payload['n_wavelengths']} "
             f"wavelengths {payload['wl_min_um']:g}-{payload['wl_max_um']:g} um"]
    if payload.get("settings"):
        lines.append("  settings: " + ", ".join(f"{k}={v!r}" for k, v in payload["settings"].items()))
    lines.append("  path          |S|^2 min dB   max dB")
    for p in payload["paths"]:
        lines.append(f"  {p['in'] + ' -> ' + p['out']:<12}  {p['min_db']:>10.3f}  {p['max_db']:>8.3f}")
    state = [("passive" if phys["passive"] else "NOT passive"), ("reciprocal" if phys["reciprocal"] else
             "NOT reciprocal"), ("lossless" if phys["lossless"] else "lossy")]
    lines.append(f"  physicality: {', '.join(state)} (max singular value {phys['max_singular_value']:.9f})")
    lines += [f"  warning: {w}" for w in payload["warnings"]]
    if payload["output"] is not None:
        lines.append(f"wrote {payload['output']['path']} ({payload['output']['format']})")
    return "\n".join(lines)


def render(payload: dict[str, Any]) -> str:
    where = f" from PDK {payload['pdk']}" if payload["pdk"] else ""
    return render_spectrum(payload, f"model {payload['model']}{where}")
