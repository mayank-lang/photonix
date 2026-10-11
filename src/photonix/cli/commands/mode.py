"""``photonix mode``: full-vector FDE modes of a strip waveguide or a PDK cross-section."""
from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from typing import Any

from .._common import CliError, ExitCode, add_json_flag, schema_id

__all__ = ["register", "run", "render", "load_pdk", "material_index", "PDKS"]

SCHEMA = schema_id("mode")


def _demo():
    from photonix.pdk import demo_pdk

    return demo_pdk()


#: PDKs the command line can name (``--pdk``).
PDKS: dict[str, Callable] = {"demo": _demo, "photonix_demo": _demo}


def load_pdk(name: str):
    """Instantiate a PDK by its command-line name."""
    key = name.strip().lower()
    if key not in PDKS:
        raise CliError(f"unknown PDK {name!r}; available: {', '.join(sorted(PDKS))}")
    return PDKS[key]()


_MATERIALS = {
    "si": "silicon", "silicon": "silicon",
    "sio2": "silica", "silica": "silica", "oxide": "silica",
    "si3n4": "silicon_nitride", "sin": "silicon_nitride", "nitride": "silicon_nitride",
}


def material_index(spec: str, wl: float) -> tuple[float, str]:
    """Refractive index of ``spec`` at ``wl``: a number, ``air``, or a Sellmeier material name.

    Names: ``Si`` (Li), ``SiO2`` (Malitson 1965), ``Si3N4`` (Luke 2015), from
    :mod:`photonix.em.materials`. Returns ``(index, description)``.

    >>> material_index("3.4757", 1.55)
    (3.4757, '3.4757')
    >>> n, what = material_index("SiO2", 1.55); round(n, 3), what
    (1.444, 'silica (Sellmeier)')
    """
    text = spec.strip()
    try:
        value = float(text)
    except ValueError:
        value = None
    if value is not None:
        if not math.isfinite(value) or value <= 0:
            raise CliError(f"refractive index {spec!r} must be positive and finite")
        return value, text
    key = text.lower()
    if key == "air":
        return 1.0, "air"
    if key not in _MATERIALS:
        raise CliError(f"unknown material {spec!r}; use a number, air, or one of Si, SiO2, Si3N4")
    from photonix.em import materials

    fn = getattr(materials, _MATERIALS[key])
    return float(fn(wl)), f"{_MATERIALS[key]} (Sellmeier)"


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "mode", help="solve full-vector waveguide modes (n_eff, polarization, guidance)",
        description="Solve hybrid modes of a rectangular strip (or of a PDK cross-section through its layer "
                    "stack) with the full-vector Yee-grid FDE solver photonix.em.solve_modes_fullvector. "
                    "With default arguments the fundamental n_eff equals photonix.em.n_eff_fullvector().")
    p.add_argument("--wl", type=float, default=1.55, help="wavelength in um (default 1.55)")
    p.add_argument("--width", type=float, default=None, help="core width in um (default 0.5, or the PDK's)")
    p.add_argument("--thickness", type=float, default=None, help="core thickness in um (default 0.22)")
    p.add_argument("--core", default="3.4757", help="core index or material: Si, SiO2, Si3N4, air (default 3.4757)")
    p.add_argument("--clad", default="1.444", help="cladding index or material (default 1.444)")
    p.add_argument("--pdk", default=None, help="take the geometry and materials from a PDK (e.g. demo)")
    p.add_argument("--xs", default=None, help="PDK cross-section name (default: the PDK's default)")
    p.add_argument("--modes", type=int, default=1, help="number of modes (default 1)")
    p.add_argument("--resolution", type=int, default=40, help="grid points per um (default 40)")
    p.add_argument("--margin", type=float, default=1.5, help="cladding margin around the core in um (default 1.5)")
    p.add_argument("--converge", action="store_true",
                   help="also run a 3-level grid study (resolution x 1, 1.5, 2.25) of the fundamental n_eff")
    p.add_argument("--rtol", type=float, default=1e-3, help="relative tolerance of the grid study (default 1e-3)")
    p.add_argument("--require-converged", action="store_true",
                   help="exit 1 when the grid study does not certify convergence (implies --converge)")
    add_json_flag(p)
    p.set_defaults(handler=run, render=render)


def _solver(args: argparse.Namespace) -> tuple[Callable[[int, int], Any], dict[str, Any]]:
    from photonix.em import solve_modes_fullvector

    if args.wl <= 0 or not math.isfinite(args.wl):
        raise CliError("--wl must be positive and finite (um)")
    if args.modes < 1:
        raise CliError("--modes must be at least 1")
    if args.resolution < 4:
        raise CliError("--resolution must be at least 4 points per um")
    if args.margin <= 0:
        raise CliError("--margin must be positive")
    if args.pdk is None:
        if args.xs is not None:
            raise CliError("--xs needs --pdk")
        width = 0.5 if args.width is None else args.width
        thickness = 0.22 if args.thickness is None else args.thickness
        if width <= 0 or thickness <= 0:
            raise CliError("--width and --thickness must be positive")
        n_core, core = material_index(args.core, args.wl)
        n_clad, clad = material_index(args.clad, args.wl)
        if n_core <= n_clad:
            raise CliError(f"the core index ({n_core:.6g}) must exceed the cladding index ({n_clad:.6g})")
        inputs = {"geometry": "rectangular strip", "width_um": width, "thickness_um": thickness,
                  "core": core, "clad": clad, "n_core": n_core, "n_clad": n_clad}

        def solve(resolution: int, num_modes: int):
            return solve_modes_fullvector(wl=args.wl, width=width, thickness=thickness, n_core=n_core,
                                          n_clad=n_clad, num_modes=num_modes, resolution=resolution,
                                          margin=args.margin)
    else:
        if args.thickness is not None or args.core != "3.4757" or args.clad != "1.444":
            raise CliError("--thickness/--core/--clad come from the PDK layer stack; do not combine them with --pdk")
        pdk = load_pdk(args.pdk)
        try:
            xs = pdk.cross_section(args.xs)
        except KeyError as exc:
            raise CliError(f"{exc.args[0]}; available: {', '.join(sorted(pdk.cross_sections))}") from exc
        if pdk.layer_stack is None:
            raise CliError(f"PDK {pdk.name!r} has no layer stack to build a cross-section from")
        width = xs.width if args.width is None else args.width
        if width <= 0:
            raise CliError("--width must be positive")
        stack = pdk.layer_stack
        inputs = {"geometry": "pdk cross-section", "pdk": pdk.name, "cross_section": xs.name,
                  "layer": list(xs.layer), "width_um": width}

        def solve(resolution: int, num_modes: int):
            cs = stack.cross_section({xs.layer: width}, wl=args.wl, resolution=resolution, margin=args.margin)
            return solve_modes_fullvector(wl=args.wl, eps=cs.eps, grid=(cs.x, cs.y), num_modes=num_modes)

    inputs.update(wl_um=args.wl, resolution=args.resolution, margin_um=args.margin, num_modes=args.modes)
    return solve, inputs


def run(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    solve, inputs = _solver(args)
    result = solve(args.resolution, args.modes)
    modes = []
    for i, n in enumerate(result.n_eff):
        te = None if result.te_fraction is None else float(result.te_fraction[i])
        modes.append({
            "index": i,
            "n_eff": float(complex(n).real),
            "n_eff_imag": float(complex(n).imag),
            "te_fraction": te,
            "polarization": None if te is None else ("TE-like" if te > 0.5 else "TM-like"),
            "guided": None if result.guided is None else bool(result.guided[i]),
        })
    payload: dict[str, Any] = {"schema": SCHEMA, "solver": "photonix.em.solve_modes_fullvector",
                               "inputs": inputs, "modes": modes, "convergence": None}
    code = ExitCode.OK
    if args.converge or args.require_converged:
        from photonix.em import estimate_convergence

        if not 0 < args.rtol < 1:
            raise CliError("--rtol must lie in (0, 1)")
        levels = [args.resolution, round(1.5 * args.resolution), round(2.25 * args.resolution)]
        values = [result.neff0] + [solve(r, 1).neff0 for r in levels[1:]]
        study = estimate_convergence(levels, values, rtol=args.rtol)
        payload["convergence"] = {
            "quantity": "Re(n_eff) of mode 0",
            "resolutions": list(study.resolutions),
            "values": [float(v) for v in study.values],
            "extrapolated": float(study.extrapolated),
            "observed_order": float(study.observed_order),
            "estimated_absolute_error": float(study.estimated_absolute_error),
            "grid_convergence_index": float(study.grid_convergence_index),
            "asymptotic": bool(study.asymptotic),
            "converged": bool(study.converged),
            "rtol": float(study.rtol),
        }
        if args.require_converged and not study.converged:
            code = ExitCode.CHECK_FAILED
    return payload, code


def render(payload: dict[str, Any]) -> str:
    inp = payload["inputs"]
    if inp["geometry"] == "rectangular strip":
        head = (f"strip {inp['width_um']:g} x {inp['thickness_um']:g} um, core {inp['core']} (n={inp['n_core']:.6g}), "
                f"clad {inp['clad']} (n={inp['n_clad']:.6g})")
    else:
        head = f"PDK {inp['pdk']} cross-section {inp['cross_section']!r} (layer {tuple(inp['layer'])}), " \
               f"width {inp['width_um']:g} um"
    lines = [head, f"wl = {inp['wl_um']:g} um, resolution {inp['resolution']}/um, margin {inp['margin_um']:g} um "
                   f"({payload['solver']})",
             "  mode   n_eff          TE fraction  polarization  guided"]
    for m in payload["modes"]:
        te = "-" if m["te_fraction"] is None else f"{m['te_fraction']:.4f}"
        lines.append(f"  {m['index']:>4}   {m['n_eff']:.10f}   {te:>11}  {m['polarization'] or '-':<12}  {m['guided']}")
    c = payload["convergence"]
    if c is not None:
        lines.append(f"grid study {c['resolutions']}: values {', '.join(f'{v:.6f}' for v in c['values'])}")
        lines.append(f"  extrapolated {c['extrapolated']:.6f}, observed order {c['observed_order']:.2f}, "
                     f"GCI {c['grid_convergence_index']:.2e}, asymptotic {c['asymptotic']}, converged {c['converged']} "
                     f"(rtol {c['rtol']:g})")
    return "\n".join(lines)
