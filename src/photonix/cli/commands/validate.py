"""``photonix validate``: packaged analytic self-checks plus a reproducibility manifest.

Each check compares an installed photonix computation with a closed-form or
published reference, under a stated tolerance, so a fresh install (a new
platform, BLAS, NumPy/SciPy/JAX version or backend) can be certified to
reproduce them. The checks are *anchors*: they establish that the equations
and their implementation agree in limits where the answer is known
(evidence level 2 in ``docs/VALIDATION.md``), not that any particular device
model is calibrated.
"""
from __future__ import annotations

import argparse
import math
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from .._common import ExitCode, add_json_flag, schema_id

__all__ = ["register", "run", "render", "CHECKS", "Check"]

SCHEMA = schema_id("validate")


@dataclass(frozen=True)
class Check:
    """One self-check: ``compute() -> (value, expected, error)`` compared with ``tolerance``."""

    name: str
    description: str
    reference: str
    evidence: str
    tolerance: float
    quick: bool
    compute: Callable[[], tuple[float, float, float]]


def _ring_closed_form() -> tuple[float, float, float]:
    """Feedback-solved all-pass ring vs. t - a e^{-j phi} over 1 - t a e^{-j phi}."""
    import photonix as px
    from photonix import components as c

    wl = np.linspace(1.54, 1.56, 401)
    radius, coupling, loss = 10.0, 0.2, 2.0
    s = px.as_sdict(px.circuit.ring(radius=radius, coupling=coupling, loss_db_cm=loss)(wl=wl))
    through = np.asarray(s[("o1", "o2")])
    rt = np.asarray(c.straight(wl=wl, length=2 * math.pi * radius, neff=2.4, ng=4.2, loss_db_cm=loss)[("o1", "o2")])
    t = math.sqrt(1.0 - coupling)
    expected = (t - rt) / (1.0 - t * rt)
    return float(np.abs(through).min()), float(np.abs(expected).min()), float(np.max(np.abs(through - expected)))


def _mzi_closed_form() -> tuple[float, float, float]:
    """Two-coupler MZI netlist vs. its transfer-matrix product t^2 a1 + k^2 a2 and t k (a1 + a2)."""
    import photonix as px
    from photonix import components as c

    wl = np.linspace(1.5, 1.6, 201)
    s = px.as_sdict(px.circuit.mzi(delta_length=20.0, length=50.0, coupling=0.5)(wl=wl))
    a1 = np.asarray(c.straight(wl=wl, length=70.0)[("o1", "o2")])
    a2 = np.asarray(c.straight(wl=wl, length=50.0)[("o1", "o2")])
    t, k = math.sqrt(0.5), -1j * math.sqrt(0.5)
    bar, cross = t * t * a1 + k * k * a2, t * k * (a1 + a2)
    err = max(float(np.max(np.abs(np.asarray(s[("o1", "o4")]) - bar))),
              float(np.max(np.abs(np.asarray(s[("o1", "o3")]) - cross))))
    total = np.abs(np.asarray(s[("o1", "o4")])) ** 2 + np.abs(np.asarray(s[("o1", "o3")])) ** 2
    return float(total.min()), 1.0, max(err, float(np.max(np.abs(total - 1.0))))


def _bragg_peak() -> tuple[float, float, float]:
    """Uniform grating: |r|^2 = tanh^2(kappa L) exactly at the Bragg wavelength."""
    from photonix import components as c

    period, n_periods, dn = 0.3229, 300, 0.03
    wl_b = float(c.bragg_wavelength(period))
    kl = float(c.grating_kappa(dn, wl_b)) * period * n_periods
    r = float(np.abs(c.bragg_grating(wl=wl_b, period=period, n_periods=n_periods, dn=dn)[("o1", "o1")]) ** 2)
    expected = math.tanh(kl) ** 2
    return r, expected, abs(r - expected) / expected


def _slab(polarization: str) -> Callable[[], tuple[float, float, float]]:
    def compute() -> tuple[float, float, float]:
        from photonix import em

        value = float(em.slab_neff(polarization=polarization, resolution=80))
        expected = float(em.slab_neff_analytic(polarization=polarization))
        return value, expected, abs(value - expected) / expected

    compute.__doc__ = f"1-D finite-difference slab {polarization.upper()}0 vs. the transcendental equation."
    return compute


def _passivity_all_models() -> tuple[float, float, float]:
    """Largest singular value and reciprocity error of every registered model at its defaults."""
    import photonix as px
    from photonix.components import MODELS

    wl = np.linspace(1.5, 1.6, 5)
    worst_sv, worst_rec = 0.0, 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fn in cast(Mapping[str, Callable], MODELS).values():
            d = px.analyze_sparameters(fn(wl=wl))
            worst_sv = max(worst_sv, float(np.max(d.maximum_singular_value)))
            worst_rec = max(worst_rec, d.worst_reciprocity_error)
    return worst_sv, 1.0, max(worst_sv - 1.0, worst_rec, 0.0)


def _fullvector_soi() -> tuple[float, float, float]:
    """Full-vector FDE TE0 of the 500 x 220 nm SOI strip vs. the literature band 2.40-2.50."""
    from photonix import em

    value = float(em.n_eff_fullvector(wl=1.55, width=0.5, thickness=0.22, resolution=40))
    centre, half = 2.45, 0.05
    return value, centre, max(0.0, abs(value - centre) - half)


#: Self-checks in run order. ``quick`` checks are closed-form and take seconds.
CHECKS: tuple[Check, ...] = (
    Check("ring_allpass_closed_form",
          "netlist-solved all-pass ring (coupler + feedback waveguide) equals the closed-form through response "
          "(t - a e^{-j phi}) / (1 - t a e^{-j phi}) over 401 wavelengths (max complex amplitude error)",
          "A. Yariv, Electron. Lett. 36, 321 (2000); W. Bogaerts et al., Laser Photonics Rev. 6, 47 (2012)",
          "2 (analytic)", 1e-12, True, _ring_closed_form),
    Check("mzi_transfer_matrix",
          "netlist-solved 50/50 MZI equals the coupler-arm-coupler transfer-matrix product and conserves power "
          "(max complex amplitude error)",
          "transfer-matrix composition of photonix.components.directional_coupler and straight",
          "2 (analytic)", 1e-12, True, _mzi_closed_form),
    Check("bragg_peak_reflectance",
          "uniform Bragg grating peak reflectance equals tanh^2(kappa L) at the Bragg wavelength (relative error)",
          "H. Kogelnik and C. V. Shank, J. Appl. Phys. 43, 2327 (1972); T. Erdogan, J. Lightwave Technol. 15, "
          "1277 (1997)",
          "2 (analytic)", 1e-6, True, _bragg_peak),
    Check("slab_te0_analytic",
          "finite-difference slab TE0 n_eff (resolution 80/um, Richardson) vs. the exact dispersion relation "
          "(relative error)",
          "symmetric-slab eigenvalue equation, e.g. K. Okamoto, Fundamentals of Optical Waveguides, 2nd ed., "
          "ch. 2 (2006)",
          "2 (analytic)", 1e-3, True, _slab("te")),
    Check("slab_tm0_analytic",
          "finite-difference slab TM0 n_eff (resolution 80/um, Richardson) vs. the exact dispersion relation "
          "(relative error)",
          "symmetric-slab eigenvalue equation, e.g. K. Okamoto, Fundamentals of Optical Waveguides, 2nd ed., "
          "ch. 2 (2006)",
          "2 (analytic)", 1e-3, True, _slab("tm")),
    Check("models_passive_reciprocal",
          "every photonix.components.MODELS entry at default parameters is passive (largest singular value <= 1) "
          "and reciprocal at 5 wavelengths (worst excess of either)",
          "power-normalized S-matrix contraction and symmetry (photonix.analyze_sparameters)",
          "1 (physicality)", 1e-9, True, _passivity_all_models),
    Check("fullvector_soi_strip_te0",
          "full-vector FDE TE0 n_eff of a 500 x 220 nm Si/SiO2 strip at 1.55 um (resolution 40/um) lies in the "
          "published band 2.45 +/- 0.05 (distance outside the band)",
          "L. Chrostowski and M. Hochberg, Silicon Photonics Design, Cambridge Univ. Press (2015), ch. 3 "
          "(500 x 220 nm strip TE0 n_eff about 2.44 at 1550 nm)",
          "4 (published band; coarse)", 0.0, False, _fullvector_soi),
)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("validate", help="run analytic self-checks and print a reproducibility manifest",
                       description="Run packaged self-checks of the installed photonix against closed-form and "
                                   "published references, and emit them together with photonix info as a "
                                   "reproducibility manifest. Exit 1 if any check fails.")
    p.add_argument("--quick", action="store_true", help="only the closed-form checks (a few seconds)")
    p.add_argument("--check", action="append", default=None, metavar="NAME", choices=[c.name for c in CHECKS],
                   help="run only this check; repeatable")
    add_json_flag(p)
    p.set_defaults(handler=run, render=render)


def _run_one(check: Check) -> dict[str, Any]:
    start = time.perf_counter()
    entry: dict[str, Any] = {"name": check.name, "description": check.description, "reference": check.reference,
                             "evidence": check.evidence, "tolerance": check.tolerance, "value": None,
                             "expected": None, "error": None, "passed": False, "exception": None}
    try:
        value, expected, error = check.compute()
        entry.update(value=float(value), expected=float(expected), error=float(error),
                     passed=bool(math.isfinite(error) and error <= check.tolerance))
    except Exception as exc:  # a crashing check is a failed check, never a crashed manifest
        entry["exception"] = f"{type(exc).__name__}: {exc}"
    entry["seconds"] = round(time.perf_counter() - start, 4)
    return entry


def run(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    from photonix.diagnostics import runtime_info

    selected = [c for c in CHECKS if (c.name in args.check if args.check else (c.quick or not args.quick))]
    start = time.perf_counter()
    results = [_run_one(c) for c in selected]
    passed = all(r["passed"] for r in results)
    payload = {
        "schema": SCHEMA,
        "quick": bool(args.quick),
        "passed": passed,
        "n_checks": len(results),
        "n_failed": sum(not r["passed"] for r in results),
        "elapsed_s": round(time.perf_counter() - start, 3),
        "checks": results,
        "runtime": runtime_info().as_dict(),
    }
    return payload, ExitCode.OK if passed else ExitCode.CHECK_FAILED


def render(payload: dict[str, Any]) -> str:
    rt = payload["runtime"]
    lines = [f"photonix {rt['photonix_version']} self-checks ({rt['backend']} backend, {rt['default_real_dtype']}, "
             f"Python {rt['python_version']})"]
    width = max((len(r["name"]) for r in payload["checks"]), default=4)
    for r in payload["checks"]:
        status = "pass" if r["passed"] else "FAIL"
        if r["exception"]:
            detail = r["exception"]
        else:
            relation = "<=" if r["passed"] else ">"
            detail = (f"value {r['value']:.10g} vs {r['expected']:.10g}: error {r['error']:.3e} {relation} "
                      f"{r['tolerance']:.1e}")
        lines.append(f"  {status}  {r['name']:<{width}}  {detail}  ({r['seconds']:.2f} s)")
    lines.append(f"{payload['n_checks'] - payload['n_failed']}/{payload['n_checks']} passed in "
                 f"{payload['elapsed_s']:.1f} s" + ("" if payload["passed"] else " -- FAILED"))
    return "\n".join(lines)
