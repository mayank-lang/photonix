"""Tidy3D adapter (stub).

Mirrors the benchmark cases in Flexcompute Tidy3D (cloud FDTD + local mode
solver). Stub: raises ``ImportError`` unless ``tidy3d`` is installed. Note Tidy3D
FDTD runs are cloud jobs (credentials + credits required); the local
``ModeSolver`` can do the cross-section n_eff cases offline.

Contract: ``run_all()`` returns ``{case_key: float}``; keys match
``benchmarks/cases.py``.
"""
from __future__ import annotations


def _require_tidy3d():
    try:
        import tidy3d  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "tidy3d is not installed. `pip install tidy3d` and configure API "
            "credentials to enable this adapter (mode solves are local; FDTD is cloud)."
        ) from e


def neff_soi_strip_te0() -> float:
    """500x220 nm Si strip TE0 via the Tidy3D local ModeSolver."""
    _require_tidy3d()
    # TODO: build Simulation cross-section + ModeSolver, return mode_data n_eff.
    raise NotImplementedError("Tidy3D ModeSolver cross-section: fill in.")


def run_all() -> dict[str, float]:
    _require_tidy3d()
    out: dict[str, float] = {}
    for key, fn in {"soi_strip_te0_neff": neff_soi_strip_te0}.items():
        try:
            out[key] = float(fn())
        except NotImplementedError:
            pass
    return out
