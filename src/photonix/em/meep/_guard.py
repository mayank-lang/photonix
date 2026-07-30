"""Optional-dependency guard and unit helpers for the Meep backend.

Meep (and its mode solver MPB) are heavy, conda-installed C++/Python packages and
are therefore an *optional* dependency of photonix. Nothing in this subpackage may
import :mod:`meep` at module import time -- every entry point routes through
:func:`require_meep` / :func:`require_mpb` so that ``import photonix.em`` stays
clean on a stock ``pip install`` with no Meep present.

Unit convention bridge
----------------------
photonix works in **micrometres** for length and wavelength (see
:mod:`photonix.core.units`). Meep is unit-agnostic and normalises to a chosen
length scale ``a``; we always pick ``a = 1 um``. With that choice Meep's
dimensionless frequency is ``f = a / lambda = 1 / lambda_um`` and a guided mode's
effective index is ``n_eff = k / f`` where ``k`` is the propagation constant Meep
reports in units of ``2*pi/a``. :func:`meep_frequency` and :func:`n_eff_from_k`
are the single source of truth for those two conversions.
"""
from __future__ import annotations

import importlib.util
import math

__all__ = [
    "HAS_MEEP",
    "HAS_MPB",
    "require_meep",
    "require_mpb",
    "meep_frequency",
    "n_eff_from_k",
    "k_from_n_eff",
]

# Cheap, import-free probes so callers (and tests) can branch without paying the
# cost of importing Meep or catching ImportError.
try:
    HAS_MEEP = importlib.util.find_spec("meep") is not None
except (ImportError, ModuleNotFoundError, ValueError):
    HAS_MEEP = False
try:
    HAS_MPB = importlib.util.find_spec("meep.mpb") is not None if HAS_MEEP else False
except (ImportError, ModuleNotFoundError, ValueError):
    HAS_MPB = False

_INSTALL_HINT = (
    "Meep is not installed in this environment. The photonix Meep backend "
    "(photonix.em.meep) is an optional extension; install it via conda, e.g. "
    "`conda install -c conda-forge pymeep` (or `pymeep=*=mpi_mpich_*` for the "
    "parallel build), then re-run."
)


def require_meep():
    """Import and return the :mod:`meep` module, or raise a helpful ImportError."""
    try:
        import meep as mp
    except Exception as e:  # noqa: BLE001 - re-raised as ImportError with guidance
        raise ImportError(_INSTALL_HINT) from e
    return mp


def require_mpb():
    """Return ``(meep, meep.mpb)`` or raise a helpful ImportError.

    MPB ships inside the Meep conda package as ``meep.mpb``; a from-source Meep
    build without MPB will import :mod:`meep` but not ``meep.mpb``.
    """
    mp = require_meep()
    try:
        from meep import mpb
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            _INSTALL_HINT + " (this build of Meep is missing the bundled MPB "
            "mode solver `meep.mpb`)."
        ) from e
    return mp, mpb


def meep_frequency(wl_um: float) -> float:
    """Vacuum wavelength [um] -> Meep dimensionless frequency (``a = 1 um``)."""
    wl = float(wl_um)
    if not math.isfinite(wl) or wl <= 0:
        raise ValueError("wl_um must be positive and finite")
    return 1.0 / wl


def n_eff_from_k(k: float, freq: float) -> float:
    """Meep propagation constant ``k`` [2*pi/a] and frequency -> effective index."""
    frequency = float(freq)
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("freq must be positive and finite")
    return float(k) / frequency


def k_from_n_eff(n_eff: float, freq: float) -> float:
    """Effective index and Meep frequency -> propagation constant ``k`` [2*pi/a]."""
    effective_index = float(n_eff)
    frequency = float(freq)
    if not math.isfinite(effective_index):
        raise ValueError("n_eff must be finite")
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("freq must be positive and finite")
    return effective_index * frequency
