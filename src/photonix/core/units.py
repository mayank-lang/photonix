"""Unit helpers and frequency/wavelength conversions.

Internal convention: **lengths and wavelengths in micrometers (um)**, frequencies
in **terahertz (THz)** for public-facing helpers, SI elsewhere. All conversions
are differentiable (pure arithmetic on the active backend).
"""
from __future__ import annotations

from typing import Any

from .backend import xp
from .constants import C0_UM_S

# Length scale factors relative to 1 meter ----------------------------------- #
m = 1.0
mm = 1e-3
um = 1e-6
nm = 1e-9
# Relative-to-um helpers (multiply a value in the given unit to get um)
nm_to_um = 1e-3
mm_to_um = 1e3

__all__ = [
    "m", "mm", "um", "nm", "nm_to_um", "mm_to_um",
    "wl_to_freq", "freq_to_wl", "wl_to_omega", "freq_to_omega",
    "db_to_lin", "lin_to_db", "db_per_cm_to_alpha_um",
]


def wl_to_freq(wl_um: Any) -> Any:
    """Wavelength [um] -> frequency [THz]."""
    return C0_UM_S * 1e-12 / xp.asarray(wl_um)


def freq_to_wl(freq_thz: Any) -> Any:
    """Frequency [THz] -> wavelength [um]."""
    return C0_UM_S * 1e-12 / xp.asarray(freq_thz)


def wl_to_omega(wl_um: Any) -> Any:
    """Wavelength [um] -> angular frequency [rad/s]."""
    return 2.0 * xp.pi * C0_UM_S / xp.asarray(wl_um)


def freq_to_omega(freq_thz: Any) -> Any:
    """Frequency [THz] -> angular frequency [rad/s]."""
    return 2.0 * xp.pi * xp.asarray(freq_thz) * 1e12


def db_to_lin(db: Any) -> Any:
    """Convert power ratio in dB to linear amplitude transmission."""
    return 10.0 ** (xp.asarray(db) / 20.0)


def lin_to_db(amp: Any) -> Any:
    """Convert linear amplitude transmission to power ratio in dB."""
    return 20.0 * xp.log10(xp.abs(xp.asarray(amp)))


def db_per_cm_to_alpha_um(loss_db_per_cm: Any) -> Any:
    """Propagation loss [dB/cm] -> field attenuation coefficient alpha [1/um].

    The field decays as ``exp(-alpha * L)`` with ``L`` in um.
    """
    # dB/cm power loss -> Np/cm (field) -> 1/um
    ln10 = 2.302585092994046
    return xp.asarray(loss_db_per_cm) * ln10 / 20.0 / 1e4
