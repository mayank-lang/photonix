"""Shared input normalization for the macromodel subpackage (private)."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from photonix.core.constants import C0_UM_S
from photonix.core.sparams import sdict_to_sdense
from photonix.core.types import is_sdict

__all__: list[str] = []


def default_ports(n: int) -> tuple[str, ...]:
    """Canonical Photonix optical port names ``o1..oN``."""
    return tuple(f"o{i + 1}" for i in range(n))


def dense_data(
    S: Any, n_samples: int, ports: Sequence[str] | None = None
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return ``(S[:, out, in], ports)`` as complex128 NumPy data.

    ``S`` may be an array of shape ``(n_samples,)`` (one transfer function,
    treated as a one-port), ``(n_samples, n, n)`` in the matrix convention
    ``S[:, out, in]``, or an array-valued SDict with ``(in, out)`` keys.
    """
    if is_sdict(S):
        dense, port_map = sdict_to_sdense(S, None if ports is None else list(ports))
        data = np.asarray(dense, dtype=complex)
        names = tuple(sorted(port_map, key=port_map.__getitem__))
        if data.ndim == 2:
            data = np.broadcast_to(data, (n_samples, *data.shape)).copy()
    else:
        data = np.asarray(S, dtype=complex)
        if data.ndim == 1:
            data = data[:, None, None]
        if data.ndim != 3 or data.shape[1] != data.shape[2]:
            raise ValueError("S must have shape (n_samples,), (n_samples, n, n) or be an SDict")
        names = default_ports(data.shape[1]) if ports is None else tuple(str(p) for p in ports)
    if data.shape[0] != n_samples:
        raise ValueError(f"S has {data.shape[0]} samples but {n_samples} frequencies/wavelengths were given")
    if len(names) != data.shape[1] or len(set(names)) != len(names):
        raise ValueError(f"ports must name the {data.shape[1]} ports uniquely, got {names!r}")
    if not np.all(np.isfinite(data.real)) or not np.all(np.isfinite(data.imag)):
        raise ValueError("S must contain finite complex values")
    return data, names


def frequency_grid(freq_hz: Any, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort samples by increasing frequency and validate the grid."""
    f = np.asarray(freq_hz, dtype=float)
    if f.ndim != 1 or f.size < 2:
        raise ValueError("frequencies must be a one-dimensional array with at least two samples")
    if not np.all(np.isfinite(f)):
        raise ValueError("frequencies must be finite")
    order = np.argsort(f)
    f = f[order]
    if np.any(np.diff(f) <= 0):
        raise ValueError("frequencies (or wavelengths) must be distinct")
    return f, data[order]


def wavelengths_to_hz(wavelengths: Any) -> np.ndarray:
    """Convert positive wavelengths in micrometres to frequencies in Hz."""
    wl = np.asarray(wavelengths, dtype=float)
    if wl.ndim != 1 or not np.all(np.isfinite(wl)) or np.any(wl <= 0):
        raise ValueError("wavelengths must be a one-dimensional array of positive finite values (um)")
    return C0_UM_S / wl
