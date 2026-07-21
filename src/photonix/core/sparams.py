"""Utilities for manipulating scattering objects (SDict / SDense / SCoo).

These helpers are the lingua franca between modules: component models emit
``SDict``s, the circuit solver consumes/produces them, and visualization reads
them. Everything here is differentiable (pure backend arithmetic) and works under
both the JAX and NumPy backends.
"""
from __future__ import annotations

from collections.abc import Iterable

from .backend import xp
from .types import (
    PortMap,
    PortName,
    SDense,
    SDict,
    SType,
    is_scoo,
    is_sdense,
    is_sdict,
    ports_of,
)

__all__ = [
    "sdict_to_sdense",
    "sdense_to_sdict",
    "as_sdict",
    "as_sdense",
    "reciprocal",
    "is_reciprocal",
    "is_passive",
    "power",
    "insertion_loss_db",
    "phase",
    "validate_sdict",
]


def _broadcast_shape(values: Iterable) -> tuple[int, ...]:
    shape: tuple[int, ...] = ()
    for v in values:
        arr = xp.asarray(v)
        shape = xp.broadcast_shapes(shape, arr.shape)
    return shape


def sdict_to_sdense(sdict: SDict, ports: list[PortName] | None = None) -> SDense:
    """Convert a sparse :data:`SDict` to a dense ``(S, port_map)`` pair.

    The batch/wavelength dimensions are broadcast and placed *first*, so ``S`` has
    shape ``(..., n_ports, n_ports)``.
    """
    if ports is None:
        ports = ports_of(sdict)
    port_map: PortMap = {p: i for i, p in enumerate(ports)}
    n = len(ports)

    batch = _broadcast_shape(sdict.values()) if sdict else ()
    S = xp.zeros((*batch, n, n), dtype=complex)
    for (pi, pj), val in sdict.items():
        i, j = port_map[pi], port_map[pj]
        v = xp.broadcast_to(xp.asarray(val, dtype=complex), batch) if batch else xp.asarray(val, dtype=complex)
        S = _set(S, i, j, v)
    return S, port_map


def _set(S, i: int, j: int, value):
    """Backend-agnostic ``S[..., i, j] = value`` returning a new array."""
    if hasattr(S, "at"):  # JAX immutable update
        return S.at[..., i, j].set(value)
    S = S.copy()
    S[..., i, j] = value
    return S


def sdense_to_sdict(sdense: SDense, drop_zeros: bool = True, tol: float = 0.0) -> SDict:
    """Convert a dense ``(S, port_map)`` back to an :data:`SDict`."""
    S, port_map = sdense
    inv = {idx: name for name, idx in port_map.items()}
    n = len(port_map)
    out: SDict = {}
    for i in range(n):
        for j in range(n):
            val = S[..., i, j]
            if drop_zeros:
                mag = float(xp.max(xp.abs(val)))
                if mag <= tol:
                    continue
            out[(inv[i], inv[j])] = val
    return out


def as_sdict(x: SType) -> SDict:
    """Coerce any scattering form to an :data:`SDict`."""
    if is_sdict(x):
        return dict(x)
    if is_sdense(x):
        return sdense_to_sdict(x)
    if is_scoo(x):
        Si, Sj, Sx, pm = x
        inv = {idx: name for name, idx in pm.items()}
        out: SDict = {}
        for a, b, v in zip(list(Si), list(Sj), list(Sx), strict=False):
            out[(inv[int(a)], inv[int(b)])] = v
        return out
    raise TypeError(f"Cannot coerce {type(x)!r} to SDict")


def as_sdense(x: SType, ports: list[PortName] | None = None) -> SDense:
    """Coerce any scattering form to dense ``(S, port_map)``."""
    if is_sdense(x):
        return x
    return sdict_to_sdense(as_sdict(x), ports)


def reciprocal(sdict: SDict) -> SDict:
    """Return a reciprocal version of ``sdict`` by symmetrizing entries."""
    out: SDict = dict(sdict)
    for (i, j), v in sdict.items():
        if (j, i) in out:
            out[(j, i)] = 0.5 * (out[(j, i)] + v)
        else:
            out[(j, i)] = v
    return out


def is_reciprocal(sdict: SDict, atol: float = 1e-6) -> bool:
    """Check S_ij == S_ji within ``atol`` for all listed couplings."""
    for (i, j), v in sdict.items():
        vt = sdict.get((j, i))
        if vt is None:
            return False
        if float(xp.max(xp.abs(xp.asarray(v) - xp.asarray(vt)))) > atol:
            return False
    return True


def is_passive(x: SType, atol: float = 1e-6) -> bool:
    """Check passivity: largest singular value of S is <= 1 (no power gain)."""
    S, _ = as_sdense(x)
    S2 = xp.reshape(S, (-1, S.shape[-2], S.shape[-1]))
    for k in range(S2.shape[0]):
        sv = xp.linalg.svd(S2[k], compute_uv=False)
        if float(xp.max(sv)) > 1.0 + atol:
            return False
    return True


def power(coeff) -> float | object:
    """|coeff|^2 -- transmitted/reflected power fraction."""
    return xp.abs(xp.asarray(coeff)) ** 2


def insertion_loss_db(coeff):
    """Insertion loss in dB: ``-10*log10(|coeff|^2)`` (positive = loss)."""
    return -10.0 * xp.log10(power(coeff))


def phase(coeff):
    """Phase of a coefficient in radians."""
    return xp.angle(xp.asarray(coeff))


def validate_sdict(sdict: SDict) -> None:
    """Raise ``ValueError`` if ``sdict`` is structurally malformed."""
    if not is_sdict(sdict):
        raise ValueError("Object is not a valid SDict (expected {(str, str): value}).")
    for key in sdict:
        if not (isinstance(key, tuple) and len(key) == 2):
            raise ValueError(f"SDict key {key!r} must be a (in_port, out_port) tuple.")
        a, b = key
        if not (isinstance(a, str) and isinstance(b, str)):
            raise ValueError(f"SDict port names must be strings, got {key!r}.")
