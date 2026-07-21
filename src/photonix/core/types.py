"""Core type system for photonix scattering models.

photonix represents the linear optical behaviour of any component or circuit as a
**scattering dictionary** (``SDict``): a mapping from an ordered pair of port
names ``(in_port, out_port)`` to a complex field-amplitude coefficient (possibly
an array over wavelength/parameter sweeps).

A **model** is any callable ``f(*, wl=..., **params) -> SType`` returning a
scattering object. ``SType`` is the union of three interchangeable forms:

``SDict``
    ``{(p_i, p_j): coeff}`` -- human-friendly, sparse.
``SDense``
    ``(S, port_map)`` where ``S`` is an ``(..., n, n)`` array and ``port_map``
    maps port name -> index. Efficient for the solver.
``SCoo``
    ``(Si, Sj, Sx, port_map)`` -- coordinate/sparse form for very large circuits.

Conversion helpers live in :mod:`photonix.core.sparams`.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

Float = Any        # backend float scalar or array
Complex = Any      # backend complex scalar or array
Array = Any        # any backend array

PortName = str
PortPair = tuple[PortName, PortName]

SDict = dict[PortPair, Complex]
"""Sparse scattering dictionary: ``(in_port, out_port) -> complex coefficient``."""

PortMap = dict[PortName, int]
"""Maps a port name to its row/column index in a dense S-matrix."""

SDense = tuple[Array, PortMap]
"""Dense form: ``(S, port_map)`` with ``S`` shape ``(..., n_ports, n_ports)``."""

SCoo = tuple[Array, Array, Array, PortMap]
"""Coordinate sparse form: ``(Si, Sj, Sx, port_map)``."""

SType = SDict | SDense | SCoo
"""Any accepted scattering representation."""

Settings = dict[str, Any]


@runtime_checkable
class Model(Protocol):
    """Protocol for a photonix component model.

    A model is a callable returning an :data:`SType`. By convention it accepts a
    keyword ``wl`` (wavelength in um, scalar or array) plus any number of
    component parameters as keywords, all with sensible defaults so that
    ``model()`` is always valid.
    """

    def __call__(self, **settings: Any) -> SType: ...


ModelFactory = Callable[..., Model]
"""A callable that returns a :class:`Model` (e.g. a parametrized model builder)."""


def is_sdict(x: Any) -> bool:
    """True if ``x`` looks like an :data:`SDict`."""
    if not isinstance(x, Mapping):
        return False
    if len(x) == 0:
        return True
    k = next(iter(x))
    return isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], str)


def is_sdense(x: Any) -> bool:
    """True if ``x`` looks like an :data:`SDense` ``(S, port_map)``."""
    return (
        isinstance(x, tuple)
        and len(x) == 2
        and hasattr(x[0], "shape")
        and isinstance(x[1], Mapping)
    )


def is_scoo(x: Any) -> bool:
    """True if ``x`` looks like an :data:`SCoo` ``(Si, Sj, Sx, port_map)``."""
    return (
        isinstance(x, tuple)
        and len(x) == 4
        and isinstance(x[3], Mapping)
    )


def ports_of(x: SType) -> list[PortName]:
    """Return the sorted list of port names referenced by any scattering form."""
    if is_sdict(x):
        names: set[str] = set()
        for i, j in x:
            names.add(i)
            names.add(j)
        return sorted(names)
    if is_sdense(x) or is_scoo(x):
        return sorted(x[-1], key=lambda n: x[-1][n])
    raise TypeError(f"Unrecognized scattering type: {type(x)!r}")


__all__ = [
    "Float", "Complex", "Array", "PortName", "PortPair",
    "SDict", "PortMap", "SDense", "SCoo", "SType", "Settings",
    "Model", "ModelFactory",
    "is_sdict", "is_sdense", "is_scoo", "ports_of",
]
