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
from typing import Any, Protocol, cast, runtime_checkable

Float = Any        # backend float scalar or array
Complex = Any      # backend complex scalar or array
Array = Any        # any backend array

PortName = str
PortPair = tuple[PortName, PortName]

SDict = dict[PortPair, Complex]
"""Sparse scattering dictionary: ``(in_port, out_port) -> complex coefficient``."""


class AliasedSDict(dict):
    """An :data:`SDict` that also accepts *legacy* port names on lookup.

    photonix names optical ports ``o1, o2, ... oN`` everywhere (see
    ``BUILD_SPEC.md``). A few early models used semantic names instead
    (``in0``/``out0`` on the MZI, ``i1``/``t1``/``d2`` on the add-drop ring).
    Those names are kept working as **read-only aliases**: the mapping still
    *stores* only canonical keys, so :func:`ports_of`, the circuit solver and
    every passivity/reciprocity check see one port per physical terminal, but
    ``s[("in0", "out0")]`` resolves to ``s[("o1", "o2")]``.

    Storing the legacy pairs as real entries instead would double-count
    terminals and silently corrupt any circuit built from the model, which is
    why lookup-time resolution is used.

    Examples
    --------
    >>> s = AliasedSDict({("o1", "o2"): 1.0}, aliases={"in0": "o1", "out0": "o2"})
    >>> sorted(s)                      # canonical keys only
    [('o1', 'o2')]
    >>> s[("in0", "out0")]             # legacy lookup still works
    1.0
    >>> ("in0", "out0") in s
    True
    """

    __slots__ = ("aliases",)

    def __init__(self, mapping=(), *, aliases: Mapping[str, str] | None = None):
        super().__init__(mapping)
        #: legacy port name -> canonical port name
        self.aliases: dict[str, str] = dict(aliases or {})

    def _canonical(self, key):
        if not (isinstance(key, tuple) and len(key) == 2):
            return key
        a, b = key
        return (self.aliases.get(a, a), self.aliases.get(b, b))

    def __missing__(self, key):
        canon = self._canonical(key)
        if canon != key and dict.__contains__(self, canon):
            return dict.__getitem__(self, canon)
        raise KeyError(key)

    def __contains__(self, key):
        return dict.__contains__(self, key) or dict.__contains__(self, self._canonical(key))

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def copy(self) -> AliasedSDict:
        return AliasedSDict(self, aliases=self.aliases)

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
    if is_sdense(x):
        port_map = cast(SDense, x)[1]
        return sorted(port_map, key=port_map.__getitem__)
    if is_scoo(x):
        port_map = cast(SCoo, x)[3]
        return sorted(port_map, key=port_map.__getitem__)
    raise TypeError(f"Unrecognized scattering type: {type(x)!r}")


__all__ = [
    "Float", "Complex", "Array", "PortName", "PortPair",
    "SDict", "AliasedSDict", "PortMap", "SDense", "SCoo", "SType", "Settings",
    "Model", "ModelFactory",
    "is_sdict", "is_sdense", "is_scoo", "ports_of",
]
