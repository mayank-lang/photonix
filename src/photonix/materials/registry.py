"""Name -> material registry (case- and punctuation-insensitive, with aliases)."""
from __future__ import annotations

import difflib
import re

from .model import MaterialSpec

__all__ = ["register", "get_material", "list_materials", "unregister"]

_REGISTRY: dict[str, MaterialSpec] = {}
_ALIASES: dict[str, str] = {}      # normalised alias -> normalised canonical key


def _norm(name: str) -> str:
    return re.sub(r"[^0-9a-z]", "", str(name).lower())


def register(spec: MaterialSpec, *, aliases: tuple[str, ...] | list[str] = (), overwrite: bool = False) -> MaterialSpec:
    """Add ``spec`` to the registry under ``spec.name`` and any ``aliases``.

    Lookup ignores case and punctuation (``"SU-8"``, ``"su8"`` and ``"Su_8"`` are
    the same key). Re-registering an existing name raises unless ``overwrite``.

    Examples
    --------
    >>> from photonix.materials import MaterialSpec, constant_model, register, get_material, unregister
    >>> _ = register(MaterialSpec("MyPolymer", constant_model(1.49)), aliases=["my-poly"])
    >>> get_material("my_poly").name
    'MyPolymer'
    >>> unregister("MyPolymer")
    """
    if not isinstance(spec, MaterialSpec):
        raise TypeError("register() expects a MaterialSpec")
    key = _norm(spec.name)
    names = [key] + [_norm(a) for a in aliases]
    if not overwrite:
        for n in names:
            if n in _REGISTRY or n in _ALIASES:
                raise ValueError(f"material name {n!r} is already registered (pass overwrite=True)")
    _REGISTRY[key] = spec
    for a in names[1:]:
        if a != key:
            _ALIASES[a] = key
    return spec


def unregister(name: str) -> None:
    """Remove a material and its aliases (mainly for tests and user overrides)."""
    key = _ALIASES.get(_norm(name), _norm(name))
    _REGISTRY.pop(key, None)
    for a in [a for a, k in _ALIASES.items() if k == key]:
        del _ALIASES[a]


def get_material(name: str | MaterialSpec) -> MaterialSpec:
    """Look up a registered material by name or alias.

    A :class:`MaterialSpec` passes through unchanged, so APIs can accept either.

    Examples
    --------
    >>> from photonix.materials import get_material
    >>> get_material("silicon nitride").name
    'Si3N4'
    >>> get_material("sapphire") is get_material("Al2O3")
    True
    """
    if isinstance(name, MaterialSpec):
        return name
    key = _norm(name)
    key = _ALIASES.get(key, key)
    try:
        return _REGISTRY[key]
    except KeyError:
        known = sorted({s.name for s in _REGISTRY.values()})
        close = difflib.get_close_matches(str(name), known, n=3, cutoff=0.5)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise KeyError(f"unknown material {name!r}.{hint} Known: {', '.join(known)}") from None


def list_materials(tag: str | None = None) -> list[str]:
    """Sorted canonical names of the registered materials, optionally filtered by tag.

    Examples
    --------
    >>> from photonix.materials import list_materials
    >>> {"Si", "SiO2", "Si3N4", "LiNbO3", "TiN"} <= set(list_materials())
    True
    >>> "Cu" in list_materials(tag="metal")
    True
    """
    specs = list(_REGISTRY.values())
    if tag is not None:
        specs = [s for s in specs if tag in s.tags]
    return sorted(s.name for s in specs)
