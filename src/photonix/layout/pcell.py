"""Parametric cells with deterministic, settings-addressable names and caching.

:func:`pcell` decorates a layout generator ``f(**settings) -> Cell``:

* **Deterministic names.** The returned cell is named
  ``f"{name}_{sha1(canonical_json(settings))[:8]}"``, where ``settings`` are
  all the generator's arguments with defaults applied. Canonical JSON sorts
  keys, rounds floats to 1e-9 (µm, i.e. 1 fm grid noise is ignored) and
  writes them with :func:`repr`, turns tuples into lists, NumPy scalars and
  arrays into numbers and lists, callables into their qualified names and
  cells into their names. The hash therefore depends on the settings only --
  not on Python's per-process ``hash()`` -- and is identical across sessions
  and machines.
* **Caching.** An LRU cache keyed by the canonical string returns the *same*
  :class:`~photonix.layout.Cell` object for identical calls, so GDS export
  writes one structure per distinct setting and names never need ``$n``
  suffixes. Do not mutate a cached cell in place; call the generator with
  other settings instead (or ``fn.cache_clear()``).
* **Identity.** The cell records ``{"component": name, "settings": {...}}``
  (read it with :func:`pcell_info`), which
  :func:`~photonix.layout.schematic_from_layout` uses to write a schematic
  back out.

A renamed leaf cell without a ``model`` keeps simulating as before: its
original name becomes its ``model``.

The approach mirrors gdsfactory's ``@gf.cell`` and KLayout PCells.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, cast

import numpy as np

from .cell import Cell

__all__ = ["pcell", "pcell_info", "canonical_settings", "settings_hash"]

F = TypeVar("F", bound=Callable[..., Cell])
_INFO_ATTR = "pcell_settings"


def _canonical(value: Any) -> Any:
    if isinstance(value, Cell):
        return {"cell": value.name}
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not math.isfinite(v):
            return repr(v)
        v = round(v, 9) + 0.0          # 1e-9 grid; -0.0 -> 0.0
        return int(v) if v.is_integer() and abs(v) < 2**53 else v
    if isinstance(value, complex):
        return {"complex": [_canonical(value.real), _canonical(value.imag)]}
    if isinstance(value, np.ndarray):
        return [_canonical(v) for v in value.tolist()] if value.ndim else _canonical(value.item())
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if callable(value):
        mod = getattr(value, "__module__", "") or ""
        qual = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
        return {"callable": f"{mod}.{qual}"}
    raise TypeError(f"cannot canonicalize PCell setting of type {type(value).__name__}: {value!r}")


def canonical_settings(settings: Mapping[str, Any]) -> str:
    """Canonical JSON text of a settings mapping (module docs).

    Examples
    --------
    >>> canonical_settings({"width": 0.5, "length": 10, "layer": (1, 0)})
    '{"layer":[1,0],"length":10,"width":0.5}'
    >>> canonical_settings({"x": 0.1 + 0.2}) == canonical_settings({"x": 0.3})
    True
    """
    return json.dumps(_canonical(dict(settings)), sort_keys=True, separators=(",", ":"), allow_nan=False)


def settings_hash(settings: Mapping[str, Any], n: int = 8) -> str:
    """First ``n`` hex digits of the SHA-1 of :func:`canonical_settings`.

    Examples
    --------
    >>> settings_hash({"length": 10.0}) == settings_hash({"length": 10})
    True
    >>> len(settings_hash({"length": 10.0}))
    8
    """
    return hashlib.sha1(canonical_settings(settings).encode("utf-8")).hexdigest()[:n]


def pcell_info(cell: Cell) -> dict[str, Any] | None:
    """The ``{"component": ..., "settings": ...}`` record of a PCell-made cell, or ``None``.

    Examples
    --------
    >>> from photonix.layout import components as lc
    >>> straight = pcell(lc.straight)
    >>> pcell_info(straight(length=5.0))["settings"]["length"]
    5
    """
    info = getattr(cell, _INFO_ATTR, None)
    return dict(info) if isinstance(info, Mapping) else None


def pcell(fn: F | None = None, *, name: str | None = None, cache_size: int = 1024) -> Any:
    """Decorate a layout generator as a parametric cell (module docs).

    Parameters
    ----------
    fn
        The generator. ``@pcell`` and ``@pcell(name=..., cache_size=...)``
        both work.
    name
        Component name used in cell names and schematic records (default:
        ``fn.__name__``).
    cache_size
        Maximum number of cached cells (least recently used are dropped);
        ``0`` disables caching (names stay deterministic).

    Returns
    -------
    callable
        The wrapped generator, with ``cache_clear()``, ``cache_info()`` and
        ``component_name`` attributes.

    Examples
    --------
    >>> from photonix.layout import components as lc
    >>> straight = pcell(lc.straight)
    >>> a, b = straight(length=10.0), straight(10.0)
    >>> a is b, a.name == f"straight_{settings_hash({'length': 10.0, 'width': 0.5, 'layer': (1, 0)})}"
    (True, True)
    >>> straight(length=12.0).name != a.name, a.model
    (True, 'straight')
    """
    if fn is None:
        return functools.partial(pcell, name=name, cache_size=cache_size)
    if cache_size < 0:
        raise ValueError("cache_size must be >= 0")
    component = name or fn.__name__
    sig = inspect.signature(fn)
    cache: OrderedDict[str, Cell] = OrderedDict()
    stats = {"hits": 0, "misses": 0}

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Cell:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        settings = dict(bound.arguments)
        for pname, p in sig.parameters.items():       # flatten **kwargs into the settings
            if p.kind is inspect.Parameter.VAR_KEYWORD and pname in settings:
                settings.update(settings.pop(pname))
        key = canonical_settings(settings)
        if cache_size and key in cache:
            stats["hits"] += 1
            cache.move_to_end(key)
            return cache[key]
        stats["misses"] += 1
        cell = fn(*bound.args, **bound.kwargs)
        if not isinstance(cell, Cell):
            raise TypeError(f"PCell {component!r} must return a Cell, got {type(cell).__name__}")
        if cell.model is None and not cell.references:
            cell.model = cell.name                   # keep simulating as the original leaf model
        cell.name = f"{component}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}"
        setattr(cell, _INFO_ATTR, {"component": component, "settings": json.loads(key)})
        if cache_size:
            cache[key] = cell
            if len(cache) > cache_size:
                cache.popitem(last=False)
        return cell

    def cache_clear() -> None:
        cache.clear()
        stats["hits"] = stats["misses"] = 0

    def cache_info() -> dict[str, int]:
        return {"hits": stats["hits"], "misses": stats["misses"], "size": len(cache), "maxsize": cache_size}

    w = cast(Any, wrapper)
    w.cache_clear = cache_clear
    w.cache_info = cache_info
    w.component_name = component
    return wrapper
