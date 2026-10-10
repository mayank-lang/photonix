"""Netlist / circuit data structures for photonix.

A :class:`Netlist` is a declarative description of a photonic circuit:

* **instances** — named placements of a model (by model name) with optional
  per-instance parameter overrides (``settings``);
* **connections** — internal links between two instance ports, written as
  ``("inst1", "o2") <-> ("inst2", "o1")``;
* **ports** — the *external* (circuit-level) ports, each mapping a public name
  to one internal instance port.

The netlist is a pure data structure: it carries no JAX arrays and performs no
math, so it is trivially picklable / hashable-by-content and independent of the
backend. The :mod:`photonix.circuit.solver` consumes it (together with a mapping
of model-name -> callable model) to produce a differentiable composite model.

A :class:`Circuit` is a thin, mutable *builder* around :class:`Netlist` with a
fluent API (``add``/``connect``/``expose``) for ergonomic construction in code.

Examples
--------
>>> nl = Netlist(
...     instances={"wg": "straight", "dc": "coupler"},
...     connections={("dc", "o2"): ("wg", "o1")},
...     ports={"in0": ("dc", "o1"), "out0": ("wg", "o2")},
... )
>>> sorted(nl.external_ports)
['in0', 'out0']
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from photonix.core.types import PortName, Settings

__all__ = [
    "InstancePort",
    "Netlist",
    "Circuit",
]

# An (instance_name, port_name) pair identifying one terminal in the circuit.
InstancePort = tuple[str, PortName]


def _as_instance_port(x: Any) -> InstancePort:
    """Coerce ``x`` to a validated ``(instance, port)`` tuple."""
    if not (isinstance(x, tuple) and len(x) == 2):
        raise TypeError(
            f"Expected an (instance, port) tuple, got {x!r}."
        )
    inst, port = x
    if not (isinstance(inst, str) and isinstance(port, str)):
        raise TypeError(
            f"(instance, port) names must both be strings, got {x!r}."
        )
    return (inst, port)


@dataclass
class Netlist:
    """Immutable-by-convention description of a photonic circuit topology.

    Parameters
    ----------
    instances
        Mapping ``instance_name -> model_name``. A model name is resolved
        against the ``models`` mapping passed to the solver. To attach
        per-instance parameter overrides give a 2-form value
        ``{"model": name, "settings": {...}}`` *or* register them separately in
        :attr:`settings`.
    connections
        Mapping of internal links ``(inst_a, port_a) -> (inst_b, port_b)``.
        Connections are undirected (optical reciprocity is the model's concern);
        each physical link appears once.
    ports
        Mapping of external circuit port name -> internal ``(instance, port)``.
        Only ports listed here survive into the composite ``SDict``.
    settings
        Optional mapping ``instance_name -> {param: value}`` of per-instance
        parameter overrides, merged on top of any inline ``settings`` and below
        call-time overrides.
    name
        Optional human-readable circuit name.

    Notes
    -----
    Instance values may be given either as a bare model-name string or as a dict
    ``{"model": str, "settings": dict}``. After construction the canonical model
    name lives in :attr:`instances` and overrides in :attr:`settings`.
    """

    instances: dict[str, str] = field(default_factory=dict)
    connections: dict[InstancePort, InstancePort] = field(default_factory=dict)
    ports: dict[PortName, InstancePort] = field(default_factory=dict)
    settings: dict[str, Settings] = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError(f"Netlist name must be a string or None, got {self.name!r}.")
        normalized_settings: dict[str, Settings] = {}
        for inst, values in self.settings.items():
            if not isinstance(inst, str) or not isinstance(values, Mapping):
                raise TypeError("settings must map instance-name strings to mappings.")
            normalized_settings[inst] = dict(values)
        self.settings = normalized_settings

        # Normalize instances that carry inline {"model": ..., "settings": ...}.
        norm_instances: dict[str, str] = {}
        for inst, spec in self.instances.items():
            if not isinstance(inst, str):
                raise TypeError(f"Instance names must be strings, got {inst!r}.")
            if isinstance(spec, Mapping):
                model_name = spec.get("model")
                if not isinstance(model_name, str):
                    raise TypeError(
                        f"Instance {inst!r} dict-spec needs a string 'model', "
                        f"got {spec!r}."
                    )
                norm_instances[inst] = model_name
                inline = spec.get("settings")
                if inline is not None:
                    if not isinstance(inline, Mapping):
                        raise TypeError(
                            f"Instance {inst!r} settings must be a mapping, got {inline!r}."
                        )
                    merged = {**dict(inline), **self.settings.get(inst, {})}
                    self.settings[inst] = merged
            elif isinstance(spec, str):
                norm_instances[inst] = spec
            else:
                raise TypeError(
                    f"Instance {inst!r} must map to a model name or a "
                    f"{{'model': ..., 'settings': ...}} dict, got {spec!r}."
                )
        self.instances = norm_instances

        # Normalize connection / port keys to validated tuples.
        self.connections = {
            _as_instance_port(a): _as_instance_port(b)
            for a, b in self.connections.items()
        }
        normalized_ports: dict[PortName, InstancePort] = {}
        for name, ip in self.ports.items():
            if not isinstance(name, str):
                raise TypeError(f"External port names must be strings, got {name!r}.")
            normalized_ports[name] = _as_instance_port(ip)
        self.ports = normalized_ports

    # -- builder-style mutators --------------------------------------------- #
    def add(self, name: str, model: str, **settings: Any) -> Netlist:
        """Add an instance ``name`` of model ``model`` with optional overrides.

        Returns ``self`` so calls can be chained.
        """
        if not isinstance(name, str) or not isinstance(model, str):
            raise TypeError("Instance and model names must be strings.")
        if name in self.instances:
            raise ValueError(f"Instance {name!r} already exists.")
        self.instances[name] = model
        if settings:
            self.settings[name] = {**self.settings.get(name, {}), **settings}
        return self

    def _all_connected_terminals(self) -> set[InstancePort]:
        """Return every terminal that participates in a connection."""
        s: set[InstancePort] = set()
        for a, b in self.connections.items():
            s.add(a)
            s.add(b)
        return s

    def connect(self, a: InstancePort, b: InstancePort) -> Netlist:
        """Connect internal port ``a`` to internal port ``b``.

        Raises immediately if either terminal is already connected or exposed.
        """
        a = _as_instance_port(a)
        b = _as_instance_port(b)
        if a == b:
            raise ValueError(f"A terminal cannot be connected to itself: {a!r}.")
        used = self._all_connected_terminals()
        for terminal in (a, b):
            if terminal in used:
                raise ValueError(
                    f"Terminal {terminal!r} is already in a connection. "
                    f"Each terminal may be connected at most once."
                )
            exposed_ips = set(self.ports.values())
            if terminal in exposed_ips:
                raise ValueError(
                    f"Terminal {terminal!r} is already exposed as an external "
                    f"port. A terminal cannot be both connected and exposed."
                )
        self.connections[a] = b
        return self

    def expose(self, external: PortName, internal: InstancePort) -> Netlist:
        """Expose ``internal`` ``(instance, port)`` as circuit port ``external``.

        Raises immediately if the terminal is already in a connection.
        """
        ip = _as_instance_port(internal)
        if not isinstance(external, str):
            raise TypeError(f"External port names must be strings, got {external!r}.")
        if external in self.ports:
            raise ValueError(f"External port {external!r} is already defined.")
        used = self._all_connected_terminals()
        if ip in used:
            raise ValueError(
                f"Terminal {ip!r} is already in an internal connection. "
                f"A terminal cannot be both connected and exposed."
            )
        if ip in self.ports.values():
            other = next(name for name, terminal in self.ports.items() if terminal == ip)
            raise ValueError(
                f"Terminal {ip!r} is already exposed as {other!r}; use port aliases "
                "instead of exposing one physical terminal twice."
            )
        self.ports[external] = ip
        return self

    def set(self, instance: str, **settings: Any) -> Netlist:
        """Set per-instance parameter overrides for ``instance``."""
        if instance not in self.instances:
            raise KeyError(f"Cannot set parameters for unknown instance {instance!r}.")
        self.settings[instance] = {**self.settings.get(instance, {}), **settings}
        return self

    # -- derived views ------------------------------------------------------ #
    @property
    def external_ports(self) -> list[PortName]:
        """Sorted list of external (circuit-level) port names."""
        return sorted(self.ports)

    def instance_ports(self, instance: str, model_ports: Iterable[PortName]) -> list[str]:
        """Return globally-unique terminal names ``f"{instance},{port}"``.

        Used internally by the solver to namespace per-instance ports.
        """
        return [f"{instance}{TERMINAL_SEP}{p}" for p in model_ports]

    def merged_settings(self, instance: str, overrides: Settings | None = None) -> Settings:
        """Merge static + call-time overrides for ``instance`` (call-time wins)."""
        out = dict(self.settings.get(instance, {}))
        if overrides:
            out.update(overrides)
        return out

    def validate(self) -> None:
        """Raise ``ValueError`` if the netlist is structurally invalid.

        Checks performed:

        1. All instance names referenced in connections/ports exist.
        2. No terminal appears in more than one connection (would create
           energy — see PHYSICS_AUDIT §A4).
        3. No terminal is both internally connected and externally exposed
           (the solver would silently drive it from two sources).
        """
        known = set(self.instances)
        unknown_settings = set(self.settings) - known
        if unknown_settings:
            raise ValueError(
                f"Settings reference unknown instances: {sorted(unknown_settings)!r}."
            )
        for a, b in self.connections.items():
            if a == b:
                raise ValueError(f"A terminal cannot be connected to itself: {a!r}.")
            for inst, _ in (a, b):
                if inst not in known:
                    raise ValueError(
                        f"Connection references unknown instance {inst!r}."
                    )
        for ext, (inst, _) in self.ports.items():
            if inst not in known:
                raise ValueError(
                    f"External port {ext!r} references unknown instance {inst!r}."
                )

        # A4 — no terminal may appear in more than one connection.
        seen: dict[InstancePort, int] = {}
        for idx, (a, b) in enumerate(self.connections.items()):
            for terminal in (a, b):
                if terminal in seen:
                    raise ValueError(
                        f"Terminal {terminal!r} appears in multiple connections "
                        f"(connection {seen[terminal]} and {idx}). Each terminal "
                        f"may be connected at most once."
                    )
                seen[terminal] = idx

        # A4 — an exposed port must not also be internally connected.
        connected = set(seen)
        exposed: dict[InstancePort, str] = {}
        for ext, ip in self.ports.items():
            if ip in connected:
                raise ValueError(
                    f"External port {ext!r} maps to {ip!r}, which is also "
                    f"in an internal connection. A terminal cannot be both "
                    f"connected and exposed."
                )
            if ip in exposed:
                raise ValueError(
                    f"External ports {exposed[ip]!r} and {ext!r} both map to "
                    f"{ip!r}. Each physical terminal may be exposed only once."
                )
            exposed[ip] = ext

    # -- serialization ------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """A JSON-compatible description of the netlist.

        Settings must be finite numbers, strings, booleans, ``None`` or
        (nested) lists/arrays/string-keyed mappings of those; NumPy/JAX scalars
        and arrays are converted and tuples are stored as lists (so they reload
        as lists). The output is strict JSON (no ``NaN``/``Infinity``).
        Callables (e.g. a custom ``neff(wl)``) cannot be serialized and raise
        ``TypeError`` naming the instance and setting.

        Examples
        --------
        >>> nl = Netlist(name="pair")
        >>> _ = nl.add("a", "straight", length=5.0).add("b", "straight")
        >>> _ = nl.connect(("a", "o2"), ("b", "o1"))
        >>> _ = nl.expose("in", ("a", "o1")).expose("out", ("b", "o2"))
        >>> Netlist.from_json(nl.to_json()) == nl
        True
        """
        self.validate()
        instances: dict[str, Any] = {}
        for inst, model in self.instances.items():
            entry: dict[str, Any] = {"model": model}
            if self.settings.get(inst):
                entry["settings"] = {k: _json_value(v, inst, k) for k, v in self.settings[inst].items()}
            instances[inst] = entry
        return {
            "format": "photonix.netlist",
            "version": 1,
            "name": self.name,
            "instances": instances,
            "connections": [[list(a), list(b)] for a, b in self.connections.items()],
            "ports": {ext: list(ip) for ext, ip in self.ports.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Netlist:
        """Inverse of :meth:`to_dict`."""
        if data.get("format", "photonix.netlist") != "photonix.netlist":
            raise ValueError(f"not a photonix netlist: format={data.get('format')!r}")
        if int(data.get("version", 1)) != 1:
            raise ValueError(f"unsupported netlist version {data.get('version')!r}")
        try:
            connections = {tuple(a): tuple(b) for a, b in data.get("connections", [])}
        except (TypeError, ValueError) as exc:
            raise ValueError("connections must be [[inst, port], [inst, port]] pairs") from exc
        nl = cls(
            instances=dict(data.get("instances", {})),
            connections=connections,
            ports={ext: tuple(ip) for ext, ip in dict(data.get("ports", {})).items()},
            name=data.get("name"),
        )
        nl.validate()
        return nl

    def to_json(self, path: str | os.PathLike[str] | None = None, *, indent: int = 2) -> str:
        """Serialize to JSON text; also write it to ``path`` if given."""
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=False, allow_nan=False)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        return text

    @classmethod
    def from_json(cls, source: str | os.PathLike[str]) -> Netlist:
        """Load from JSON text or from a path to a ``.json`` file."""
        text = os.fspath(source)
        if not text.lstrip().startswith("{"):
            with open(text, encoding="utf-8") as fh:
                text = fh.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        nm = f" {self.name!r}" if self.name else ""
        return (
            f"<Netlist{nm}: {len(self.instances)} instances, "
            f"{len(self.connections)} connections, {len(self.ports)} ports>"
        )


def _json_value(value: Any, inst: str, key: str) -> Any:
    """Convert a setting to plain JSON types or raise a precise ``TypeError``."""
    import math

    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError(f"setting {key!r} of instance {inst!r} is not finite ({value}); strict JSON cannot hold it")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(v, inst, key) for v in value]
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise TypeError(f"setting {key!r} of instance {inst!r} has non-string mapping keys; JSON requires strings")
        return {k: _json_value(v, inst, key) for k, v in value.items()}
    if callable(value):
        raise TypeError(f"setting {key!r} of instance {inst!r} is callable and cannot be serialized")
    try:
        import numpy as np

        arr = np.asarray(value)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"setting {key!r} of instance {inst!r} is not JSON-serializable") from exc
    if arr.dtype.kind in "biuf":
        if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
            raise TypeError(f"setting {key!r} of instance {inst!r} contains non-finite values")
        return arr.tolist()
    if arr.dtype.kind == "c":
        raise TypeError(f"setting {key!r} of instance {inst!r} is complex; store real and imaginary parts")
    raise TypeError(f"setting {key!r} of instance {inst!r} is not JSON-serializable")


# Separator used to namespace an instance's port into a global terminal name.
# Chosen to avoid clashing with typical port names ("o1", "in0", ...).
TERMINAL_SEP = ","


class Circuit(Netlist):
    """Mutable builder alias for :class:`Netlist`.

    :class:`Circuit` is :class:`Netlist` with the same fields; it exists so that
    user code can read fluently (``Circuit().add(...).connect(...)``) and so the
    public API exposes both names. Construct an empty one and build it up:

    >>> c = Circuit(name="cascade")
    >>> _ = c.add("a", "twoport").add("b", "twoport")
    >>> _ = c.connect(("a", "o2"), ("b", "o1"))
    >>> _ = c.expose("in0", ("a", "o1")).expose("out0", ("b", "o2"))
    >>> c.external_ports
    ['in0', 'out0']
    """
