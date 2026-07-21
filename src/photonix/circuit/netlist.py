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
        # Normalize instances that carry inline {"model": ..., "settings": ...}.
        norm_instances: dict[str, str] = {}
        for inst, spec in self.instances.items():
            if isinstance(spec, Mapping):
                model_name = spec.get("model")
                if not isinstance(model_name, str):
                    raise TypeError(
                        f"Instance {inst!r} dict-spec needs a string 'model', "
                        f"got {spec!r}."
                    )
                norm_instances[inst] = model_name
                inline = spec.get("settings")
                if inline:
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
        self.ports = {
            str(name): _as_instance_port(ip) for name, ip in self.ports.items()
        }

    # -- builder-style mutators --------------------------------------------- #
    def add(self, name: str, model: str, **settings: Any) -> Netlist:
        """Add an instance ``name`` of model ``model`` with optional overrides.

        Returns ``self`` so calls can be chained.
        """
        if name in self.instances:
            raise ValueError(f"Instance {name!r} already exists.")
        self.instances[name] = model
        if settings:
            self.settings[name] = {**self.settings.get(name, {}), **settings}
        return self

    def connect(self, a: InstancePort, b: InstancePort) -> Netlist:
        """Connect internal port ``a`` to internal port ``b``."""
        a = _as_instance_port(a)
        b = _as_instance_port(b)
        self.connections[a] = b
        return self

    def expose(self, external: PortName, internal: InstancePort) -> Netlist:
        """Expose ``internal`` ``(instance, port)`` as circuit port ``external``."""
        self.ports[str(external)] = _as_instance_port(internal)
        return self

    def set(self, instance: str, **settings: Any) -> Netlist:
        """Set per-instance parameter overrides for ``instance``."""
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
        """Raise ``ValueError`` if the netlist references unknown instances."""
        known = set(self.instances)
        for a, b in self.connections.items():
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

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        nm = f" {self.name!r}" if self.name else ""
        return (
            f"<Netlist{nm}: {len(self.instances)} instances, "
            f"{len(self.connections)} connections, {len(self.ports)} ports>"
        )


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
