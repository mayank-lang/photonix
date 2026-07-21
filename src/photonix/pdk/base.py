"""PDK-agnostic process design kit interface.

A :class:`Pdk` is a registry that ties a component *name* to (1) a layout
generator producing a :class:`~photonix.layout.Cell`, (2) an optional circuit
*model* producing an ``SDict``, and (3) default settings for each (kept separate
because layout params like ``width`` differ from model params like ``neff``). It
also carries a layer map. PDKs let one design target different foundries by
swapping the kit.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = ["Layer", "ComponentSpec", "Pdk"]


@dataclass(frozen=True)
class Layer:
    """A GDS layer definition."""

    name: str
    layer: int
    datatype: int = 0

    @property
    def tuple(self) -> tuple[int, int]:
        return (self.layer, self.datatype)


@dataclass
class ComponentSpec:
    """Registration record for one PDK component."""

    name: str
    layout: Callable | None = None
    model: Callable | None = None
    layout_settings: dict = field(default_factory=dict)
    model_settings: dict = field(default_factory=dict)


class Pdk:
    """A process design kit: named components + layer map.

    Examples
    --------
    >>> from photonix.pdk import Pdk, Layer
    >>> import photonix.layout.components as lc
    >>> pdk = Pdk("demo")
    >>> _ = pdk.add_layer(Layer("WG", 1, 0))
    >>> _ = pdk.add_component("straight", layout=lc.straight, layout_settings={"width": 0.5})
    >>> cell = pdk.get_layout("straight", length=5.0)
    >>> len(cell.ports)
    2
    """

    def __init__(self, name: str):
        self.name = name
        self.components: dict[str, ComponentSpec] = {}
        self.layers: dict[str, Layer] = {}

    def add_layer(self, layer: Layer) -> Pdk:
        self.layers[layer.name] = layer
        return self

    def add_component(self, name: str, *, layout=None, model=None, layout_settings=None, model_settings=None) -> Pdk:
        self.components[name] = ComponentSpec(
            name, layout, model, dict(layout_settings or {}), dict(model_settings or {})
        )
        return self

    def get_layout(self, name: str, **overrides):
        """Instantiate the layout :class:`Cell` for component ``name``."""
        spec = self._spec(name)
        if spec.layout is None:
            raise ValueError(f"Component {name!r} has no layout generator.")
        return spec.layout(**{**spec.layout_settings, **overrides})

    def get_model(self, name: str):
        """Return the circuit model callable for component ``name`` (or None)."""
        return self._spec(name).model

    def model_settings(self, name: str) -> dict:
        """Default model settings for component ``name``."""
        return dict(self._spec(name).model_settings)

    def evaluate(self, name: str, *, wl=1.55, **overrides):
        """Evaluate the circuit model of ``name`` with PDK defaults + overrides."""
        spec = self._spec(name)
        if spec.model is None:
            raise ValueError(f"Component {name!r} has no circuit model.")
        return spec.model(wl=wl, **{**spec.model_settings, **overrides})

    def _spec(self, name: str) -> ComponentSpec:
        if name not in self.components:
            raise KeyError(f"Component {name!r} not in PDK {self.name!r}.")
        return self.components[name]

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Pdk {self.name!r}: {len(self.components)} components, {len(self.layers)} layers>"
