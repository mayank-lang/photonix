"""PDK-agnostic process design kit interface.

A :class:`Pdk` is a registry that ties a component *name* to (1) a layout
generator producing a :class:`~photonix.layout.Cell`, (2) an optional circuit
*model* producing an ``SDict``, and (3) default settings for each (kept separate
because layout params like ``width`` differ from model params like ``neff``). It
also carries a layer map. PDKs let one design target different foundries by
swapping the kit.

Beyond the component registry a PDK can carry

* :class:`~photonix.pdk.CrossSection` definitions -- waveguide types whose
  compact-model parameters :meth:`Pdk.models` binds into every compatible
  model, so :meth:`Pdk.simulate` evaluates an extracted layout with the
  kit's waveguide properties rather than library defaults;
* a :class:`~photonix.pdk.LayerStack` for mode solving and Meep export;
* per-layer design rules for :meth:`Pdk.drc_precheck`.
"""
from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

from .process import ProcessStudy
from .stack import CrossSection, LayerStack

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
        self.process_studies: dict[str, ProcessStudy] = {}
        self.cross_sections: dict[str, CrossSection] = {}
        self.default_cross_section: str | None = None
        self.layer_stack: LayerStack | None = None
        self.design_rules: dict[str, dict[str, float]] = {}
        self._model_overrides: dict[str, Callable] = {}

    def add_layer(self, layer: Layer) -> Pdk:
        self.layers[layer.name] = layer
        return self

    def add_component(self, name: str, *, layout=None, model=None, layout_settings=None, model_settings=None) -> Pdk:
        self.components[name] = ComponentSpec(
            name, layout, model, dict(layout_settings or {}), dict(model_settings or {})
        )
        return self

    def add_process_study(self, name: str, study: ProcessStudy) -> Pdk:
        """Register a named, explicitly calibrated process study."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("process study registry name must be a non-empty string")
        if not isinstance(study, ProcessStudy):
            raise TypeError("study must be a ProcessStudy")
        self.process_studies[name.strip()] = study
        return self

    def get_process_study(self, name: str) -> ProcessStudy:
        """Return a registered process study by name."""
        if name not in self.process_studies:
            raise KeyError(f"Process study {name!r} not in PDK {self.name!r}.")
        return self.process_studies[name]

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

    # -- cross-sections, stack, rules ----------------------------------------- #
    def add_cross_section(self, xs: CrossSection, *, default: bool = False) -> Pdk:
        """Register a waveguide cross-section (the first one becomes the default)."""
        if not isinstance(xs, CrossSection):
            raise TypeError("xs must be a CrossSection")
        self.cross_sections[xs.name] = xs
        if default or self.default_cross_section is None:
            self.default_cross_section = xs.name
        return self

    def cross_section(self, name: str | None = None) -> CrossSection:
        """A registered cross-section (``None`` for the default)."""
        key = self.default_cross_section if name is None else name
        if key is None or key not in self.cross_sections:
            raise KeyError(f"cross-section {name!r} not in PDK {self.name!r}")
        return self.cross_sections[key]

    def set_layer_stack(self, stack: LayerStack) -> Pdk:
        if not isinstance(stack, LayerStack):
            raise TypeError("stack must be a LayerStack")
        self.layer_stack = stack
        return self

    def add_design_rules(self, layer: str, **rules: float) -> Pdk:
        """Add ``min_width``/``min_space``/``min_area`` rules for a named layer."""
        if layer not in self.layers:
            raise KeyError(f"layer {layer!r} not in PDK {self.name!r}")
        self.design_rules.setdefault(layer, {}).update({k: float(v) for k, v in rules.items()})
        return self

    def drc_precheck(self, cell):
        """Run :func:`photonix.layout.run_drc_precheck` with this kit's rules."""
        from photonix.layout.drc import run_drc_precheck

        return run_drc_precheck(cell, {self.layers[n].tuple: r for n, r in self.design_rules.items()})

    # -- simulation ----------------------------------------------------------- #
    def add_model(self, name: str, model: Callable, **defaults) -> Pdk:
        """Override (or add) the extraction model called ``name`` with bound defaults."""
        if not callable(model):
            raise TypeError("model must be callable")
        self._model_overrides[name] = _bind(model, defaults, declared=False) if defaults else model
        return self

    def models(self, cross_section: str | None = None) -> dict[str, Callable]:
        """Model registry for extracted layouts of this kit.

        Starts from :data:`photonix.components.MODELS`, binds the selected
        cross-section's ``model_settings`` into the parameters each model
        declares as waveguide properties (its ``waveguide_parameters``
        attribute; see :data:`photonix.components.WAVEGUIDE_PARAMETERS`),
        then applies :meth:`add_model` overrides. Call-time settings still win.
        Binding is by declaration rather than by parameter name because names
        are reused with different meanings (``wl0`` is a dispersion reference
        for a waveguide but the peak wavelength of a grating coupler).
        Component registrations made with :meth:`add_component` do not change
        this registry; they serve :meth:`get_layout` and :meth:`evaluate`.
        """
        from photonix.components import MODELS

        explicit = self.cross_sections or cross_section is not None
        defaults = self.cross_section(cross_section).model_settings if explicit else {}
        registry = {name: _bind(cast(Callable, fn), defaults, declared=True) for name, fn in MODELS.items()}
        registry.update(self._model_overrides)
        return registry

    def simulate(self, cell, *, wl=1.55, cross_section: str | None = None, **overrides):
        """Extract ``cell``'s netlist and simulate it with :meth:`models`."""
        from photonix.circuit import circuit_from_netlist
        from photonix.layout import extract_netlist

        return circuit_from_netlist(extract_netlist(cell), self.models(cross_section))(wl=wl, **overrides)

    def route(self, port1, port2, *, cross_section: str | None = None, **kwargs):
        """:func:`photonix.layout.route_manhattan` with a cross-section's width, layer and radius."""
        from photonix.layout import route_manhattan

        xs = self.cross_section(cross_section)
        opts = {"radius": xs.radius, "width": xs.width, "layer": xs.layer, **kwargs}
        return route_manhattan(port1, port2, **opts)

    def _spec(self, name: str) -> ComponentSpec:
        if name not in self.components:
            raise KeyError(f"Component {name!r} not in PDK {self.name!r}.")
        return self.components[name]

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Pdk {self.name!r}: {len(self.components)} components, {len(self.layers)} layers>"


def _bind(fn: Callable, defaults: Mapping, *, declared: bool) -> Callable:
    """Bind defaults into ``fn``; call-time kwargs win.

    With ``declared=True`` only the names in ``fn.waveguide_parameters`` are
    bound; otherwise every default that ``fn``'s signature accepts.
    """
    if declared:
        params = set(getattr(fn, "waveguide_parameters", ()))
        accepted = {k: v for k, v in defaults.items() if k in params}
    else:
        try:
            sig = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            sig = None
        if sig is None or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.values()):
            accepted = dict(defaults)
        else:
            unknown = sorted(set(defaults) - set(sig))
            if unknown:
                raise TypeError(f"model {getattr(fn, '__name__', fn)!r} does not accept {unknown}")
            accepted = dict(defaults)
    if not accepted:
        return fn

    @functools.wraps(fn)
    def bound(**kwargs):
        return fn(**{**accepted, **kwargs})

    return bound
