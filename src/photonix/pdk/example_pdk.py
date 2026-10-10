"""An open example PDK ("photonix_demo") on a generic 220 nm SOI process.

A *reference* kit for tutorials and tests, not a real foundry process. It wires
the compact models and layout generators with strip-waveguide defaults, a
dispersive Si/SiO2 layer stack, a ``strip`` cross-section, and illustrative
width/space rules, and also registers two **EME-backed** components (``taper``
and ``mmi1x2``) whose circuit models are computed rigorously from geometry via
:mod:`photonix.em.components` (slower; opt in by calling ``pdk.evaluate``).

None of the numbers here are qualified foundry data.
"""
from __future__ import annotations

from photonix import components as _models
from photonix.layout import components as _layout

from .base import Layer, Pdk
from .stack import CrossSection, LayerLevel, LayerStack

__all__ = ["demo_pdk", "LAYERS"]

LAYERS = {
    "WG": Layer("WG", 1, 0),
    "SLAB": Layer("SLAB", 2, 0),
    "METAL": Layer("METAL", 11, 0),
    "N": Layer("N", 20, 0),
    "P": Layer("P", 21, 0),
    "LABEL": Layer("LABEL", 66, 0),
    "FIBER": Layer("FIBER", 203, 0),
}

_WG_LAYOUT = {"width": 0.5}
_WG_MODEL = {"neff": 2.4, "ng": 4.2, "wl0": 1.55, "loss_db_cm": 2.0}


def _stack() -> LayerStack:
    from photonix.em.materials import silica, silicon

    return LayerStack([
        LayerLevel("oxide", None, 5.0, -2.5, silica),       # BOX + blanket cladding
        LayerLevel("slab", LAYERS["SLAB"].tuple, 0.09, 0.0, silicon),
        LayerLevel("core", LAYERS["WG"].tuple, 0.22, 0.0, silicon),
    ])


def demo_pdk() -> Pdk:
    """Construct and return the ``photonix_demo`` PDK.

    Examples
    --------
    >>> pdk = demo_pdk()
    >>> "mmi1x2" in pdk.components and "taper" in pdk.components
    True
    >>> cell = pdk.get_layout("straight", length=8.0)
    >>> len(cell.ports)
    2
    >>> pdk.cross_section().model_settings["loss_db_cm"]     # used by pdk.simulate
    2.0
    """
    from photonix.em import components as _eme

    pdk = Pdk("photonix_demo")
    for layer in LAYERS.values():
        pdk.add_layer(layer)
    pdk.set_layer_stack(_stack())
    pdk.add_cross_section(CrossSection("strip", LAYERS["WG"].tuple, 0.5, radius=5.0,
                                       model_settings=_WG_MODEL), default=True)
    pdk.add_design_rules("WG", min_width=0.15, min_space=0.15)
    pdk.add_design_rules("METAL", min_width=2.0, min_space=2.0)

    pdk.add_component("straight", layout=_layout.straight, model=_models.straight,
                      layout_settings=_WG_LAYOUT, model_settings=_WG_MODEL)
    pdk.add_component("bend", layout=_layout.bend_circular, model=_models.bend,
                      layout_settings=_WG_LAYOUT, model_settings=_WG_MODEL)
    pdk.add_component("bend_euler", layout=_layout.bend_euler, model=_models.bend_euler,
                      layout_settings=_WG_LAYOUT, model_settings=_WG_MODEL)
    pdk.add_component("sbend", layout=_layout.sbend, layout_settings=_WG_LAYOUT)
    pdk.add_component("directional_coupler", layout=_layout.directional_coupler, model=_models.directional_coupler,
                      layout_settings=_WG_LAYOUT, model_settings={"coupling": 0.5})
    pdk.add_component("coupler", layout=_layout.coupler, model=_models.evanescent_coupler,
                      layout_settings=_WG_LAYOUT, model_settings={"gap": 0.2, "length": 10.0, **_WG_MODEL})
    pdk.add_component("ring", layout=_layout.ring, model=_models.all_pass_ring,
                      layout_settings=_WG_LAYOUT, model_settings=_WG_MODEL)
    pdk.add_component("ring_single", layout=_layout.ring_single, model=_models.ring_single,
                      layout_settings=_WG_LAYOUT, model_settings=_WG_MODEL)
    pdk.add_component("ring_double", layout=_layout.ring_double, model=_models.ring_double,
                      layout_settings=_WG_LAYOUT, model_settings=_WG_MODEL)
    pdk.add_component("mmi2x2", layout=_layout.mmi2x2, model=_models.mmi2x2)
    pdk.add_component("crossing", layout=_layout.crossing, model=_models.crossing)
    pdk.add_component("grating_coupler", layout=_layout.grating_coupler, model=_models.grating_coupler,
                      model_settings={"wl0": 1.55, "bandwidth": 0.035, "peak_loss_db": 3.0})
    pdk.add_component("edge_coupler", layout=_layout.edge_coupler, model=_models.edge_coupler)
    pdk.add_component("bragg_grating", layout=_layout.bragg_grating, model=_models.bragg_grating,
                      model_settings={"neff": 2.4, "ng": 4.2, "wl0": 1.55})
    pdk.add_component("heater", layout=_layout.heater_straight, model=_models.heater_phase_shifter,
                      model_settings={"p_pi": 25.0, **_WG_MODEL})
    pdk.add_component("pn_modulator", layout=_layout.pn_modulator, model=_models.pn_phase_shifter,
                      model_settings=_WG_MODEL)
    pdk.add_component("pad", layout=_layout.pad)
    pdk.add_component("mzi", layout=_layout.mzi)
    pdk.add_component("awg", model=_models.awg)
    pdk.add_component("mmi_nxn", model=_models.mmi_nxn)

    # EME-backed (rigorous, geometry-based) components.
    pdk.add_component("taper", layout=_layout.taper, model=_eme.taper,
                      layout_settings={"width1": 0.5, "width2": 1.0, "length": 20.0},
                      model_settings={"width1": 0.5, "width2": 1.0, "length": 20.0,
                                      "num_sections": 30, "num_modes": 6})
    pdk.add_component("mmi1x2", layout=_layout.mmi1x2, model=_eme.mmi1x2,
                      model_settings={"width_mmi": 2.5, "length_mmi": 29.5, "gap": 1.0,
                                      "num_modes": 12})
    return pdk
