"""An open example PDK ("photonix_demo") on a generic 220 nm SOI process.

A *reference* kit for tutorials and tests, not a real foundry process. It wires
the analytic component models and layout generators with strip-waveguide
defaults, and also registers two **EME-backed** components (``taper`` and
``mmi1x2``) whose circuit models are computed rigorously from geometry via
:mod:`photonix.em.components` (slower; opt in by calling ``pdk.evaluate``).
"""
from __future__ import annotations

from photonix import components as _models
from photonix.layout import components as _layout

from .base import Layer, Pdk

__all__ = ["demo_pdk", "LAYERS"]

LAYERS = {
    "WG": Layer("WG", 1, 0),
    "SLAB": Layer("SLAB", 2, 0),
    "METAL": Layer("METAL", 11, 0),
    "LABEL": Layer("LABEL", 66, 0),
}

_WG_LAYOUT = {"width": 0.5}
_WG_MODEL = {"neff": 2.4, "ng": 4.2, "wl0": 1.55, "loss_db_cm": 2.0}


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
    """
    from photonix.em import components as _eme

    pdk = Pdk("photonix_demo")
    for layer in LAYERS.values():
        pdk.add_layer(layer)

    pdk.add_component("straight", layout=_layout.straight, model=_models.straight,
                      layout_settings=_WG_LAYOUT, model_settings=_WG_MODEL)
    pdk.add_component("bend", layout=_layout.bend_circular, model=_models.bend,
                      layout_settings=_WG_LAYOUT, model_settings=_WG_MODEL)
    pdk.add_component("directional_coupler", layout=_layout.mmi1x2, model=_models.directional_coupler,
                      layout_settings=_WG_LAYOUT, model_settings={"coupling": 0.5})
    pdk.add_component("ring", layout=_layout.ring, model=_models.all_pass_ring,
                      layout_settings=_WG_LAYOUT, model_settings=_WG_MODEL)
    pdk.add_component("grating_coupler", layout=_layout.grating_coupler, model=_models.grating_coupler,
                      model_settings={"wl0": 1.55, "bandwidth": 0.035, "peak_loss_db": 3.0})

    # EME-backed (rigorous, geometry-based) components.
    pdk.add_component("taper", layout=_layout.taper, model=_eme.taper,
                      layout_settings={"width1": 0.5, "width2": 1.0, "length": 20.0},
                      model_settings={"width1": 0.5, "width2": 1.0, "length": 20.0,
                                      "num_sections": 30, "num_modes": 6})
    pdk.add_component("mmi1x2", layout=_layout.mmi1x2, model=_eme.mmi1x2,
                      model_settings={"width_mmi": 2.5, "length_mmi": 29.5, "gap": 1.0,
                                      "num_modes": 12})
    return pdk
