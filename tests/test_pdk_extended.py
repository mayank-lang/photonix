"""Layer stacks, cross-sections, PDK-aware simulation, routing and rule checks."""
from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import requires_gdstk

import photonix as px
import photonix.layout as lay
from photonix.layout import components as lc
from photonix.pdk import CrossSection, LayerLevel, LayerStack, Pdk


def test_stack_cross_section_reproduces_rectangular_waveguide():
    stack = LayerStack([
        LayerLevel("clad", None, 10.0, -5.0, 1.444),
        LayerLevel("core", (1, 0), 0.22, -0.11, 3.4757),
    ])
    ours = stack.cross_section({(1, 0): 0.5}, resolution=40, margin=1.5)
    ref = px.em.rectangular_waveguide(width=0.5, thickness=0.22, resolution=40, margin=1.5)
    assert ours.eps.shape == ref.eps.shape
    assert np.allclose(ours.eps, ref.eps, atol=1e-12)
    assert np.allclose(ours.x, ref.x) and np.allclose(ours.y, ref.y)


def test_rib_has_higher_index_than_strip_and_dispersion_is_normal():
    stack = px.pdk.demo_pdk().layer_stack
    strip = stack.solve_modes({(1, 0): 0.5}, resolution=30).neff0
    rib = stack.solve_modes({(1, 0): 0.5, (2, 0): 4.0}, resolution=30).neff0
    assert rib > strip
    longer = stack.solve_modes({(1, 0): 0.5}, wl=1.6, resolution=30).neff0
    assert longer < strip


def test_meep_layer_specs_follow_the_stack():
    specs = {s.layer: s for s in px.pdk.demo_pdk().layer_stack.to_meep_layer_specs(1.55)}
    core = specs[(1, 0)]
    assert core.thickness == pytest.approx(0.22) and core.z_center == pytest.approx(0.11)
    assert core.epsilon == pytest.approx(float(px.em.silicon(1.55)) ** 2)
    assert set(specs) == {(1, 0), (2, 0)}


def test_stack_validation():
    with pytest.raises(ValueError, match="thickness"):
        LayerLevel("bad", None, 0.0, 0.0, 1.0)
    stack = LayerStack([LayerLevel("core", (1, 0), 0.2, 0.0, 3.4)])
    with pytest.raises(ValueError, match="background"):
        stack.cross_section({(1, 0): 0.5})
    with pytest.raises(ValueError, match="patterned"):
        stack.cross_section({(9, 0): 0.5})


def test_pdk_simulate_uses_cross_section_loss():
    pdk = px.pdk.demo_pdk()
    top = lay.Cell("top")
    top.add_ref(pdk.get_layout("straight", length=10000.0), name="wg")      # 1 cm
    s = pdk.simulate(top, wl=1.55)
    assert -10 * math.log10(float(px.power(s[("wg_o1", "wg_o2")]))) == pytest.approx(2.0, rel=1e-9)
    # library defaults alone would be lossless
    s0 = px.circuit.circuit_from_netlist(lay.extract_netlist(top))(wl=1.55)
    assert float(px.power(s0[("wg_o1", "wg_o2")])) == pytest.approx(1.0)
    # call-time settings still win over the bound cross-section defaults
    s1 = pdk.simulate(top, wl=1.55, wg={"loss_db_cm": 0.0})
    assert float(px.power(s1[("wg_o1", "wg_o2")])) == pytest.approx(1.0)


def test_models_bind_only_declared_waveguide_parameters():
    pdk = px.pdk.demo_pdk()
    reg = pdk.models()
    assert reg["mmi1x2"] is px.components.MODELS["mmi1x2"]     # takes no neff/ng/loss
    assert reg["straight"] is not px.components.MODELS["straight"]
    # An O-band cross-section must not move a grating coupler's peak or an
    # AWG's channel plan, although both have a parameter called wl0.
    oband = Pdk("oband")
    oband.add_cross_section(CrossSection("x", (1, 0), 0.4, model_settings={
        "neff": 2.5, "ng": 4.3, "wl0": 1.31, "loss_db_cm": 1.0}))
    reg = oband.models()
    assert reg["grating_coupler"] is px.components.MODELS["grating_coupler"]
    wl = np.linspace(1.5, 1.6, 2001)
    peak = wl[np.argmax(np.abs(np.asarray(reg["grating_coupler"](wl=wl)[("o1", "o2")])))]
    assert peak == pytest.approx(1.55, abs=1e-4)
    d = px.components.awg_design(n_channels=4, channel_spacing=0.0016, neff=2.5, ng=4.3)
    wl_k = d["channel_wavelengths"][1]
    sweep = np.linspace(wl_k - 0.0004, wl_k + 0.0004, 801)
    t = np.asarray(px.power(reg["awg"](wl=sweep, n_channels=4, channel_spacing=0.0016)[("o1", "o3")]))
    assert sweep[np.argmax(t)] == pytest.approx(wl_k, abs=2e-6)  # channel plan still centred on 1.55 um
    assert 0.9 < t.max() < 1.0                                    # but the arms carry the bound loss
    # the waveguide itself does pick up the cross-section loss
    s = reg["straight"](wl=1.31, length=1e4)
    assert -10 * math.log10(float(px.power(s[("o1", "o2")]))) == pytest.approx(1.0)


def test_add_model_override_reaches_simulation():
    pdk = px.pdk.demo_pdk()
    pdk.add_model("mmi1x2", px.components.mmi1x2, loss_db=0.5)
    top = lay.Cell("top")
    top.add_ref(lc.mmi1x2(), name="m")
    s = pdk.simulate(top)
    assert float(px.power(s[("m_o1", "m_o2")])) == pytest.approx(0.5 * 10 ** -0.05)


def test_pdk_route_uses_the_cross_section():
    pdk = px.pdk.demo_pdk()
    r = pdk.route(lay.Port("a", (0, 0), 0), lay.Port("b", (40, 30), 180))
    assert r.polygons[0][1] == (1, 0)
    assert r.model_settings["n_bends"] == 2


@requires_gdstk  # the DRC pre-check merges polygons with gdstk
def test_pdk_drc_precheck():
    pdk = px.pdk.demo_pdk()
    assert pdk.drc_precheck(pdk.get_layout("ring_single", gap=0.2)).clean
    report = pdk.drc_precheck(pdk.get_layout("ring_single", gap=0.1))
    assert report.summary() == {("min_space", (1, 0)): 1}
    with pytest.raises(KeyError):
        pdk.add_design_rules("NOPE", min_width=1.0)


def test_every_demo_component_builds_and_evaluates():
    pdk = px.pdk.demo_pdk()
    slow = {"taper", "mmi1x2"}           # EME-backed, covered elsewhere
    for name, spec in pdk.components.items():
        if spec.layout is not None:
            cell = pdk.get_layout(name)
            assert isinstance(cell, lay.Cell)
        if spec.model is not None and name not in slow:
            s = pdk.evaluate(name, wl=1.55)
            px.core.validate_sdict(s)


def test_cross_section_registry():
    pdk = Pdk("x")
    with pytest.raises(KeyError):
        pdk.cross_section()
    pdk.add_cross_section(CrossSection("a", (1, 0), 0.5))
    pdk.add_cross_section(CrossSection("b", (1, 0), 0.8, radius=20.0), default=True)
    assert pdk.cross_section().name == "b" and pdk.cross_section("a").width == 0.5
    with pytest.raises(ValueError):
        CrossSection("c", (1, 0), -1.0)
