"""Tests for the PDK system and the example PDK."""
from __future__ import annotations

import photonix as px


def test_demo_pdk_components():
    pdk = px.pdk.demo_pdk()
    assert pdk.name == "photonix_demo"
    assert "straight" in pdk.components
    assert "WG" in pdk.layers


def test_pdk_layout_and_model():
    pdk = px.pdk.demo_pdk()
    cell = pdk.get_layout("straight", length=8.0)
    assert len(cell.ports) == 2
    s = pdk.evaluate("straight", wl=1.55, length=8.0)
    assert ("o1", "o2") in s
    # 2 dB/cm over 8 um is a tiny loss -> power close to 1
    assert 0.99 < float(px.power(s[("o1", "o2")])) <= 1.0


def test_pdk_layer_map():
    pdk = px.pdk.demo_pdk()
    assert pdk.layers["WG"].tuple == (1, 0)
    assert pdk.layers["METAL"].tuple == (11, 0)


def test_pdk_eme_components_registered():
    import photonix as px

    pdk = px.pdk.demo_pdk()
    assert "taper" in pdk.components and "mmi1x2" in pdk.components
    # the taper EME model is callable and returns a 2-port SDict (small/fast settings)
    s = pdk.evaluate("taper", wl=1.55, width1=0.5, width2=0.6, length=5.0,
                     num_sections=5, num_modes=4, points=141)
    assert ("o1", "o2") in s
