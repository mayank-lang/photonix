"""Regression coverage for non-EM edge cases found during the package audit."""
from __future__ import annotations

import numpy as np
import pytest

import photonix as px


def test_nonreciprocal_dense_orientation_and_empty_passivity():
    S, pm = px.core.sdict_to_sdense({("in", "out"): 2j})
    assert S[pm["out"], pm["in"]] == 2j
    assert S[pm["in"], pm["out"]] == 0
    assert px.core.sdense_to_sdict((S, pm))[("in", "out")] == 2j
    assert px.core.is_passive({})


def test_circuit_rejects_duplicate_exposure():
    netlist = px.circuit.Netlist(instances={"a": "straight"})
    netlist.expose("x", ("a", "o1"))
    with pytest.raises(ValueError, match="already exposed"):
        netlist.expose("y", ("a", "o1"))


def test_vertical_route_and_native_cell_plotting():
    route = px.layout.route(
        px.layout.Port("a", (0, 0), width=0.5),
        px.layout.Port("b", (0, 10), width=0.5),
    )
    assert np.isclose(np.ptp(route.polygons[0][0][:, 0]), 0.5)
    assert len(px.viz.plot_cell(px.layout.components.straight()).patches) == 1


def test_complex_mode_plot_and_zero_overlap():
    assert px.viz.plot_mode(np.ones((3, 4), dtype=complex) * (1 + 1j)) is not None
    with pytest.raises(ValueError, match="zero-norm"):
        px.modes.overlap(np.zeros(2), np.zeros(2))


def test_pdk_directional_coupler_layout_matches_model():
    pdk = px.pdk.demo_pdk()
    cell = pdk.get_layout("directional_coupler")
    assert cell.name == "directional_coupler"
    assert set(cell.ports) == set(px.core.ports_of(pdk.evaluate("directional_coupler")))
