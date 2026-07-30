"""Meep-free contracts for generalized modal-port orchestration."""
from __future__ import annotations

import types

import numpy as np
import pytest

import photonix.em.meep.multiport as multiport
from photonix.em import meep
from photonix.layout import Cell


def _prepared(name: str = "junction"):
    cell = Cell(name)
    cell.add_polygon([(-2, -0.3), (2, -0.3), (2, 0.3), (-2, 0.3)], layer=(1, 0))
    cell.add_port("west", center=(-2, 0), orientation=180, width=0.6, layer=(1, 0))
    cell.add_port("east", center=(2, 0), orientation=0, width=0.6, layer=(1, 0))
    return meep.prepare_layout(cell, [meep.LayerSpec((1, 0), epsilon=12.0)], margin=1.0)


def test_multiport_plan_expands_modal_terminals_and_counts_runs():
    plan = meep.plan_multiport(
        _prepared(),
        [1.50, 1.55],
        port_modes={"east": (2, 1), "west": meep.PortModeSpec((1, 3))},
    )
    assert plan.terminal_names == ("east:m2", "east:m1", "west:m1", "west:m3")
    assert plan.run_count == 8
    assert meep.modal_terminal("drop", 4) == "drop:m4"


def test_multiport_plan_rejects_bad_grids_ports_and_diagonal_planes():
    prepared = _prepared()
    with pytest.raises(ValueError, match="strictly increasing"):
        meep.plan_multiport(prepared, [1.55, 1.55])
    with pytest.raises(KeyError, match="unknown"):
        meep.plan_multiport(prepared, [1.55], port_modes={"missing": (1,)})
    with pytest.raises(ValueError, match="positive MPB"):
        meep.PortModeSpec((0,))

    cell = Cell("diagonal")
    cell.add_polygon([(0, 0), (1, 0), (1, 1)], layer=(1, 0))
    cell.add_port("diag", center=(0, 0), orientation=45, width=0.5, layer=(1, 0))
    diagonal = meep.prepare_layout(cell, [meep.LayerSpec((1, 0), epsilon=4.0)], margin=1.0)
    with pytest.raises(ValueError, match="axis-aligned"):
        meep.plan_multiport(diagonal, [1.55])


def test_multiport_assembles_every_measured_column_without_reciprocity(monkeypatch):
    prepared = _prepared()
    calls = []

    def fake_run(plan, incident, wl, **_kwargs):
        calls.append((incident.name, wl))
        incoming_index = plan.terminal_names.index(incident.name)
        result = {}
        for outgoing_index, name in enumerate(plan.terminal_names):
            outgoing = complex(outgoing_index + 1, incoming_index + 1)
            result[name] = (outgoing, 2.0 if name == incident.name else 0.0)
        return result

    monkeypatch.setattr(multiport, "_run_excitation", fake_run)
    dataset = meep.simulate_multiport_sparameters(
        prepared,
        wavelengths=[1.50, 1.60],
        resolution=20,
        port_modes={"west": (1, 2), "east": (1,)},
        pml=0.5,
    )
    assert dataset.ports == ("west:m1", "west:m2", "east:m1")
    assert dataset.s.shape == (2, 3, 3)
    for wi in range(2):
        for outgoing in range(3):
            for incoming in range(3):
                assert dataset.s[wi, outgoing, incoming] == complex(outgoing + 1, incoming + 1) / 2
    assert len(calls) == 6
    assert dataset.metadata["device_runs"] == 6
    assert dataset.metadata["normalization"] == "device-incident"

    with pytest.raises(ValueError, match="orchestrator-owned"):
        meep.simulate_multiport_sparameters(
            prepared,
            wavelengths=[1.55],
            resolution=20,
            pml=0.5,
            simulation_kwargs={"sources": []},
        )


def test_multiport_reference_normalizes_and_subtracts_launch_reflection(monkeypatch):
    device = _prepared("device")
    reference = _prepared("reference")

    def fake_run(plan, incident, _wl, **_kwargs):
        is_reference = plan.prepared.name == "reference"
        result = {}
        for terminal in plan.terminals:
            name = terminal.name
            same_physical_port = terminal.port_name == incident.port_name
            if is_reference:
                result[name] = (
                    0.25 if same_physical_port else 3.0,
                    4.0 if name == incident.name else 0.0,
                )
            else:
                result[name] = (
                    1.25 if same_physical_port else 2.0,
                    7.0 if name == incident.name else 0.0,
                )
        return result

    monkeypatch.setattr(multiport, "_run_excitation", fake_run)
    dataset = meep.simulate_multiport_sparameters(
        device,
        wavelengths=[1.55],
        resolution=20,
        port_modes={"west": (1, 2), "east": (1,)},
        reference=reference,
        pml=0.5,
    )
    assert np.allclose(np.diag(dataset.s[0]), 0.25)
    # Cross-mode reflection on the incident physical port also has its launch
    # background removed; transmission to a different physical port does not.
    assert dataset.s[0, 1, 0] == pytest.approx(0.25)
    assert dataset.s[0, 2, 0] == pytest.approx(0.5)
    assert dataset.metadata["reference_runs"] == 3
    assert dataset.metadata["normalization"] == "reference"


def test_one_excitation_uses_port_local_inward_source_and_outward_decomposition(monkeypatch):
    plan = meep.plan_multiport(
        _prepared(),
        [1.55],
        port_modes={
            "west": meep.PortModeSpec((1, 2), source_offset=0.05, monitor_offset=0.15),
            "east": (1,),
        },
    )
    source_calls = []
    region_calls = []

    class FakeSource:
        def __init__(self, *args, **kwargs):
            source_calls.append((args, kwargs))

    fake_mp = types.SimpleNamespace(
        NO_PARITY="no-parity",
        NO_DIRECTION="no-direction",
        GaussianSource=lambda f, **kwargs: ("gaussian", f, kwargs),
        EigenModeSource=FakeSource,
        stop_when_dft_decayed=lambda **kwargs: ("stop", kwargs),
    )

    def fake_region(port, **kwargs):
        region_calls.append((port.name, kwargs))
        normal = tuple(port.outward_normal)
        return types.SimpleNamespace(
            region=types.SimpleNamespace(center=(port.name, "center"), size=(port.name, "size"), name=port.name),
            outward_kpoint=normal,
            inward_kpoint=tuple(-value for value in normal),
        )

    class FakeSimulation:
        def __init__(self):
            self.run_condition = None

        def add_mode_monitor(self, _fcen, _df, _nfreq, region):
            return region.name

        def run(self, *, until_after_sources):
            self.run_condition = until_after_sources

        def get_eigenmode_coefficients(self, monitor, bands, **kwargs):
            assert kwargs["direction"] == "no-direction"
            guessed = kwargs["kpoint_func"](1 / 1.55, bands[0])
            expected = tuple(plan.prepared.ports[monitor].outward_normal)
            assert guessed == expected
            alpha = np.zeros((len(bands), 1, 2), dtype=complex)
            for index, band in enumerate(bands):
                alpha[index, 0] = (band + 0.1j, band + 0.2j)
            return types.SimpleNamespace(alpha=alpha)

    simulation = FakeSimulation()
    monkeypatch.setattr(multiport, "require_mpb", lambda: (fake_mp, object()))
    monkeypatch.setattr(multiport, "port_region", fake_region)
    monkeypatch.setattr(
        multiport,
        "build_layout_simulation",
        lambda *_args, **_kwargs: (simulation, object()),
    )
    coefficients = multiport._run_excitation(
        plan,
        plan.terminals[0],
        1.55,
        resolution=20,
        pml=0.5,
        fwidth_frac=0.1,
        dft_decay_tol=1e-8,
        minimum_run_time=5,
        maximum_run_time=100,
        default_material=None,
        simulation_kwargs={},
    )
    assert source_calls[0][1]["eig_kpoint"] == pytest.approx((1.0, 0.0))
    assert source_calls[0][1]["direction"] == "no-direction"
    assert region_calls[0] == ("west", {"span": None, "z_span": 0.0, "inward_offset": 0.05})
    assert ("west", {"span": None, "z_span": 0.0, "inward_offset": 0.15}) in region_calls
    assert coefficients["west:m2"] == (2 + 0.1j, 2 + 0.2j)
    assert simulation.run_condition == (
        "stop",
        {"tol": 1e-8, "minimum_run_time": 5, "maximum_run_time": 100},
    )
