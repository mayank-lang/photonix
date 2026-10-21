"""Schematic-driven layout, pic-YAML round trips and deterministic PCells."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_gdstk

import photonix.layout as lay
from photonix.circuit import Netlist, compile_circuit
from photonix.layout import (
    Placement,
    RouteSpec,
    SchematicSpec,
    build_layout,
    extract_netlist,
    pcell,
    pcell_info,
    schematic_from_layout,
    settings_hash,
)
from photonix.layout import components as lc
from photonix.pdk import demo_pdk

SRC = Path(__file__).resolve().parents[1] / "src"


def _spec() -> SchematicSpec:
    nl = Netlist(name="two_mmis")
    nl.add("m0", "mmi2x2").add("m1", "mmi2x2").add("s0", "straight", length=15.0)
    nl.connect(("m0", "o4"), ("s0", "o1"))
    nl.expose("in", ("m0", "o1")).expose("in2", ("m0", "o2"))
    nl.expose("out", ("m1", "o4")).expose("out2", ("m1", "o3"))
    return SchematicSpec(
        nl,
        placements={"m0": Placement(), "m1": Placement(x=150.0, y=30.5, port="o1")},
        routes={"bundle": RouteSpec({"s0,o2": "m1,o1", "m0,o3": "m1,o2"}, {"radius": 5.0}, "route_bundle")},
    )


def _pairs(conn):
    return {frozenset(item) for item in conn.items()}


def test_build_layout_closes_schematic_and_simulates():
    pdk = demo_pdk()
    spec = _spec()
    cell = build_layout(spec, pdk)
    m1 = next(r for r in cell.references if r.name == "m1")
    assert m1.origin == pytest.approx((150.0, 30.0), abs=1e-12)           # port o1 anchored at (150, 30.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                   # no misplaced / width-mismatched ports
        nl = extract_netlist(cell)
    assert set(nl.instances) == {"m0", "m1", "s0", "bundle_0", "bundle_1"}
    expected = {
        frozenset({("m0", "o4"), ("s0", "o1")}),
        frozenset({("s0", "o2"), ("bundle_0", "o1")}), frozenset({("bundle_0", "o2"), ("m1", "o1")}),
        frozenset({("m0", "o3"), ("bundle_1", "o1")}), frozenset({("bundle_1", "o2"), ("m1", "o2")}),
    }
    assert _pairs(nl.connections) == expected                            # LVS-lite: topology is the schematic's
    assert nl.ports == spec.netlist.ports

    # The schematic with the drawn route lengths inserted simulates like the layout.
    ref = Netlist(name="expected")
    ref.add("m0", "mmi2x2").add("m1", "mmi2x2").add("s0", "straight", length=15.0)
    for k in (0, 1):
        r = next(x for x in cell.references if x.name == f"bundle_{k}")
        ref.add(f"bundle_{k}", "route", **r.cell.model_settings)
    for a, b in (tuple(p) for p in expected):
        ref.connect(a, b)
    for ext, ip in spec.netlist.ports.items():
        ref.expose(ext, ip)
    wl = np.linspace(1.53, 1.57, 9)
    s_layout = pdk.simulate(cell, wl=wl)
    s_ref = compile_circuit(ref, pdk.models())(wl=wl)
    assert max(float(np.max(np.abs(np.asarray(s_layout[k]) - np.asarray(s_ref[k])))) for k in s_ref) <= 1e-12


def test_schematic_from_layout_recovers_placements_connections_and_routes():
    spec = _spec()
    cell = build_layout(spec, demo_pdk())
    back = schematic_from_layout(cell)
    assert back.placements["m0"].x == pytest.approx(0.0, abs=1e-3)
    assert (back.placements["m1"].x, back.placements["m1"].y) == pytest.approx((150.0, 30.0), abs=1e-3)
    assert back.placements["s0"].x == pytest.approx(6.0, abs=1e-3)
    assert _pairs(back.netlist.connections) == _pairs(spec.netlist.connections)
    links = {frozenset((a, b)) for r in back.routes.values() for a, b in r.links.items()}
    assert links == {frozenset(item) for item in spec.routes["bundle"].links.items()}
    assert back.netlist.ports == spec.netlist.ports
    # Placing every instance where it was found reproduces the geometry to 1 nm.
    again = schematic_from_layout(build_layout(back, demo_pdk()))
    for inst, p in back.placements.items():
        q = again.placements[inst]
        assert (q.x, q.y, q.rotation, q.mirror) == pytest.approx((p.x, p.y, p.rotation, p.mirror), abs=1e-3)


def test_conflicting_connections_raise():
    nl = Netlist()
    nl.add("a", "straight", length=10.0).add("b", "straight", length=10.0)
    nl.connect(("a", "o2"), ("b", "o1"))
    nl.expose("o1", ("a", "o1")).expose("o2", ("b", "o2"))
    ok = build_layout(SchematicSpec(nl, {"a": Placement(), "b": Placement(x=10.0)}))
    assert ok.ports["o2"].center == pytest.approx((20.0, 0.0))
    with pytest.raises(ValueError, match="cannot be satisfied"):
        build_layout(SchematicSpec(nl, {"a": Placement(), "b": Placement(x=12.0)}))
    with pytest.raises(ValueError, match="already connected"):
        SchematicSpec(nl, routes={"r": RouteSpec({"a,o2": "b,o2"})})
    with pytest.raises(KeyError, match="not found"):
        build_layout(SchematicSpec(Netlist(instances={"x": "no_such_component"})))


def test_unplaced_islands_go_to_origin_and_routes_use_strategies():
    nl = Netlist()
    nl.add("a", "straight", length=10.0).add("b", "straight", length=10.0).add("c", "straight", length=4.0)
    nl.connect(("b", "o2"), ("c", "o1"))
    spec = SchematicSpec(nl, {"b": Placement(x=50.0, y=40.0, rotation=90.0)},
                         {"r1": RouteSpec({"a,o2": "b,o1"}, {"radius": 5.0}, "route_auto")})
    cell = build_layout(spec)
    a = next(r for r in cell.references if r.name == "a")
    assert a.origin == (0.0, 0.0)
    assert set(extract_netlist(cell).instances) == {"a", "b", "c", "r1_0"}
    spec2 = SchematicSpec(nl, spec.placements, {"r1": RouteSpec({"a,o2": "b,o1"}, {"radius": 5.0}, "route_manhattan")})
    assert "r1_0" in extract_netlist(build_layout(spec2)).instances
    with pytest.raises(ValueError, match="routing strategy"):
        RouteSpec({"a,o2": "b,o1"}, strategy="nope")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# pic-YAML
# --------------------------------------------------------------------------- #
GF_SAMPLE = """
name: sample_mmis
instances:
  mmi_long:
    component: mmi2x2
    settings: {length: 10.0}
  mmi_short:
    component: mmi2x2
    settings: {length: 5.0}
placements:
  mmi_short: {port: o1, x: 20, y: 10}
  mmi_long: {port: o1, x: 120, y: 60, rotation: 0}
routes:
  optical:
    links: {'mmi_short,o4': 'mmi_long,o1'}
    settings: {radius: 5.0}
    routing_strategy: route_auto
ports:
  o1: mmi_short,o1
  o2: mmi_long,o4
"""


def test_yaml_round_trip_is_idempotent_and_reads_gdsfactory_sample(tmp_path):
    pytest.importorskip("yaml")
    spec = SchematicSpec.from_yaml(GF_SAMPLE)
    assert spec.placements["mmi_short"].port == "o1" and spec.routes["optical"].strategy == "route_auto"
    text = spec.to_yaml(tmp_path / "s.yml")
    again = SchematicSpec.from_yaml(tmp_path / "s.yml")
    assert again == spec and again.to_yaml() == text
    s2 = _spec()
    assert SchematicSpec.from_yaml(s2.to_yaml()) == s2
    cell = build_layout(spec)
    assert cell.name == "sample_mmis"
    assert cell.ports["o1"].center == pytest.approx((20.0, 10.0))
    nl = extract_netlist(cell)
    assert _pairs(nl.connections) == {frozenset({("mmi_short", "o4"), ("optical_0", "o1")}),
                                      frozenset({("optical_0", "o2"), ("mmi_long", "o1")})}


# --------------------------------------------------------------------------- #
# PCells
# --------------------------------------------------------------------------- #
def test_pcell_identity_names_and_cache():
    straight = pcell(lc.straight)
    a, b = straight(10.0), straight(length=10.0, width=0.5)
    assert a is b
    c = straight(length=10.000000000001)        # below the 1e-9 grid: same cell
    assert c is a
    d = straight(length=10.5)
    assert d is not a and d.name != a.name and d.name.startswith("straight_")
    assert pcell_info(d) == {"component": "straight", "settings": {"length": 10.5, "width": 0.5, "layer": [1, 0]}}
    assert a.name == f"straight_{settings_hash({'length': 10, 'width': 0.5, 'layer': [1, 0]})}"
    info = straight.cache_info()
    assert info["hits"] == 2 and info["misses"] == 2
    straight.cache_clear()
    assert straight(10.0) is not a and straight(10.0).name == a.name
    ring = pcell(lc.ring)                      # a leaf without a model keeps simulating as 'ring'
    assert ring(radius=7.0).model == "ring"
    mzi = pcell(name="my_mzi")(lc.mzi)
    m = mzi(delta_length=15.0)
    assert m.model is None and m.name.startswith("my_mzi_") and extract_netlist(m).instances
    nocache = pcell(lc.straight, cache_size=0)
    assert nocache(3.0) is not nocache(3.0) and nocache(3.0).name == nocache(3.0).name


def test_pcell_names_are_stable_across_interpreters():
    code = textwrap.dedent("""
        from photonix.layout import components as lc, pcell
        f = pcell(lc.taper)
        print(f(length=12.5, width1=0.45, width2=1.1).name)
    """)
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONHASHSEED": "random"}
    names = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                            check=True).stdout.strip() for _ in range(2)}
    local = pcell(lc.taper)(length=12.5, width1=0.45, width2=1.1).name
    assert names == {local}


@requires_gdstk
def test_gds_export_writes_one_structure_per_pcell_setting(tmp_path):
    import gdstk

    straight = pcell(lc.straight)
    top = lay.Cell("top")
    for k in range(10):
        top.add_ref(straight(10.0), origin=(0.0, 3.0 * k), name=f"s{k}")
    top.add_ref(straight(20.0), origin=(0.0, -5.0), name="long")
    path = tmp_path / "pc.gds"
    lay.write_gds(top, path)
    names = sorted(c.name for c in gdstk.read_gds(str(path)).cells)
    assert names == sorted(["top", straight(10.0).name, straight(20.0).name])
    assert not any("$" in n for n in names)
