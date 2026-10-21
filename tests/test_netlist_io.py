"""SAX / gdsfactory netlist dictionaries: round trips, recursive netlists, input forms."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import unitary_group

import photonix as px
from photonix.circuit import (
    HierarchicalNetlist,
    Netlist,
    Subcircuit,
    compile_circuit,
    compile_hierarchical,
    from_sax_netlist,
    parse_instance_port,
    to_sax_netlist,
)


def _mzi() -> Netlist:
    nl = Netlist(name="mzi")
    nl.add("c1", "coupler", coupling=0.5).add("c2", "coupler", coupling=0.5)
    nl.add("top", "straight", length=30.0).add("bot", "straight", length=10.0)
    nl.connect(("c1", "o4"), ("top", "o1")).connect(("top", "o2"), ("c2", "o1"))
    nl.connect(("c1", "o3"), ("bot", "o1")).connect(("bot", "o2"), ("c2", "o2"))
    nl.expose("o1", ("c1", "o1")).expose("o2", ("c1", "o2")).expose("o3", ("c2", "o3")).expose("o4", ("c2", "o4"))
    return nl


def _lattice() -> Netlist:
    nl = Netlist(name="lattice")
    for i, k in enumerate((0.5, 0.2, 0.5)):
        nl.add(f"c{i}", "coupler", coupling=k)
    for i, dl in enumerate((20.0, 40.0)):
        nl.add(f"t{i}", "straight", length=10 + dl, neff=2.4, ng=4.2).add(f"b{i}", "straight", length=10.0)
        nl.add(f"ps{i}", "phase_shifter", phase0=0.1 * i, length=0.0)
        nl.connect((f"c{i}", "o4"), (f"t{i}", "o1")).connect((f"t{i}", "o2"), (f"ps{i}", "o1"))
        nl.connect((f"ps{i}", "o2"), (f"c{i + 1}", "o1"))
        nl.connect((f"c{i}", "o3"), (f"b{i}", "o1")).connect((f"b{i}", "o2"), (f"c{i + 1}", "o2"))
    nl.expose("o1", ("c0", "o1")).expose("o2", ("c0", "o2")).expose("o3", ("c2", "o3")).expose("o4", ("c2", "o4"))
    return nl


def _crow() -> Netlist:
    nl = Netlist(name="crow")
    for k, kappa in enumerate((0.2, 0.02, 0.2)):
        nl.add(f"c{k}", "coupler", coupling=kappa)
    wg = {"length": math.pi * 10.0, "loss_db_cm": 2.0}
    for r in (1, 2):
        nl.add(f"r{r}_right", "straight", **wg).add(f"r{r}_left", "straight", **wg)
        nl.connect((f"c{r - 1}", "o4"), (f"r{r}_right", "o1")).connect((f"r{r}_right", "o2"), (f"c{r}", "o3"))
        nl.connect((f"c{r}", "o2"), (f"r{r}_left", "o1")).connect((f"r{r}_left", "o2"), (f"c{r - 1}", "o1"))
    nl.expose("o1", ("c0", "o2")).expose("o2", ("c0", "o3")).expose("o3", ("c2", "o4")).expose("o4", ("c2", "o1"))
    return nl


def _mesh() -> Netlist:
    prog = px.circuit.clements_decompose(unitary_group.rvs(4, random_state=3))
    nl = Netlist(name="mesh")
    cur: list = [None] * 4
    first: list = [None] * 4

    def attach(m, a, b):
        if cur[m] is None:
            first[m] = a
        else:
            nl.connect(cur[m], a)
        cur[m] = b

    for k, (m, th, ph) in enumerate(prog.mzis):
        nl.add(f"ext{k}", "phase_shifter", phase0=ph, length=0.0).add(f"bs{k}a", "coupler", coupling=0.5)
        nl.add(f"int{k}", "phase_shifter", phase0=th, length=0.0).add(f"bs{k}b", "coupler", coupling=0.5)
        nl.connect((f"ext{k}", "o2"), (f"bs{k}a", "o1")).connect((f"bs{k}a", "o4"), (f"int{k}", "o1"))
        nl.connect((f"int{k}", "o2"), (f"bs{k}b", "o1")).connect((f"bs{k}a", "o3"), (f"bs{k}b", "o2"))
        attach(m, (f"ext{k}", "o1"), (f"bs{k}b", "o4"))
        attach(m + 1, (f"bs{k}a", "o2"), (f"bs{k}b", "o3"))
    for m in range(4):
        nl.expose(f"o{m + 1}", first[m]).expose(f"o{8 - m}", cur[m])
    return nl


@pytest.mark.parametrize("build", [_mzi, _lattice, _crow, _mesh])
def test_netlist_sax_round_trip(build):
    nl = build()
    d = to_sax_netlist(nl)
    assert all("," in k and "," in v for k, v in d["connections"].items())
    back = from_sax_netlist(d)
    assert back == nl
    models = {"coupler": px.components.directional_coupler, "straight": px.components.straight,
              "phase_shifter": px.components.phase_shifter}
    wl = px.linspace(1.54, 1.56, 5)
    a, b = compile_circuit(nl, models)(wl=wl), compile_circuit(back, models)(wl=wl)
    assert all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)


def test_recursive_sax_netlist_round_trips_hierarchy():
    arm = Netlist(name="arm")
    arm.add("wg", "straight", length="=2*pi*radius").add("ps", "phase_shifter", length=0.0, phase0="=phase")
    arm.connect(("wg", "o2"), ("ps", "o1"))
    arm.expose("o1", ("wg", "o1")).expose("o2", ("ps", "o2"))
    top = Netlist(name="chain")
    top.add("a", "arm", radius=5.0).add("b", "arm", phase="=2*g")
    top.connect(("a", "o2"), ("b", "o1"))
    top.expose("in", ("a", "o1")).expose("out", ("b", "o2"))
    hnl = HierarchicalNetlist(top, {"arm": Subcircuit(arm, {"radius": 10.0, "phase": 0.0}, doc="an arm")},
                              parameters={"g": 0.25}, name="hier")
    d = to_sax_netlist(hnl)
    assert set(d) == {"top_level", "arm"}
    assert d["top_level"]["instances"]["a"] == {"component": "arm", "settings": {"radius": 5.0}}
    back = from_sax_netlist(d)
    assert back == hnl
    s1 = compile_hierarchical(hnl)(wl=1.55)
    s2 = compile_hierarchical(back)(wl=1.55)
    assert abs(complex(s1[("in", "out")]) - complex(s2[("in", "out")])) == 0.0
    with pytest.raises(ValueError, match="top_level"):
        from_sax_netlist({"x": d["arm"], "y": d["arm"]})


def test_gdsfactory_input_forms():
    d = {
        "name": "gf",
        "instances": {"a": {"component": "straight", "settings": {"length": 3}}, "b": "straight"},
        "placements": {"a": {"x": 0, "y": 0}},
        "nets": [{"p1": "a, o2", "p2": "b,o1"}],
        "ports": {"o1": "a,o1", "o2": "b,o2"},
    }
    nl = from_sax_netlist(d)
    assert nl.connections == {("a", "o2"): ("b", "o1")} and nl.settings == {"a": {"length": 3}}
    assert nl.name == "gf"
    for bad in ("a", "a,b,c", ",o1", "a,"):
        with pytest.raises(ValueError):
            parse_instance_port(bad)
    with pytest.raises(ValueError, match="nets"):
        from_sax_netlist({"instances": {}, "nets": [{"p1": "a,o1"}]})
    with pytest.raises(ValueError, match="must not contain ','"):
        to_sax_netlist(Netlist(instances={"a,b": "straight"}, ports={"o1": ("a,b", "o1")}))
