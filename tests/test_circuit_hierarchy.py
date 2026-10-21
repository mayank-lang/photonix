"""Hierarchical netlists: equivalence with flat circuits, caching, serialization, expressions."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from conftest import finite_diff, requires_jax
from scipy.stats import unitary_group

import photonix as px
import photonix.components as comp
from photonix.circuit import (
    HierarchicalNetlist,
    Netlist,
    Subcircuit,
    compile_circuit,
    compile_hierarchical,
    engine,
    evaluate_expression,
    load_netlist,
)

TOL = 1e-12


def _max_diff(a, b):
    assert set(a) == set(b)
    return max(float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k])))) for k in a)


def _attenuated_phase(*, wl=1.55, phase0=0.0, loss_db=0.0):
    base = comp.phase_shifter(wl=wl, length=0.0, phase0=phase0)
    amp = 10.0 ** (-loss_db / 20.0)
    return {k: amp * v for k, v in base.items()}


MODELS = {"coupler": comp.directional_coupler, "straight": comp.straight, "phase_shifter": comp.phase_shifter,
          "attenuated_phase": _attenuated_phase}


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def _stage_subcircuit() -> Subcircuit:
    """One lattice delay stage: upper arm (length + dl, extra phase), lower arm."""
    nl = Netlist(name="stage")
    wg = {"neff": "=neff", "ng": "=ng", "loss_db_cm": "=loss_db_cm", "wl0": 1.55}
    nl.add("top", "straight", length="=length + dl", **wg)
    nl.add("ps", "phase_shifter", phase0="=phase", length=0.0)
    nl.add("bot", "straight", length="=length", **wg)
    nl.connect(("top", "o2"), ("ps", "o1"))
    nl.expose("a1", ("top", "o1")).expose("a2", ("ps", "o2"))
    nl.expose("b1", ("bot", "o1")).expose("b2", ("bot", "o2"))
    return Subcircuit(nl, {"length": 10.0, "dl": 20.0, "phase": 0.0, "neff": 2.4, "ng": 4.2, "loss_db_cm": 0.0})


def _lattice_hnl(couplings, dls, phases) -> HierarchicalNetlist:
    top = Netlist(name="lattice")
    for i, k in enumerate(couplings):
        top.add(f"c{i}", "coupler", coupling=k)
    for i, (dl, ph) in enumerate(zip(dls, phases, strict=True)):
        top.add(f"s{i}", "stage", dl=dl, phase=ph)
        top.connect((f"c{i}", "o4"), (f"s{i}", "a1"))
        top.connect((f"s{i}", "a2"), (f"c{i + 1}", "o1"))
        top.connect((f"c{i}", "o3"), (f"s{i}", "b1"))
        top.connect((f"s{i}", "b2"), (f"c{i + 1}", "o2"))
    last = len(couplings) - 1
    top.expose("o1", ("c0", "o1")).expose("o2", ("c0", "o2"))
    top.expose("o4", (f"c{last}", "o4")).expose("o3", (f"c{last}", "o3"))
    return HierarchicalNetlist(top, {"stage": _stage_subcircuit()})


def _mzi_subcircuit() -> Subcircuit:
    nl = Netlist(name="mzi")
    nl.add("ext", "attenuated_phase", phase0="=phi", loss_db="=loss_db")
    nl.add("bsa", "coupler", coupling="=split")
    nl.add("int", "attenuated_phase", phase0="=theta", loss_db="=loss_db")
    nl.add("bsb", "coupler", coupling="=split")
    nl.connect(("ext", "o2"), ("bsa", "o1"))
    nl.connect(("bsa", "o4"), ("int", "o1"))
    nl.connect(("int", "o2"), ("bsb", "o1"))
    nl.connect(("bsa", "o3"), ("bsb", "o2"))
    nl.expose("o1", ("ext", "o1")).expose("o2", ("bsa", "o2"))
    nl.expose("o4", ("bsb", "o4")).expose("o3", ("bsb", "o3"))
    return Subcircuit(nl, {"theta": 0.0, "phi": 0.0, "split": 0.5, "loss_db": 0.0}, doc="mesh MZI")


def _mesh_hnl(program, *, splitter_coupling=0.5, phase_loss_db=0.0) -> HierarchicalNetlist:
    """The mesh of :func:`photonix.circuit.mzi_mesh` rebuilt from an ``mzi`` subcircuit."""
    n = program.n
    top = Netlist(name=f"mesh{n}")
    current: list = [None] * n
    starts: list = [None] * n

    def attach(mode, t_in, t_out):
        if current[mode] is None:
            starts[mode] = t_in
        else:
            top.connect(current[mode], t_in)
        current[mode] = t_out

    for k, (mode, theta, phi) in enumerate(program.mzis):
        top.add(f"mzi{k}", "mzi", theta=theta, phi=phi, split="=split", loss_db="=loss_db")
        attach(mode, (f"mzi{k}", "o1"), (f"mzi{k}", "o4"))
        attach(mode + 1, (f"mzi{k}", "o2"), (f"mzi{k}", "o3"))
    for mode in range(n):
        top.add(f"out{mode}", "attenuated_phase", phase0=float(program.output_phases[mode]), loss_db="=loss_db")
        attach(mode, (f"out{mode}", "o1"), (f"out{mode}", "o2"))
        top.expose(f"o{mode + 1}", starts[mode])
        top.expose(f"o{2 * n - mode}", current[mode])
    return HierarchicalNetlist(top, {"mzi": _mzi_subcircuit()},
                               parameters={"split": splitter_coupling, "loss_db": phase_loss_db})


# --------------------------------------------------------------------------- #
# (1) Equivalence with flat circuits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["hierarchical", "flatten"])
def test_lattice_of_stage_subcircuits_equals_builder(mode):
    couplings, dls, phases = (0.5, 0.3, 0.6), (20.0, 40.0), (0.2, -0.7)
    wl = px.linspace(1.5, 1.6, 101)
    ref = px.circuit.lattice_filter(couplings=couplings, delta_lengths=dls, phases=phases)(wl=wl)
    s = compile_hierarchical(_lattice_hnl(couplings, dls, phases), MODELS, mode=mode)(wl=wl)
    assert _max_diff(s, ref) <= TOL


def test_mzi_chain_hand_flattened():
    """Two MZI subcircuits in series vs a hand-written flat netlist."""
    top = Netlist()
    top.add("m0", "mzi", theta=0.4, phi=1.1).add("m1", "mzi", theta=2.0, phi=-0.3, split=0.4)
    top.connect(("m0", "o4"), ("m1", "o1")).connect(("m0", "o3"), ("m1", "o2"))
    top.expose("i1", ("m0", "o1")).expose("i2", ("m0", "o2")).expose("e1", ("m1", "o4")).expose("e2", ("m1", "o3"))
    hnl = HierarchicalNetlist(top, {"mzi": _mzi_subcircuit()})

    flat = Netlist()
    for m, (th, ph, k) in {"m0": (0.4, 1.1, 0.5), "m1": (2.0, -0.3, 0.4)}.items():
        flat.add(f"{m}.ext", "attenuated_phase", phase0=ph, loss_db=0.0)
        flat.add(f"{m}.bsa", "coupler", coupling=k)
        flat.add(f"{m}.int", "attenuated_phase", phase0=th, loss_db=0.0)
        flat.add(f"{m}.bsb", "coupler", coupling=k)
        flat.connect((f"{m}.ext", "o2"), (f"{m}.bsa", "o1"))
        flat.connect((f"{m}.bsa", "o4"), (f"{m}.int", "o1"))
        flat.connect((f"{m}.int", "o2"), (f"{m}.bsb", "o1"))
        flat.connect((f"{m}.bsa", "o3"), (f"{m}.bsb", "o2"))
    flat.connect(("m0.bsb", "o4"), ("m1.ext", "o1")).connect(("m0.bsb", "o3"), ("m1.bsa", "o2"))
    flat.expose("i1", ("m0.ext", "o1")).expose("i2", ("m0.bsa", "o2"))
    flat.expose("e1", ("m1.bsb", "o4")).expose("e2", ("m1.bsb", "o3"))

    assert hnl.flatten() == flat
    wl = px.linspace(1.52, 1.58, 7)
    ref = compile_circuit(flat, MODELS)(wl=wl)
    for mode in ("hierarchical", "flatten"):
        assert _max_diff(compile_hierarchical(hnl, MODELS, mode=mode)(wl=wl), ref) <= TOL


def _ring_subcircuit() -> Subcircuit:
    nl = Netlist(name="ring")
    wg = {"length": "=pi*radius", "neff": 2.4, "ng": 4.2, "loss_db_cm": "=loss_db_cm", "wl0": 1.55}
    nl.add("right", "straight", **wg).add("left", "straight", **wg)
    nl.expose("r1", ("right", "o1")).expose("r2", ("right", "o2"))
    nl.expose("l1", ("left", "o1")).expose("l2", ("left", "o2"))
    return Subcircuit(nl, {"radius": 10.0, "loss_db_cm": 2.0})


@pytest.mark.parametrize("couplings", [(0.3, 0.05, 0.3), (0.2, 0.02, 0.03, 0.2)])
def test_crow_of_ring_subcircuits_equals_builder(couplings):
    n_rings = len(couplings) - 1
    top = Netlist(name="crow")
    for k, kappa in enumerate(couplings):
        top.add(f"c{k}", "coupler", coupling=kappa)
    for r in range(1, n_rings + 1):
        top.add(f"r{r}", "ring", radius="=radius")
        top.connect((f"c{r - 1}", "o4"), (f"r{r}", "r1"))
        top.connect((f"r{r}", "r2"), (f"c{r}", "o3"))
        top.connect((f"c{r}", "o2"), (f"r{r}", "l1"))
        top.connect((f"r{r}", "l2"), (f"c{r - 1}", "o1"))
    top.expose("o1", ("c0", "o2")).expose("o2", ("c0", "o3"))
    last = f"c{n_rings}"
    if n_rings % 2:
        top.expose("o3", (last, "o1")).expose("o4", (last, "o4"))
    else:
        top.expose("o3", (last, "o4")).expose("o4", (last, "o1"))
    hnl = HierarchicalNetlist(top, {"ring": _ring_subcircuit()}, parameters={"radius": 10.0})
    wl = px.linspace(1.54, 1.56, 801)
    ref = px.circuit.crow(couplings=couplings, radius=8.0)(wl=wl)
    for mode in ("hierarchical", "flatten"):
        s = compile_hierarchical(hnl, MODELS, mode=mode)(wl=wl, params={"radius": 8.0})
        assert _max_diff(s, ref) <= TOL


def test_three_level_nesting_agrees_with_flat_solve():
    seg = Netlist()
    seg.add("wg", "straight", length="=2*len_half", neff=2.4, ng=4.2, loss_db_cm=3.0)
    seg.expose("o1", ("wg", "o1")).expose("o2", ("wg", "o2"))
    pair = Netlist()
    pair.add("s1", "seg", len_half="=L/2").add("ps", "phase_shifter", length=0.0, phase0="=phi")
    pair.add("s2", "seg", len_half="=L")
    pair.connect(("s1", "o2"), ("ps", "o1")).connect(("ps", "o2"), ("s2", "o1"))
    pair.expose("o1", ("s1", "o1")).expose("o2", ("s2", "o2"))
    mzi = Netlist()
    mzi.add("c1", "coupler", coupling=0.5).add("c2", "coupler", coupling=0.5)
    mzi.add("up", "pair", L="=L + dL", phi="=phi").add("lo", "pair", L="=L", phi=0.0)
    mzi.connect(("c1", "o4"), ("up", "o1")).connect(("up", "o2"), ("c2", "o1"))
    mzi.connect(("c1", "o3"), ("lo", "o1")).connect(("lo", "o2"), ("c2", "o2"))
    mzi.expose("o1", ("c1", "o1")).expose("o2", ("c1", "o2")).expose("o3", ("c2", "o3")).expose("o4", ("c2", "o4"))
    top = Netlist()
    top.add("m", "mzi3", L=15.0, dL=7.5, phi=0.3).add("fb", "straight", length=40.0, loss_db_cm=1.0)
    top.connect(("m", "o3"), ("fb", "o1")).connect(("fb", "o2"), ("m", "o2"))   # feedback loop through the MZI
    top.expose("in", ("m", "o1")).expose("out", ("m", "o4"))
    hnl = HierarchicalNetlist(top, {
        "seg": Subcircuit(seg, {"len_half": 5.0}),
        "pair": Subcircuit(pair, {"L": 10.0, "phi": 0.0}),
        "mzi3": Subcircuit(mzi, {"L": 10.0, "dL": 0.0, "phi": 0.0}),
    })
    flat = hnl.flatten()
    assert flat.settings["m.up.s2.wg"]["length"] == 2 * 22.5
    assert flat.settings["m.up.s1.wg"]["length"] == 22.5
    assert flat.settings["m.lo.s1.wg"]["length"] == 15.0
    wl = px.linspace(1.54, 1.56, 201)
    ref = compile_circuit(flat, MODELS, backend="dense")(wl=wl)
    for mode in ("hierarchical", "flatten"):
        for backend in ("dense", "sparse", "reduce"):
            s = compile_hierarchical(hnl, MODELS, mode=mode, backend=backend)(wl=wl)
            assert _max_diff(s, ref) <= TOL, (mode, backend)


@pytest.mark.parametrize("n", [3, 5])
def test_mesh_from_mzi_subcircuit_equals_builder(n):
    program = px.circuit.clements_decompose(unitary_group.rvs(n, random_state=11 + n))
    ref = px.circuit.mzi_mesh(program, splitter_coupling=0.47, phase_loss_db=0.1)(wl=1.55)
    hnl = _mesh_hnl(program, splitter_coupling=0.47, phase_loss_db=0.1)
    for mode in ("hierarchical", "flatten"):
        assert _max_diff(compile_hierarchical(hnl, MODELS, mode=mode)(wl=1.55), ref) <= TOL
    ideal = compile_hierarchical(_mesh_hnl(program), MODELS)(wl=1.55)
    u = unitary_group.rvs(n, random_state=11 + n)
    got = np.array([[complex(ideal[(f"o{i + 1}", f"o{2 * n - j}")]) for i in range(n)] for j in range(n)])
    assert np.max(np.abs(got - u)) <= 1e-12


def test_call_time_path_overrides_match_flat_overrides():
    program = px.circuit.clements_decompose(unitary_group.rvs(4, random_state=2))
    hnl = _mesh_hnl(program)
    flat_model = compile_circuit(hnl.flatten(), MODELS)
    f = compile_hierarchical(hnl, MODELS)
    s = f(wl=1.55, **{"mzi3.int": {"phase0": 0.4}, "mzi0": {"theta": 1.0}})
    ref = flat_model(wl=1.55, **{"mzi3.int": {"phase0": 0.4}, "mzi0.int": {"phase0": 1.0}})
    assert _max_diff(s, ref) <= TOL
    with pytest.raises(KeyError, match="unknown instance path"):
        f(wl=1.55, **{"mzi3.nothing": {"phase0": 0.4}})
    with pytest.raises(ValueError, match="undeclared parameter"):
        f(wl=1.55, mzi0={"radius": 1.0})


# --------------------------------------------------------------------------- #
# (2) Caching
# --------------------------------------------------------------------------- #
def _counting(models):
    calls: dict[str, int] = {}

    def wrap(name, fn):
        def model(**kw):
            calls[name] = calls.get(name, 0) + 1
            return fn(**kw)
        return model

    return {k: wrap(k, v) for k, v in models.items()}, calls


def test_mesh_16_evaluates_each_unique_block_once(monkeypatch):
    n = 16
    program = px.circuit.clements_decompose(unitary_group.rvs(n, random_state=5))
    assert program.n_mzis == 120
    models, calls = _counting(MODELS)
    solves = []
    real_solve = engine.solve
    monkeypatch.setattr(engine, "solve", lambda *a, **k: solves.append(1) or real_solve(*a, **k))
    s = compile_hierarchical(_mesh_hnl(program), models)(wl=1.55)
    assert calls["coupler"] == 1                                   # one unique coupler setting
    unique_phases = {p for _m, t, f in program.mzis for p in (t, f)} | set(map(float, program.output_phases))
    assert calls["attenuated_phase"] == len(unique_phases)
    unique_mzis = {(t, f) for _m, t, f in program.mzis}
    assert len(solves) == len(unique_mzis) + 1                     # one solve per unique MZI + the top
    u = unitary_group.rvs(n, random_state=5)
    assert abs(complex(s[("o1", f"o{2 * n}")]) - u[0, 0]) <= 1e-11

    # Identical MZIs: the subcircuit is solved once.
    same = px.circuit.MeshProgram(n=n, mzis=tuple((m, 0.7, 0.2) for m, _t, _f in program.mzis),
                                  output_phases=np.zeros(n))
    solves.clear()
    models, calls = _counting(MODELS)
    compile_hierarchical(_mesh_hnl(same), models)(wl=1.55)
    assert len(solves) == 2 and calls["coupler"] == 1 and calls["attenuated_phase"] == 3

    solves.clear()
    models, calls = _counting(MODELS)
    compile_hierarchical(_mesh_hnl(same), models, cache=False)(wl=1.55)
    assert len(solves) == 121 and calls["coupler"] == 240


def test_cache_respects_overrides_inside_a_subcircuit():
    same = px.circuit.MeshProgram(n=3, mzis=((0, 0.7, 0.2), (1, 0.7, 0.2), (0, 0.7, 0.2)),
                                  output_phases=np.zeros(3))
    hnl = _mesh_hnl(same)
    ov = {"mzi1.int": {"phase0": 2.0}}
    a = compile_hierarchical(hnl, MODELS)(wl=1.55, **ov)
    b = compile_hierarchical(hnl, MODELS, cache=False)(wl=1.55, **ov)
    c = compile_circuit(hnl.flatten(), MODELS)(wl=1.55, **ov)
    assert _max_diff(a, b) <= TOL and _max_diff(a, c) <= TOL


# --------------------------------------------------------------------------- #
# (3) Serialization
# --------------------------------------------------------------------------- #
def test_json_round_trip_keeps_expressions(tmp_path):
    hnl = _lattice_hnl((0.5, 0.4, 0.5), (20.0, 30.0), (0.0, 0.5))
    path = tmp_path / "lattice.json"
    text = hnl.to_json(path)
    back = HierarchicalNetlist.from_json(path)
    assert back == hnl
    assert HierarchicalNetlist.from_json(text) == hnl
    assert back.subcircuits["stage"].netlist.settings["top"]["length"] == "=length + dl"
    assert isinstance(load_netlist(path), HierarchicalNetlist)
    nested = HierarchicalNetlist(hnl.top, hnl.subcircuits, parameters={"a": "=2*b", "b": 1.0}, name="other")
    assert HierarchicalNetlist.from_dict(nested.to_dict()) == nested


def test_version_one_files_still_load():
    nl = Netlist(name="pair")
    nl.add("a", "straight", length=5.0).add("b", "straight")
    nl.connect(("a", "o2"), ("b", "o1"))
    nl.expose("in", ("a", "o1")).expose("out", ("b", "o2"))
    text = nl.to_json()
    hnl = HierarchicalNetlist.from_json(text)
    assert hnl.top == nl and hnl.subcircuits == {}
    assert load_netlist(text) == nl
    s1 = compile_hierarchical(hnl)(wl=1.55)
    s2 = compile_circuit(nl)(wl=1.55)
    assert _max_diff(s1, s2) <= TOL
    with pytest.raises(ValueError, match="hierarchical netlist"):
        Netlist.from_dict(_lattice_hnl((0.5, 0.5), (20.0,), (0.0,)).to_dict())


# --------------------------------------------------------------------------- #
# (4) Expressions
# --------------------------------------------------------------------------- #
def test_expression_values():
    assert evaluate_expression("=2*pi*radius", {"radius": 10}) == 62.83185307179586
    assert evaluate_expression("-(a + 1) ** 2 / 4 % 3", {"a": 1.0}) == (-(1.0 + 1) ** 2 / 4) % 3
    assert evaluate_expression("sqrt(x) + exp(0) + log(e) + abs(-2) + min(3, x, 5) + max(1, 2)",
                               {"x": 4.0}) == 2 + 1 + 1 + 2 + 3 + 2
    assert evaluate_expression("sin(pi/2) + cos(0) + tan(0)", {}) == 2.0
    assert isinstance(evaluate_expression("2**10", {}), float)


@pytest.mark.parametrize("bad", [
    "__import__('os')", "().__class__", "open(1)", "x.real", "x[0]", "[1, 2]", "lambda: 1", "'abc'",
    "True", "sqrt(x=1)", "y", "_x", "a if a else 1", "a == 1", "sqrt", "10**10**10", "1/0", "sqrt(-1)",
    "(1).__class__", "globals()", "*a",
])
def test_expression_injection_and_errors_raise_value_error(bad):
    with pytest.raises(ValueError):
        evaluate_expression(bad, {"x": 1.0, "a": 2.0, "_x": 1.0})


def test_parameter_defaults_can_depend_on_each_other():
    nl = Netlist()
    nl.add("wg", "straight", length="=circ")
    nl.expose("o1", ("wg", "o1")).expose("o2", ("wg", "o2"))
    top = Netlist()
    top.add("a", "blk", radius=2.0).add("b", "blk")
    top.connect(("a", "o2"), ("b", "o1"))
    top.expose("o1", ("a", "o1")).expose("o2", ("b", "o2"))
    sub = Subcircuit(nl, {"radius": 1.0, "circ": "=2*pi*radius"})
    flat = HierarchicalNetlist(top, {"blk": sub}).flatten()
    assert flat.settings["a.wg"]["length"] == 4 * math.pi and flat.settings["b.wg"]["length"] == 2 * math.pi
    cyc = Subcircuit(nl, {"radius": 1.0, "circ": "=2*x", "x": "=circ/2"})
    with pytest.raises(ValueError, match="cycle"):
        HierarchicalNetlist(top, {"blk": cyc}).flatten()


# --------------------------------------------------------------------------- #
# (5) Structural errors
# --------------------------------------------------------------------------- #
def _one(model, port="o1"):
    nl = Netlist()
    nl.add("x", model)
    nl.expose("o1", ("x", port))
    return nl


def test_cycle_reports_path():
    hnl = HierarchicalNetlist(_one("A"), {"A": Subcircuit(_one("B")), "B": Subcircuit(_one("C")),
                                          "C": Subcircuit(_one("A"))})
    with pytest.raises(ValueError, match="subcircuit cycle: A -> B -> C -> A"):
        hnl.validate()
    with pytest.raises(ValueError, match="cycle"):
        compile_hierarchical(hnl)
    selfloop = HierarchicalNetlist(_one("A"), {"A": Subcircuit(_one("A"))})
    with pytest.raises(ValueError, match="subcircuit cycle: A -> A"):
        selfloop.validate()


def test_unknown_subcircuit_port_lists_available_ports():
    sub = Subcircuit(_one("straight"))
    top = Netlist()
    top.add("a", "blk").add("b", "straight")
    top.connect(("a", "o2"), ("b", "o1"))
    top.expose("o1", ("a", "o1"))
    with pytest.raises(ValueError, match=r"not an exposed port of subcircuit 'blk'; available ports: \['o1'\]"):
        HierarchicalNetlist(top, {"blk": sub}).validate()


def test_unknown_model_and_undeclared_parameter():
    hnl = HierarchicalNetlist(_one("nonexistent_model"))
    hnl.validate()                                       # structural checks only
    with pytest.raises(ValueError, match="unknown model 'nonexistent_model'"):
        compile_hierarchical(hnl)
    top = _one("blk")
    top.set("x", radius=1.0)
    with pytest.raises(ValueError, match="undeclared parameter"):
        HierarchicalNetlist(top, {"blk": Subcircuit(_one("straight"))}).validate()
    with pytest.raises(ValueError, match="undeclared parameter"):
        compile_hierarchical(HierarchicalNetlist(_one("straight"), parameters={"r": 1.0}))(wl=1.55, params={"q": 1})


def test_unconnected_subcircuit_port_is_a_matched_load():
    """An exposed subcircuit port left open in the parent terminates like a flat open terminal."""
    hnl = HierarchicalNetlist(_one("mzi", "o1"), {"mzi": _mzi_subcircuit()})
    s = compile_hierarchical(hnl, MODELS)(wl=1.55)
    ref = compile_circuit(hnl.flatten(), MODELS)(wl=1.55)
    assert _max_diff(s, ref) <= TOL


# --------------------------------------------------------------------------- #
# (6) Gradients
# --------------------------------------------------------------------------- #
@requires_jax
def test_gradient_wrt_subcircuit_parameter_matches_flatten_and_fd():
    import jax

    top = Netlist()
    top.add("c0", "coupler", coupling=0.2).add("r", "ring", radius="=radius").add("c1", "coupler", coupling=0.2)
    top.connect(("c0", "o4"), ("r", "r1")).connect(("r", "r2"), ("c1", "o3"))
    top.connect(("c1", "o2"), ("r", "l1")).connect(("r", "l2"), ("c0", "o1"))
    top.expose("o1", ("c0", "o2")).expose("o2", ("c0", "o3")).expose("o3", ("c1", "o1")).expose("o4", ("c1", "o4"))
    hnl = HierarchicalNetlist(top, {"ring": _ring_subcircuit()}, parameters={"radius": 10.0})

    def fom(mode):
        f = compile_hierarchical(hnl, MODELS, mode=mode)

        def g(radius):
            s = f(wl=1.5502, params={"radius": radius})
            return px.xp.abs(s[("o1", "o3")]) ** 2
        return g

    r0 = 10.003
    g_h = float(jax.grad(fom("hierarchical"))(r0))
    g_f = float(jax.grad(fom("flatten"))(r0))
    fd = float(finite_diff(lambda r: float(fom("hierarchical")(r)), r0, h=1e-5))
    assert abs(g_h - g_f) <= 1e-9 * max(1.0, abs(g_h))
    assert abs(g_h - fd) <= 1e-5 * max(1.0, abs(g_h))
    jitted = jax.jit(fom("hierarchical"))
    assert abs(float(jitted(r0)) - float(fom("flatten")(r0))) <= 1e-12


def test_warnings_from_floating_parts_pass_through():
    top = Netlist()
    top.add("a", "straight").add("b", "straight")
    top.expose("o1", ("a", "o1")).expose("o2", ("a", "o2"))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        compile_hierarchical(top)(wl=1.55)
    assert any(issubclass(w.category, engine.FloatingSubcircuitWarning) for w in rec)


# --------------------------------------------------------------------------- #
# (7) Layout: hierarchical extraction flattens to extract_netlist
# --------------------------------------------------------------------------- #
def _layout_fixtures():
    import photonix.layout as lay
    from photonix.layout import components as lc
    from photonix.layout.library import fiber_array

    out = {}
    out["mzi_cell"] = lc.mzi(delta_length=37.0, length=15.0)

    chain = lay.Cell("chain")
    m = lc.mzi(delta_length=20.0)
    r0 = chain.add_ref(m, name="m0")
    r1 = chain.add_ref(m, name="m1").connect("o1", r0.port("o2"))
    r2 = chain.add_ref(lc.mzi(delta_length=10.0), name="m2").connect("o1", r1.port("o2"))
    chain.add_ref(lc.straight(12.0), name="tail").connect("o1", r2.port("o2"))
    chain.expose_port(r0, "o1", "in")
    out["mzi_chain"] = chain

    inner = lay.Cell("pair")
    a = inner.add_ref(lc.straight(5.0), name="a")
    inner.add_ref(lc.straight(7.0), name="b").connect("o1", a.port("o2"))
    nested = lay.Cell("top")
    nested.add_ref(inner, origin=(100.0, 50.0), rotation=90.0, name="p1")
    nested.add_port("in", (100.0, 50.0), 270.0)
    out["rotated_pair"] = nested

    # Three levels, mirrored and rotated placements, a leaf with undeclared open ports.
    pair = lay.Cell("pair2")
    pa = pair.add_ref(lc.straight(4.0), name="a")
    pb = pair.add_ref(lc.bend_circular(radius=5.0), name="b").connect("o1", pa.port("o2"))
    pair.expose_port(pa, "o1", "o1")
    pair.expose_port(pb, "o2", "o2")
    quad = lay.Cell("quad")
    q0 = quad.add_ref(pair, name="q0")
    q1 = quad.add_ref(pair, name="q1", mirror=True).connect("o1", q0.port("o2"))
    quad.add_ref(lc.directional_coupler(), name="dc").connect("o1", q1.port("o2"))
    quad.expose_port(q0, "o1", "o1")
    top3 = lay.Cell("top3")
    t0 = top3.add_ref(quad, name="u0", origin=(0.0, 0.0), rotation=180.0)
    t1 = top3.add_ref(quad, name="u1", origin=(60.0, 0.0))
    top3.add_ref(lay.route_manhattan(t0.port("o1"), t1.port("o1"), radius=5.0), name="r")
    top3.add_ref(lc.pad(), origin=(200, 200), name="pad")
    out["three_levels"] = top3

    out["fiber_array"] = fiber_array(4)
    heater = lay.Cell("heaters")
    heater.add_ref(lc.heater_straight(50.0), name="h")
    heater.add_ref(lc.pad(), origin=(0, 50), name="p")
    out["electrical"] = heater
    out["leaf_only"] = lc.straight(10.0)
    return out


@pytest.mark.parametrize("key", ["mzi_cell", "mzi_chain", "rotated_pair", "three_levels", "fiber_array",
                                 "electrical", "leaf_only"])
def test_extract_hierarchical_flattens_to_extract_netlist(key):
    from photonix.layout import extract_hierarchical_netlist, extract_netlist

    cell = _layout_fixtures()[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flat = extract_netlist(cell)
        hnl = extract_hierarchical_netlist(cell)
    assert hnl.flatten() == flat
    wl = px.linspace(1.53, 1.57, 9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = compile_circuit(flat)(wl=wl)
        for mode in ("hierarchical", "flatten"):
            assert _max_diff(compile_hierarchical(hnl, mode=mode)(wl=wl), ref) <= TOL


def test_extracted_hierarchy_reuses_one_subcircuit_per_cell():
    from photonix.layout import extract_hierarchical_netlist

    cells = _layout_fixtures()
    hnl = extract_hierarchical_netlist(cells["mzi_chain"])
    # m0 and m1 share one cell object; m2 is an equally named cell with other geometry.
    # Subcircuit names only avoid the leaf models the design uses, so the cell name is kept.
    assert hnl.top.instances["m0"] == hnl.top.instances["m1"] == "mzi"
    assert hnl.top.instances["m2"] == "mzi_1"
    assert hnl.subcircuits["mzi"].port_names == ["o1", "o2"]
    three = extract_hierarchical_netlist(cells["three_levels"])
    assert sorted(three.subcircuits) == ["pair2", "quad"]
    assert three.subcircuits["quad"].netlist.instances == {"q0": "pair2", "q1": "pair2", "dc": "evanescent_coupler"}
    # The coupler's undeclared open terminals become named subcircuit ports.
    assert set(three.subcircuits["quad"].port_names) == {"o1", "dc_o2", "dc_o3", "dc_o4"}
    assert HierarchicalNetlist.from_json(three.to_json()) == three

    # Equal cells from separate generator calls share one subcircuit; a cell
    # named like a leaf model it contains is renamed.
    import photonix.layout as lay
    from photonix.layout import components as lc

    def block():
        c = lay.Cell("straight")
        x = c.add_ref(lc.straight(3.0), name="x")
        c.expose_port(x, "o1", "o1").expose_port(x, "o2", "o2")
        return c

    top = lay.Cell("top")
    b0 = top.add_ref(block(), name="b0")
    top.add_ref(block(), name="b1").connect("o1", b0.port("o2"))
    hnl = extract_hierarchical_netlist(top)
    assert hnl.top.instances == {"b0": "straight_cell", "b1": "straight_cell"}
    assert list(hnl.subcircuits) == ["straight_cell"]
    assert hnl.flatten() == lay.extract_netlist(top)
