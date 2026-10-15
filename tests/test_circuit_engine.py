"""Validation of the scalable circuit engine (dense / sparse / reduce backends).

Evidence used here:

* an independent NumPy implementation of the original solver,
  ``M = (I - S Gamma)^{-1} S`` with an explicit ``Gamma`` matrix and all ``n``
  right-hand sides (``_reference``), on library circuits and on random
  non-reciprocal, lossy networks with reflections and matched loads;
* closed forms: Clements meshes reproducing Haar-random target unitaries,
  unitarity/reciprocity of lossless reciprocal circuits;
* gradients: the hand-written sparse adjoint against JAX's own autodiff of the
  dense solve, against central finite differences, and ``check_grads``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pytest
from conftest import finite_diff, requires_jax
from scipy.stats import unitary_group

import photonix as px
import photonix.components as comp
from photonix.circuit import Netlist, engine
from photonix.circuit.engine import (
    CircuitSingularError,
    FloatingSubcircuitWarning,
    compile_circuit,
    compile_topology,
    gather_values,
    resolve_backend,
    solve,
    solve_matrix,
)
from photonix.circuit.reduction import innerconnect, plan_reduction, reduce_network
from photonix.circuit.sparse_linalg import SingularSampleError, SparseExposedSolver, sparse_exposed_solve

BACKENDS = ("dense", "sparse", "reduce")


# --------------------------------------------------------------------------- #
# Independent reference and helpers
# --------------------------------------------------------------------------- #
def _reference(instances, connections, ports):
    """The original algorithm, written independently: all n RHS, explicit Gamma."""
    terminals = [(i, p) for i, sd in instances.items() for p in sorted({q for key in sd for q in key})]
    idx = {t: k for k, t in enumerate(terminals)}
    n = len(terminals)
    batch = np.broadcast_shapes(*[np.shape(v) for sd in instances.values() for v in sd.values()])
    S = np.zeros((*batch, n, n), complex)
    for inst, sd in instances.items():
        for (p_in, p_out), v in sd.items():
            S[..., idx[(inst, p_out)], idx[(inst, p_in)]] = np.asarray(v)
    G = np.zeros((n, n))
    for a, b in connections.items():
        G[idx[a], idx[b]] = G[idx[b], idx[a]] = 1.0
    M = np.linalg.solve(np.eye(n) - S @ G, S)
    return {(ei, eo): M[..., idx[to], idx[ti]] for ei, ti in ports.items() for eo, to in ports.items()}


def _max_diff(a, b):
    assert set(a) == set(b)
    return max(float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k])))) for k in a)


def _capture(monkeypatch, run):
    """Run ``run()`` and return the last (instances, connections, ports) the engine saw."""
    calls = []
    original = engine._TopologyCache.get

    def spy(self, instances, connections, ports):
        calls.append((dict(instances), dict(connections), dict(ports)))
        return original(self, instances, connections, ports)

    monkeypatch.setattr(engine._TopologyCache, "get", spy)
    try:
        run()
    finally:
        monkeypatch.undo()
    return calls[-1]


def _capture_netlist(monkeypatch, build):
    """Return (netlist, models) handed to compile_circuit by a circuit builder."""
    seen = {}
    original = engine.compile_circuit

    def spy(netlist, models=None, **kw):
        seen["nl"], seen["models"] = netlist, models
        return original(netlist, models, **kw)

    monkeypatch.setattr(engine, "compile_circuit", spy)
    try:
        build()
    finally:
        monkeypatch.undo()
    return seen["nl"], seen["models"]


def _random_network(seed, *, n_inst=7, max_ports=4, batch=(3,), reciprocal=False):
    """Random passive, (non-)reciprocal network with reflections and matched loads."""
    rng = np.random.default_rng(seed)
    instances = {}
    for i in range(n_inst):
        p = int(rng.integers(1, max_ports + 1))
        mask = rng.random((p, p)) < 0.7
        np.fill_diagonal(mask, True)                       # every port appears in a key
        M = (rng.normal(size=(*batch, p, p)) + 1j * rng.normal(size=(*batch, p, p))) * mask
        if reciprocal:
            M = 0.5 * (M + np.swapaxes(M, -1, -2))
            mask = mask | mask.T
        M = M / (1.25 * np.linalg.norm(M, 2, axis=(-2, -1))[..., None, None])
        names = [f"o{k + 1}" for k in range(p)]
        instances[f"i{i}"] = {(names[c], names[r]): M[..., r, c] for r in range(p) for c in range(p) if mask[r, c]}
    terminals = [(i, p) for i, sd in instances.items() for p in sorted({q for key in sd for q in key})]
    order = rng.permutation(len(terminals))
    n_conn = int(0.6 * len(terminals)) // 2
    connections = {terminals[order[2 * k]]: terminals[order[2 * k + 1]] for k in range(n_conn)}
    rest = [terminals[j] for j in order[2 * n_conn:]]
    n_exp = max(1, int(0.6 * len(rest)))
    ports = {f"e{k}": rest[k] for k in range(n_exp)}      # remaining terminals: matched loads
    return instances, connections, ports


def _mesh_matrix(s, n):
    return np.array([[np.asarray(s[(f"o{i + 1}", f"o{2 * n - j}")]) for i in range(n)] for j in range(n)])


# --------------------------------------------------------------------------- #
# (a) Agreement with the reference on library circuits
# --------------------------------------------------------------------------- #
WL201 = np.linspace(1.50, 1.60, 201)

CIRCUITS = {
    "mzi": lambda: px.circuit.mzi(delta_length=37.0, coupling=0.3, loss_db_cm=3.0)(wl=WL201),
    "ring": lambda: px.circuit.ring(radius=7.0, coupling=0.15, loss_db_cm=2.0)(wl=WL201),
    "lattice": lambda: px.circuit.lattice_filter(couplings=(0.3, 0.6, 0.2), delta_lengths=(15.0, 30.0),
                                                 phases=(0.4, -1.1), loss_db_cm=1.0)(wl=WL201),
    "sagnac": lambda: px.circuit.sagnac_mirror(coupling=0.3, loss_db_cm=5.0)(wl=WL201),
    "ring_double": lambda: comp.ring_double(wl=WL201, radius=8.0, gap=0.2, gap2=0.25),
    **{f"crow{k}": (lambda k=k: px.circuit.crow(couplings=(0.3,) + (0.05,) * (k - 1) + (0.25,),
                                               loss_db_cm=1.0)(wl=WL201)) for k in range(1, 6)},
}


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_backends_match_reference_on_library_circuits(name, monkeypatch):
    inst, conn, ports = _capture(monkeypatch, CIRCUITS[name])
    ref = _reference(inst, conn, ports)
    for backend in BACKENDS:
        out = px.circuit.evaluate_circuit(inst, conn, ports, backend=backend)
        assert _max_diff(out, ref) <= 1e-12, backend


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8])
def test_backends_match_reference_on_imperfect_meshes(n, monkeypatch):
    U = unitary_group.rvs(n, random_state=n)
    prog = px.circuit.clements_decompose(U)
    # The mesh model is wavelength independent (ideal couplers, zero-length
    # phase shifters), so a short sweep exercises the batch axis fully.
    inst, conn, ports = _capture(
        monkeypatch, lambda: px.circuit.mzi_mesh(prog, splitter_coupling=0.46, phase_loss_db=0.05)(
            wl=np.linspace(1.54, 1.56, 5)))
    ref = _reference(inst, conn, ports)
    for backend in BACKENDS:
        assert _max_diff(px.circuit.evaluate_circuit(inst, conn, ports, backend=backend), ref) <= 1e-12, backend


def _layout_mzi():
    import photonix.layout as lay

    return px.circuit.circuit_from_netlist(lay.extract_netlist(lay.components.mzi(delta_length=37.0, length=15.0)))(
        wl=WL201)


def _layout_route():
    import photonix.layout as lay
    from photonix.layout import components as lc

    top = lay.Cell("top")
    a = top.add_ref(lc.straight(10.0), name="a")
    b = top.add_ref(lc.straight(10.0), origin=(60.0, 30.0), name="b")
    top.add_ref(lay.route_manhattan(a.port("o2"), b.port("o1"), radius=5.0), name="r")
    return px.circuit.circuit_from_netlist(lay.extract_netlist(top))(wl=WL201)


@pytest.mark.parametrize("build", [_layout_mzi, _layout_route], ids=["mzi_layout", "routed_layout"])
def test_backends_match_reference_on_extracted_layouts(build, monkeypatch):
    inst, conn, ports = _capture(monkeypatch, build)
    ref = _reference(inst, conn, ports)
    for backend in BACKENDS:
        assert _max_diff(px.circuit.evaluate_circuit(inst, conn, ports, backend=backend), ref) <= 1e-12, backend


@pytest.mark.parametrize("seed", range(12))
def test_backends_match_reference_on_random_nonreciprocal_networks(seed):
    inst, conn, ports = _random_network(seed, reciprocal=False)
    ref = _reference(inst, conn, ports)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FloatingSubcircuitWarning)
        for backend in BACKENDS:
            out = px.circuit.evaluate_circuit(inst, conn, ports, backend=backend)
            assert _max_diff(out, ref) <= 1e-12, backend


def test_auto_backend_is_bit_compatible_with_the_original_dense_path(monkeypatch):
    inst, conn, ports = _capture(monkeypatch, CIRCUITS["crow3"])
    ref = _reference(inst, conn, ports)
    out = px.circuit.evaluate_circuit(inst, conn, ports)
    assert _max_diff(out, ref) <= 1e-13
    assert list(out) == list(ref)                          # same key order


# --------------------------------------------------------------------------- #
# (b) Large meshes reproduce Haar unitaries; speed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [16, 32])
def test_sparse_mesh_reproduces_haar_unitary(n):
    U = unitary_group.rvs(n, random_state=3)
    model = px.circuit.mzi_mesh(px.circuit.clements_decompose(U))
    s = model(wl=1.55)
    assert np.max(np.abs(_mesh_matrix(s, n) - U)) <= 1e-12


def test_16x16_mesh_engine_is_fast_and_matches_dense(monkeypatch):
    n = 16
    U = unitary_group.rvs(n, random_state=4)
    prog = px.circuit.clements_decompose(U)
    wl = np.linspace(1.54, 1.56, 5)
    inst, conn, ports = _capture(monkeypatch, lambda: px.circuit.mzi_mesh(prog)(wl=wl))
    t0 = time.perf_counter()
    topo = compile_topology(inst, conn, ports)
    s = solve(topo, gather_values(topo, inst), backend="sparse")
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"compile + 5-wavelength sparse solve took {elapsed:.2f} s"
    assert resolve_backend(topo, (5,)) == "sparse"
    assert np.max(np.abs(_mesh_matrix(s, n) - U[..., None])) <= 1e-12
    dense = solve(topo, gather_values(topo, {k: {kk: np.asarray(vv)[:1] if np.ndim(vv) else vv
                                                for kk, vv in sd.items()} for k, sd in inst.items()}),
                  backend="dense")
    assert max(float(np.max(np.abs(np.asarray(dense[k]) - np.asarray(s[k])[:1]))) for k in s) <= 1e-10


def test_32x32_mesh_sweeps_101_wavelengths():
    n = 32
    U = unitary_group.rvs(n, random_state=5)
    s = px.circuit.mzi_mesh(px.circuit.clements_decompose(U))(wl=np.linspace(1.5, 1.6, 101))
    S = _mesh_matrix(s, n)
    assert S.shape == (n, n, 101)
    assert np.max(np.abs(S - U[..., None])) <= 1e-12


# --------------------------------------------------------------------------- #
# (c) Gradients
# --------------------------------------------------------------------------- #
def test_numpy_adjoint_matches_finite_differences():
    """The hand-written transpose adjoint, checked without JAX."""
    inst, conn, ports = _random_network(21, batch=(2,))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FloatingSubcircuitWarning)
        topo = compile_topology(inst, conn, ports)
    solver = topo.sparse_solver()
    v = np.asarray(gather_values(topo, inst))
    rng = np.random.default_rng(0)
    y_bar = rng.normal(size=(2, topo.m, topo.m)) + 1j * rng.normal(size=(2, topo.m, topo.m))
    v_bar = solver.vjp_numpy(v, y_bar)
    h = 1e-6
    for j in rng.choice(topo.nnz, size=min(15, topo.nnz), replace=False):
        e = np.zeros(topo.nnz)
        e[j] = h
        dY = (solver.solve_numpy(v + e) - solver.solve_numpy(v - e)) / (2 * h)   # holomorphic derivative
        fd = np.sum(y_bar * dY, axis=(-2, -1))
        assert np.allclose(v_bar[:, j], fd, atol=1e-8, rtol=1e-7)


@requires_jax
@pytest.mark.parametrize("seed", [3, 8])
def test_sparse_and_reduce_vjp_match_jax_autodiff_of_dense(seed):
    import jax

    inst, conn, ports = _random_network(seed, batch=(2,))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FloatingSubcircuitWarning)
        topo = compile_topology(inst, conn, ports)
    v = gather_values(topo, inst)
    rng = np.random.default_rng(seed)
    ct = rng.normal(size=(2, topo.m, topo.m)) + 1j * rng.normal(size=(2, topo.m, topo.m))
    grads = {}
    for backend in BACKENDS:
        y, pullback = jax.vjp(lambda x, b=backend: solve_matrix(topo, x, backend=b), v)
        grads[backend] = np.asarray(pullback(ct)[0])
    for backend in ("sparse", "reduce"):
        assert np.max(np.abs(grads[backend] - grads["dense"])) <= 1e-10, backend


@pytest.fixture
def mesh4(monkeypatch):
    U = unitary_group.rvs(4, random_state=9)
    prog = px.circuit.clements_decompose(U)
    return _capture_netlist(monkeypatch, lambda: px.circuit.mzi_mesh(prog, splitter_coupling=0.48))


def _mesh_fom(model):
    def fom(x):
        s = model(wl=1.55, int2={"phase0": x[0]}, bs3a={"coupling": x[1]})
        return px.power(s[("o1", "o8")]) + 0.3 * px.xp.real(s[("o2", "o6")] * s[("o3", "o5")])
    return fom


@requires_jax
def test_setting_gradients_agree_across_backends_and_with_fd(mesh4):
    import jax

    nl, models = mesh4
    x0 = px.xp.asarray([0.7, 0.43])
    grads = {}
    for backend in BACKENDS:
        fom = _mesh_fom(compile_circuit(nl, models, backend=backend))
        grads[backend] = np.asarray(jax.grad(fom)(x0))
    for backend in ("sparse", "reduce"):
        assert np.max(np.abs(grads[backend] - grads["dense"])) <= 1e-10, backend
    fom = _mesh_fom(compile_circuit(nl, models, backend="sparse"))
    for i in range(2):
        e = np.zeros(2)
        e[i] = 1.0
        fd = finite_diff(lambda t, e=e: float(fom(x0 + t * e)), 0.0, h=1e-6)
        assert abs(grads["sparse"][i] - fd) <= 1e-6


@requires_jax
def test_check_grads_reverse_mode_through_sparse_engine(mesh4):
    from jax.test_util import check_grads

    nl, models = mesh4
    fom = _mesh_fom(compile_circuit(nl, models, backend="sparse"))
    check_grads(fom, (px.xp.asarray([0.7, 0.43]),), order=1, modes=["rev"], eps=1e-6)


@requires_jax
def test_sparse_engine_under_jit_and_vmap(mesh4):
    import jax

    nl, models = mesh4
    fom = _mesh_fom(compile_circuit(nl, models, backend="sparse"))
    xs = px.xp.asarray([[0.7, 0.43], [1.1, 0.5], [-0.3, 0.52]])
    eager = np.array([float(fom(x)) for x in xs])
    assert np.allclose(np.asarray(jax.jit(fom)(xs[0])), eager[0], atol=1e-13)
    assert np.allclose(np.asarray(jax.vmap(fom)(xs)), eager, atol=1e-13)
    g_vmap = np.asarray(jax.vmap(jax.grad(fom))(xs))
    g_loop = np.array([np.asarray(jax.grad(fom)(x)) for x in xs])
    assert np.allclose(g_vmap, g_loop, atol=1e-12)


@requires_jax
def test_forward_mode_through_every_backend_matches_reverse_sparse(mesh4):
    import jax

    nl, models = mesh4
    x0 = px.xp.asarray([0.7, 0.43])
    rev = np.asarray(jax.jacrev(_mesh_fom(compile_circuit(nl, models, backend="sparse")))(x0))
    for backend in BACKENDS:
        fwd = np.asarray(jax.jacfwd(_mesh_fom(compile_circuit(nl, models, backend=backend)))(x0))
        assert np.max(np.abs(fwd - rev)) <= 1e-10, backend


@requires_jax
@pytest.mark.parametrize("seed", [3, 8])
def test_sparse_jvp_and_second_order_match_dense(seed):
    """Regression: the sparse solve was custom_vjp, so jvp/jacfwd/hessian raised."""
    import jax

    inst, conn, ports = _random_network(seed, batch=(2,))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FloatingSubcircuitWarning)
        topo = compile_topology(inst, conn, ports)
    v = gather_values(topo, inst)
    rng = np.random.default_rng(seed)
    dv = px.xp.asarray(rng.normal(size=v.shape) + 1j * rng.normal(size=v.shape))
    w = px.xp.asarray(rng.normal(size=(2, topo.m, topo.m)))

    def fom(t, backend):            # real scalar of a real parameter: all AD modes apply
        Y = solve_matrix(topo, v + t * dv, backend=backend)
        return px.xp.sum(w * px.xp.real(Y * px.xp.conj(Y)))

    out = {}
    for backend in BACKENDS:
        tangent = jax.jvp(lambda x, b=backend: solve_matrix(topo, x, backend=b), (v,), (dv,))[1]
        out[backend] = (
            np.asarray(tangent),
            float(jax.jacfwd(fom)(0.3, backend)),
            float(jax.hessian(fom)(0.3, backend)),                  # forward over reverse
            float(jax.grad(jax.grad(fom))(0.3, backend)),           # reverse over reverse
            float(jax.jit(jax.jacfwd(jax.jacfwd(fom), argnums=0), static_argnums=1)(0.3, backend)),
        )
    ref = out["dense"]
    for backend in ("sparse", "reduce"):
        assert np.max(np.abs(out[backend][0] - ref[0])) <= 1e-10, backend
        for got, want in zip(out[backend][1:], ref[1:], strict=True):
            assert abs(got - want) <= 1e-9 * max(1.0, abs(want)), backend
    h = 1e-4
    fd2 = (float(fom(0.3 + h, "sparse")) - 2 * float(fom(0.3, "sparse")) + float(fom(0.3 - h, "sparse"))) / h**2
    assert abs(out["sparse"][2] - fd2) <= 1e-5 * max(1.0, abs(fd2))


@requires_jax
def test_auto_backend_supports_forward_mode_on_large_circuits():
    """Regression: ``auto`` picked the reverse-only sparse path above 300 terminals,
    breaking jax.jacfwd / jax.hessian (e.g. ``fosm(second_order=True)``)."""
    import jax

    N = 160                                                   # 320 terminals > DENSE_MAX_TERMINALS
    wl = px.xp.asarray([1.55, 1.56])

    def fom(neff, backend="auto"):
        inst = {f"w{i}": comp.straight(wl=wl, length=10.0, neff=neff + 1e-3 * i) for i in range(N)}
        conns = {(f"w{i}", "o2"): (f"w{i + 1}", "o1") for i in range(N - 1)}
        ports = {"in": ("w0", "o1"), "out": (f"w{N - 1}", "o2")}
        s = px.circuit.evaluate_circuit(inst, conns, ports, backend=backend)
        return px.xp.sum(px.xp.real(s[("in", "out")]) ** 2 + px.xp.imag(s[("in", "out")]))

    inst = {f"w{i}": comp.straight(wl=wl, length=10.0) for i in range(N)}
    topo = compile_topology(inst, {(f"w{i}", "o2"): (f"w{i + 1}", "o1") for i in range(N - 1)},
                            {"in": ("w0", "o1"), "out": (f"w{N - 1}", "o2")})
    assert resolve_backend(topo, (2,)) == "sparse"
    jf = float(jax.jacfwd(fom)(2.4))
    hs = float(jax.hessian(fom)(2.4))
    assert abs(jf - float(jax.grad(fom)(2.4))) <= 1e-9 * abs(jf)
    assert abs(jf - float(jax.jacfwd(lambda x: fom(x, "dense"))(2.4))) <= 1e-9 * abs(jf)
    assert abs(hs - float(jax.hessian(lambda x: fom(x, "dense"))(2.4))) <= 1e-9 * abs(hs)


@requires_jax
def test_sparse_derivatives_with_high_degree_rows_and_factor_reuse(monkeypatch):
    """The scatter fallback of the S Gamma tangent (very unequal row degrees) and the
    per-call LU reuse: one factorization per sample for a value-and-gradient."""
    import jax

    from photonix.circuit import sparse_linalg

    wl = px.xp.asarray([1.55, 1.56, 1.57])
    inst = {"star": comp.mmi_nxn(wl=wl, n=8), **{f"w{k}": comp.straight(wl=wl, length=5.0 + k) for k in range(8)}}
    conns = {("star", f"o{k + 9}"): (f"w{k}", "o1") for k in range(8)}
    ports = {**{f"in{k}": ("star", f"o{k + 1}") for k in range(8)}, **{f"out{k}": (f"w{k}", "o2") for k in range(8)}}
    topo = compile_topology(inst, conns, ports)
    v = gather_values(topo, inst)
    solver = topo.sparse_solver()
    assert solver._sg_layers is None                         # 8-entry rows next to 1-entry rows: scatter path
    dv = px.xp.ones_like(v) * 0.1j
    t_sparse = jax.jvp(lambda x: solve_matrix(topo, x, backend="sparse"), (v,), (dv,))[1]
    t_dense = jax.jvp(lambda x: solve_matrix(topo, x, backend="dense"), (v,), (dv,))[1]
    assert np.max(np.abs(np.asarray(t_sparse) - np.asarray(t_dense))) <= 1e-10

    calls = []
    original = sparse_linalg.SparseExposedSolver._factor
    monkeypatch.setattr(sparse_linalg.SparseExposedSolver, "_factor",
                        lambda self, row: calls.append(1) or original(self, row))
    fom = lambda x: px.xp.sum(px.xp.abs(solve_matrix(topo, x * v, backend="sparse")) ** 2)  # noqa: E731
    val, g = jax.value_and_grad(fom)(1.01)
    assert len(calls) == 3                                    # X solve and transposed solve share the factors
    assert np.isfinite(float(val)) and np.isfinite(float(g))


# --------------------------------------------------------------------------- #
# (d) Physical invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", BACKENDS)
def test_lossless_reciprocal_circuits_stay_unitary_and_reciprocal(backend, monkeypatch):
    builds = [
        lambda: px.circuit.mzi_mesh(px.circuit.clements_decompose(unitary_group.rvs(6, random_state=1)),
                                    splitter_coupling=0.4)(wl=np.linspace(1.54, 1.56, 3)),
        lambda: px.circuit.crow(couplings=(0.3, 0.05, 0.05, 0.3), loss_db_cm=0.0)(wl=np.linspace(1.549, 1.551, 51)),
        lambda: px.circuit.lattice_filter(couplings=(0.3, 0.5, 0.7), delta_lengths=(10.0, 20.0))(wl=WL201),
    ]
    for build in builds:
        inst, conn, ports = _capture(monkeypatch, build)
        report = px.analyze_sparameters(px.circuit.evaluate_circuit(inst, conn, ports, backend=backend))
        assert report.worst_unitarity_error < 1e-12
        assert report.worst_reciprocity_error < 1e-12


# --------------------------------------------------------------------------- #
# (e) Singular systems and floating subnetworks
# --------------------------------------------------------------------------- #
def _decoupled_loop():
    """A bus coupler with coupling=0 whose ring is a zero-phase, lossless loop."""
    cpl = comp.directional_coupler(coupling=0.0)
    loop = comp.phase_shifter(length=0.0, phase0=0.0)
    instances = {"c": cpl, "loop": loop}
    connections = {("c", "o3"): ("loop", "o1"), ("loop", "o2"): ("c", "o2")}
    ports = {"in": ("c", "o1"), "out": ("c", "o4")}
    return instances, connections, ports


@pytest.mark.parametrize("backend", BACKENDS)
def test_decoupled_lossless_resonance_raises(backend):
    with pytest.raises(CircuitSingularError, match="singular"):
        px.circuit.evaluate_circuit(*_decoupled_loop(), backend=backend)


def _unexcited_unit_loop():
    """A lossless unit-gain loop (loop.o1 -> o2 -> w -> loop.o1) that no exposed port excites."""
    loop = {("o1", "o2"): 1.0, ("o2", "o3"): 0.5, ("o3", "o1"): 0.5}
    w = {("o1", "o2"): 1.0, ("o2", "o1"): 1.0}
    m = {("o1", "o1"): 0.9, ("o2", "o1"): 0.1, ("o1", "o2"): 0.1}
    conns = {("loop", "o2"): ("w", "o1"), ("w", "o2"): ("loop", "o1"), ("loop", "o3"): ("m", "o1")}
    return {"loop": loop, "w": w, "m": m}, conns, {"out": ("m", "o2")}


@pytest.mark.parametrize("backend", BACKENDS)
def test_exactly_singular_system_raises_on_every_backend(backend):
    """Regression: dense + JAX returned a finite value (jnp.linalg.solve never fails)."""
    with pytest.raises(CircuitSingularError, match="singular") as info:
        px.circuit.evaluate_circuit(*_unexcited_unit_loop(), backend=backend)
    if backend != "dense" or px.HAS_JAX:                      # NumPy's LinAlgError names no sample
        assert info.value.samples == [0]


@requires_jax
def test_dense_singularity_check_is_skipped_inside_a_trace():
    import jax

    inst, conn, ports = _unexcited_unit_loop()
    topo = compile_topology(inst, conn, ports)
    Y = jax.jit(lambda x: solve_matrix(topo, x, backend="dense"))(gather_values(topo, inst))
    assert np.asarray(Y).shape == (1, 1)                       # no exception crosses the trace


def test_reduce_recovers_from_a_vanishing_intermediate_denominator():
    """Regression: a regular, non-passive circuit whose first growth step (loop + w) closes a
    unit-gain loop raised CircuitSingularError on the reduce backend."""
    loop = {("o1", "o2"): 1.0, ("o1", "o3"): 0.5, ("o3", "o2"): 0.5}
    w = {("o1", "o2"): 1.0}
    m = {("o1", "o1"): 0.9, ("o2", "o1"): 0.1, ("o1", "o2"): 0.1}
    inst = {"loop": loop, "w": w, "m": m}
    conns = {("loop", "o2"): ("w", "o1"), ("w", "o2"): ("loop", "o1"), ("loop", "o3"): ("m", "o1")}
    ports = {"out": ("m", "o2")}
    ref = _reference(inst, conns, ports)
    assert np.isclose(complex(ref[("out", "out")]), -1 / 90)
    for backend in BACKENDS:
        assert _max_diff(px.circuit.evaluate_circuit(inst, conns, ports, backend=backend), ref) <= 1e-12, backend


@requires_jax
def test_singular_sample_inside_jit_returns_nan_with_warning():
    import jax

    inst, conn, ports = _decoupled_loop()
    topo = compile_topology(inst, conn, ports)
    v = gather_values(topo, inst)
    with pytest.warns(RuntimeWarning, match="singular"):
        Y = jax.jit(lambda x: solve_matrix(topo, x, backend="sparse"))(v)
    assert np.all(np.isnan(np.asarray(Y)))


def test_numpy_solver_reports_singular_samples():
    inst, conn, ports = _decoupled_loop()
    topo = compile_topology(inst, conn, ports)
    with pytest.raises(SingularSampleError) as info:
        topo.sparse_solver().solve_numpy(np.asarray(gather_values(topo, inst)))
    assert info.value.samples == [0]


def test_floating_lossless_ring_is_pruned_with_a_warning():
    wl = np.linspace(1.54, 1.56, 11)
    bus = comp.straight(wl=wl, length=30.0)
    ring = comp.phase_shifter(length=0.0, phase0=0.0)          # exactly resonant, lossless
    instances = {"bus": bus, "ring": ring}
    connections = {("ring", "o1"): ("ring", "o2")}
    ports = {"in": ("bus", "o1"), "out": ("bus", "o2")}
    for backend in BACKENDS:
        engine.TOPOLOGY_CACHE.clear()
        with pytest.warns(FloatingSubcircuitWarning, match="ring"):
            s = px.circuit.evaluate_circuit(instances, connections, ports, backend=backend)
        assert np.max(np.abs(np.asarray(s[("in", "out")]) - np.asarray(bus[("o1", "o2")]))) <= 1e-15


# --------------------------------------------------------------------------- #
# (f) Memory of long sweeps
# --------------------------------------------------------------------------- #
def test_1000_wavelength_sweep_of_16x16_mesh_memory():
    code = (
        "import resource, numpy as np, photonix as px\n"
        "from scipy.stats import unitary_group\n"
        "U = unitary_group.rvs(16, random_state=1)\n"
        "f = px.circuit.mzi_mesh(px.circuit.clements_decompose(U))\n"
        "f(wl=np.linspace(1.5, 1.6, 3))\n"
        "base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024\n"
        "s = f(wl=np.linspace(1.5, 1.6, 1000))\n"
        "err = float(np.max(np.abs(np.asarray(s[('o1', 'o32')]) - U[0, 0])))\n"
        "peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024\n"
        "import json; print(json.dumps({'base': base, 'peak': peak, 'err': err}))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, env=os.environ)
    assert out.returncode == 0, out.stderr
    stats = json.loads(out.stdout.strip().splitlines()[-1])
    assert stats["err"] <= 1e-12
    assert stats["peak"] - stats["base"] < 500.0, stats


# --------------------------------------------------------------------------- #
# (g) Instance deduplication
# --------------------------------------------------------------------------- #
def test_dedupe_evaluates_each_unique_setting_once():
    calls = []

    def coupler(*, wl=1.55, coupling=0.5):
        calls.append(coupling)
        return comp.directional_coupler(wl=wl, coupling=coupling)

    nl = Netlist(name="chain")
    couplings = [0.5, 0.3, 0.5, 0.5, 0.3, 0.2]
    for k, kappa in enumerate(couplings):
        nl.add(f"c{k}", "coupler", coupling=kappa)
    for k in range(len(couplings) - 1):
        nl.connect((f"c{k}", "o4"), (f"c{k + 1}", "o1"))
        nl.connect((f"c{k}", "o3"), (f"c{k + 1}", "o2"))
    nl.expose("o1", ("c0", "o1")).expose("o2", ("c0", "o2"))
    last = len(couplings) - 1
    nl.expose("o3", (f"c{last}", "o3")).expose("o4", (f"c{last}", "o4"))
    models = {"coupler": coupler}
    s_dedupe = compile_circuit(nl, models)(wl=1.55)
    assert sorted(calls) == [0.2, 0.3, 0.5]
    calls.clear()
    s_plain = compile_circuit(nl, models, dedupe=False)(wl=1.55)
    assert len(calls) == len(couplings)
    assert _max_diff(s_dedupe, s_plain) == 0.0
    calls.clear()
    compile_circuit(nl, models)(wl=1.55, c2={"coupling": 0.3})   # override joins the 0.3 group
    assert sorted(calls) == [0.2, 0.3, 0.5]


def test_dedupe_distinguishes_mapping_keys_of_different_types():
    """Regression: dict settings were frozen with str(key), so {1: x} and {"1": x} collided."""
    calls = []

    def phase(*, wl=1.55, table=None):
        calls.append(table)
        t = np.exp(-1j * table.get(1, 0.0))
        return {("o1", "o2"): t, ("o2", "o1"): t}

    nl = Netlist(name="d")
    nl.add("a", "ph", table={1: 1.0})
    nl.add("b", "ph", table={"1": 1.0})                      # key "1" is not 1: phase 0
    nl.add("c", "ph", table={1: 1.0})                        # equal to a: shares its evaluation
    nl.connect(("a", "o2"), ("b", "o1")).connect(("b", "o2"), ("c", "o1"))
    nl.expose("in", ("a", "o1")).expose("out", ("c", "o2"))
    s = compile_circuit(nl, {"ph": phase})(wl=1.55)
    assert len(calls) == 2
    assert np.isclose(complex(s[("in", "out")]), np.exp(-2j))
    assert engine._freeze({1: 1.0, "x": 2}) == engine._freeze({"x": 2, 1: 1.0})


@requires_jax
def test_dedupe_keeps_gradients_per_instance():
    import jax

    nl = Netlist(name="two_phases")
    nl.add("a", "phase_shifter", length=0.0).add("b", "phase_shifter", length=0.0)
    nl.connect(("a", "o2"), ("b", "o1"))
    nl.expose("i", ("a", "o1")).expose("o", ("b", "o2"))
    f = compile_circuit(nl, backend="sparse")

    def fom(x):
        s = f(wl=1.55, a={"phase0": x[0]}, b={"phase0": x[1]})
        return px.xp.real(s[("i", "o")])

    x = px.xp.asarray([0.3, 0.3])                 # equal values, distinct tracers
    g = np.asarray(jax.grad(fom)(x))
    assert np.allclose(g, -np.sin(0.6), atol=1e-12)


# --------------------------------------------------------------------------- #
# Network reduction primitives
# --------------------------------------------------------------------------- #
def test_innerconnect_matches_global_solve():
    rng = np.random.default_rng(4)
    S = rng.normal(size=(3, 5, 5)) + 1j * rng.normal(size=(3, 5, 5))
    S /= 1.5 * np.linalg.norm(S, 2, axis=(-2, -1))[..., None, None]
    names = [f"o{k + 1}" for k in range(5)]
    sd = {(names[c], names[r]): S[..., r, c] for r in range(5) for c in range(5)}
    ref = _reference({"x": sd}, {("x", "o2"): ("x", "o4")},
                     {"a": ("x", "o1"), "b": ("x", "o3"), "c": ("x", "o5")})
    R = innerconnect(S, 1, 3)
    for k_in, e_in in enumerate("abc"):
        for i_out, e_out in enumerate("abc"):
            assert np.allclose(R[..., i_out, k_in], ref[(e_in, e_out)], atol=1e-14)


def test_reduce_network_orders_agree():
    inst, conn, ports = _random_network(5, reciprocal=True)
    blocks = {}
    for name, sd in inst.items():
        plist = sorted({q for key in sd for q in key})
        idx = {p: k for k, p in enumerate(plist)}
        S = np.zeros((3, len(plist), len(plist)), complex)
        for (p_in, p_out), v in sd.items():
            S[..., idx[p_out], idx[p_in]] = v
        blocks[name] = (S, plist)
    ref = _reference(inst, conn, ports)
    names = list(ports)
    for order in ("min_degree", "sequential"):
        S, terms = reduce_network(blocks, conn, list(ports.values()), order=order)
        assert terms == tuple(ports.values())
        for k, e_in in enumerate(names):
            for i, e_out in enumerate(names):
                assert np.allclose(S[..., i, k], ref[(e_in, e_out)], atol=1e-12), order


def test_min_degree_plan_bounds_block_size_on_a_long_cascade():
    n = 200
    block_ports = [[(f"w{k}", "o1"), (f"w{k}", "o2")] for k in range(n)]
    conns = [((f"w{k}", "o2"), (f"w{k + 1}", "o1")) for k in range(n - 1)]
    plan = plan_reduction(block_ports, conns, [("w0", "o1"), (f"w{n - 1}", "o2")])
    assert plan.max_block <= 4


@pytest.mark.parametrize("ordering", ["COLAMD", "MMD_AT_PLUS_A"])
def test_reused_fill_reducing_permutation_matches_natural_order(ordering):
    inst, conn, ports = _random_network(17, n_inst=12, batch=(4,))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FloatingSubcircuitWarning)
        topo = compile_topology(inst, conn, ports)
    v = np.asarray(gather_values(topo, inst))
    args = (topo.entry_rows, topo.entry_cols, topo.partner, topo.exposed, topo.n)
    natural = SparseExposedSolver(*args, ordering="NATURAL")
    permuted = SparseExposedSolver(*args, ordering=ordering)
    y_bar = np.random.default_rng(1).normal(size=(4, topo.m, topo.m)) + 0.5j
    assert np.allclose(permuted.solve_numpy(v), natural.solve_numpy(v), atol=1e-13)
    assert permuted.chosen_ordering == ordering and permuted._perm_plan[0] is not None
    assert np.allclose(permuted.vjp_numpy(v, y_bar), natural.vjp_numpy(v, y_bar), atol=1e-12)


def test_sparse_solver_functional_form_and_validation():
    t = np.exp(-0.25j)
    Y = sparse_exposed_solve([1, 0], [0, 1], np.array([t, 2 * t]), [-1, -1], [0, 1], 2)
    assert np.allclose(np.asarray(Y), [[0, 2 * t], [t, 0]])
    with pytest.raises(ValueError, match="unique"):
        SparseExposedSolver([0, 0], [1, 1], [-1, -1], [0], 2)
    with pytest.raises(ValueError, match="symmetric"):
        SparseExposedSolver([0], [1], [1, -1], [0], 2)


# --------------------------------------------------------------------------- #
# Contract of the routed public API
# --------------------------------------------------------------------------- #
def test_validation_errors_are_preserved():
    a = comp.straight(length=1.0)
    with pytest.raises(KeyError, match="unknown instance port"):
        px.circuit.evaluate_circuit({"a": a}, {("a", "o2"): ("b", "o1")}, {"x": ("a", "o1")})
    with pytest.raises(ValueError, match="multiple connections"):
        px.circuit.evaluate_circuit({"a": a, "b": a}, {("a", "o2"): ("b", "o1"), ("b", "o1"): ("a", "o1")},
                                    {"x": ("b", "o2")})
    with pytest.raises(ValueError, match="internally connected"):
        px.circuit.evaluate_circuit({"a": a, "b": a}, {("a", "o2"): ("b", "o1")}, {"x": ("a", "o2")})
    with pytest.raises(ValueError, match="exposed more than once"):
        px.circuit.evaluate_circuit({"a": a}, {}, {"x": ("a", "o1"), "y": ("a", "o1")})
    with pytest.raises(ValueError, match="unknown circuit backend"):
        px.circuit.evaluate_circuit({"a": a}, {}, {"x": ("a", "o1")}, backend="klu")
    assert px.circuit.evaluate_circuit({"a": a}, {}, {}) == {}


def test_topology_is_compiled_once_per_structure(monkeypatch):
    engine.TOPOLOGY_CACHE.clear()
    compiled = []
    original = engine.compile_topology

    def counting(*args, **kw):
        compiled.append(1)
        return original(*args, **kw)

    monkeypatch.setattr(engine, "compile_topology", counting)
    f = px.circuit.mzi(delta_length=10.0)
    for wl in (1.54, 1.55, np.linspace(1.5, 1.6, 7)):
        f(wl=wl)
    assert len(compiled) == 1
    for length in (1.0, 2.0):
        px.circuit.evaluate_circuit({"a": comp.straight(length=length)}, {}, {"x": ("a", "o1"), "y": ("a", "o2")})
    assert len(compiled) == 2


def test_auto_backend_selection():
    small = compile_topology({"w": [("o1", "o2"), ("o2", "o1")]}, {}, {"a": ("w", "o1"), "b": ("w", "o2")})
    assert resolve_backend(small, ()) == "dense"
    assert resolve_backend(small, (10**8,)) == "sparse"         # batched dense would not fit
    with pytest.raises(ValueError):
        resolve_backend(small, (), "superlu")


def test_numpy_backend_runs_every_engine_backend():
    code = (
        "import numpy as np, photonix as px\n"
        "from scipy.stats import unitary_group\n"
        "assert px.backend_name() == 'numpy'\n"
        "from photonix.circuit import engine\n"
        "U = unitary_group.rvs(4, random_state=2)\n"
        "s = px.circuit.mzi_mesh(px.circuit.clements_decompose(U))(wl=np.linspace(1.5, 1.6, 3))\n"
        "assert type(s[('o1', 'o8')]) is np.ndarray\n"
        "ref = None\n"
        "for b in ('dense', 'sparse', 'reduce'):\n"
        "    r = px.circuit.ring(radius=5.0, coupling=0.1)\n"
        "    out = px.circuit.evaluate_circuit({'c': px.components.directional_coupler(coupling=0.1),\n"
        "        'w': px.components.straight(wl=np.linspace(1.54, 1.56, 9), length=31.4)},\n"
        "        {('c', 'o3'): ('w', 'o1'), ('w', 'o2'): ('c', 'o2')}, {'i': ('c', 'o1'), 'o': ('c', 'o4')},\n"
        "        backend=b)\n"
        "    ref = out if ref is None else ref\n"
        "    assert max(np.max(np.abs(out[k] - ref[k])) for k in out) < 1e-13\n"
        "M = np.array([[s[(f'o{i+1}', f'o{8-j}')][0] for i in range(4)] for j in range(4)])\n"
        "assert np.max(np.abs(M - U)) < 1e-12\n"
        "print('ok')\n"
    )
    env = {**os.environ, "PHOTONIX_BACKEND": "numpy"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")
