"""Multimode / polarization-aware circuits: modal netlists, polarization and multimode models, analysis."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from conftest import finite_diff, requires_jax

import photonix as px
import photonix.components as comp
from photonix.analysis.polarization import (
    dgd,
    jones_from_sdict,
    jones_to_mueller,
    pdl_db,
    polarization_extinction_db,
    principal_states,
    stokes,
)
from photonix.circuit import (
    ModalNetlist,
    ModeMismatchWarning,
    Netlist,
    compile_circuit,
    eme_to_modal,
    lift_model,
    meep_dataset_to_modal,
    split_modal_port,
)
from photonix.core.dataset import SParameterDataset
from photonix.core.quality import analyze_sparameters

TOL = 1e-12


def _max_diff(a, b):
    assert set(a) == set(b)
    return max(float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k])))) for k in a)


def _te(s):
    """Single-mode view of a TE0-only modal SDict."""
    out = {}
    for (i, o), v in s.items():
        pi, mi = split_modal_port(i)
        po, mo = split_modal_port(o)
        assert mi == mo == "TE0"
        out[(pi, po)] = v
    return out


# --------------------------------------------------------------------------- #
# (1) Single-mode equivalence
# --------------------------------------------------------------------------- #
def _ring_netlist():
    nl = Netlist(name="ring_bus")
    nl.add("c", "coupler", coupling=0.15).add("fb", "straight", length=2 * math.pi * 10, loss_db_cm=3.0)
    nl.add("in_wg", "straight", length=7.0).add("out_wg", "straight", length=3.0)
    nl.connect(("c", "o4"), ("fb", "o1")).connect(("fb", "o2"), ("c", "o3"))
    nl.connect(("in_wg", "o2"), ("c", "o1")).connect(("c", "o2"), ("out_wg", "o1"))
    nl.expose("o1", ("in_wg", "o1")).expose("o2", ("out_wg", "o2"))
    return nl


def _netlists():
    from photonix.layout import components as lc
    from photonix.layout import extract_netlist

    out = {"ring": _ring_netlist(), "layout_mzi": extract_netlist(lc.mzi(delta_length=30.0))}
    lat = Netlist(name="lattice")
    for i, k in enumerate((0.5, 0.3, 0.5)):
        lat.add(f"c{i}", "coupler", coupling=k)
    for i, dl in enumerate((20.0, 40.0)):
        lat.add(f"t{i}", "straight", length=10 + dl).add(f"b{i}", "straight", length=10.0)
        lat.connect((f"c{i}", "o4"), (f"t{i}", "o1")).connect((f"t{i}", "o2"), (f"c{i + 1}", "o1"))
        lat.connect((f"c{i}", "o3"), (f"b{i}", "o1")).connect((f"b{i}", "o2"), (f"c{i + 1}", "o2"))
    lat.expose("o1", ("c0", "o1")).expose("o2", ("c0", "o2")).expose("o3", ("c2", "o3")).expose("o4", ("c2", "o4"))
    out["lattice"] = lat
    return out


@pytest.mark.parametrize("key", ["ring", "layout_mzi", "lattice"])
def test_single_mode_modal_netlist_reproduces_flat_circuit(key):
    nl = _netlists()[key]
    wl = px.linspace(1.53, 1.57, 301)
    ref = compile_circuit(nl)(wl=wl)
    s = ModalNetlist(nl).compile()(wl=wl)
    assert _max_diff(_te(s), ref) <= 1e-14
    explicit = {k: lift_model(v, ("TE0",)) for k, v in comp.MODELS.items()}
    s2 = ModalNetlist(nl).compile(explicit)(wl=wl)
    assert _max_diff(_te(s2), ref) <= 1e-14


def test_expand_gives_an_equivalent_plain_netlist():
    nl = _ring_netlist()
    mn = ModalNetlist(nl, modes={"fb": ("TE0", "TM0"), "c": ("TE0", "TM0")}, default_modes=("TE0", "TM0"))
    flat, models = mn.expand()
    assert set(flat.ports) == {"o1:TE0", "o1:TM0", "o2:TE0", "o2:TM0"}
    assert len(flat.connections) == 2 * len(nl.connections)
    wl = px.linspace(1.54, 1.56, 11)
    assert _max_diff(compile_circuit(flat, models)(wl=wl), mn.compile()(wl=wl)) <= TOL


# --------------------------------------------------------------------------- #
# (2) TE/TM ring: independent resonances, no polarization leakage
# --------------------------------------------------------------------------- #
def _resonance(m, neff, ng, wl0, L):
    """Exact root of m * wl = n(wl) * L with n(wl) = neff - (ng - neff)(wl - wl0)/wl0."""
    slope = (ng - neff) / wl0
    return (neff + slope * wl0) * L / (m + slope * L)


def test_te_tm_ring_resonates_per_polarization():
    radius, wl0 = 10.0, 1.55
    a = math.exp(-px.core.units.db_per_cm_to_alpha_um(2.0) * 2 * math.pi * radius)
    kappa = 1.0 - (a * (1 - 1e-4)) ** 2      # just over-coupled: a deep, finite notch
    pm = {"TE0": {"neff": 2.45, "ng": 4.2}, "TM0": {"neff": 1.75, "ng": 3.8}}
    ring2 = lift_model(comp.all_pass_ring, ("TE0", "TM0"), per_mode=pm)
    nl = Netlist()
    nl.add("a", "straight", length=5.0).add("r", "ring2", coupling=kappa, radius=radius, loss_db_cm=2.0)
    nl.add("b", "straight", length=5.0)
    nl.connect(("a", "o2"), ("r", "o1")).connect(("r", "o2"), ("b", "o1"))
    nl.expose("in", ("a", "o1")).expose("out", ("b", "o2"))
    models = {"straight": comp.straight, "ring2": ring2}
    f = ModalNetlist(nl, default_modes=("TE0", "TM0")).compile(models)
    L = 2 * math.pi * radius
    for mode, p in pm.items():
        m = round(p["neff"] * L / wl0)
        wl_res = _resonance(m, p["neff"], p["ng"], wl0, L)
        assert abs(wl_res - wl0) < 0.03
        s = f(wl=np.array([wl_res, wl_res + 0.002]))
        t = np.abs(np.asarray(s[(f"in:{mode}", f"out:{mode}")])) ** 2
        tt = math.sqrt(1.0 - kappa)
        assert t[0] == pytest.approx(((tt - a) / (1 - tt * a)) ** 2, abs=1e-12)     # exact notch depth
        assert t[0] < 2e-3 and t[1] > 100 * t[0]
    s = f(wl=px.linspace(1.54, 1.56, 101))
    assert float(np.max(np.abs(np.asarray(s[("in:TE0", "out:TM0")])))) == 0.0
    assert float(np.max(np.abs(np.asarray(s[("in:TM0", "out:TE0")])))) == 0.0


# --------------------------------------------------------------------------- #
# (3) Passivity and unitarity of polarization circuits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["polarization_rotator", "polarization_splitter_rotator",
                                  "polarization_beam_splitter", "grating_coupler_2d", "jones_fiber",
                                  "multimode_straight", "asymmetric_mode_converter", "mode_mux"])
def test_new_models_are_reciprocal_and_passive_over_parameters(name):
    rng = np.random.default_rng(0)
    fn = comp.MODELS[name]
    variants = [{}]
    if name == "polarization_splitter_rotator":
        variants += [dict(loss_db=0.0, conversion_loss_db=0.0, te_crosstalk_db=-3.0, tm_crosstalk_db=-15.0)]
    if name == "mode_mux":
        variants += [dict(n_modes=4, loss_db=0.0, crosstalk_db=-6.0)]
    if name == "polarization_rotator":
        variants += [dict(angle=float(rng.uniform(0, 3)), retardance=float(rng.uniform(0, 6)))]
    if name == "multimode_straight":
        variants += [dict(coupling_per_um=0.05, length=33.0)]
    for kw in variants:
        d = analyze_sparameters(fn(wl=1.55, **kw))
        assert d.passive and d.reciprocal, (name, kw)
        assert float(np.max(d.maximum_singular_value)) <= 1 + 1e-12


def _psr_chain(loss=0.0, xt=-20.0):
    nl = Netlist()
    nl.add("rot", "polarization_rotator", loss_db=loss)
    nl.add("psr", "polarization_splitter_rotator", loss_db=loss, conversion_loss_db=loss,
           te_crosstalk_db=xt, tm_crosstalk_db=xt)
    nl.add("arm2", "straight", length=12.0).add("arm3", "straight", length=40.0, loss_db_cm=0.0)
    nl.connect(("rot", "o2"), ("psr", "o1")).connect(("psr", "o2"), ("arm2", "o1"))
    nl.connect(("psr", "o3"), ("arm3", "o1"))
    nl.expose("in", ("rot", "o1")).expose("te_out", ("arm2", "o2")).expose("tm_out", ("arm3", "o2"))
    return ModalNetlist(nl)


def test_rotator_psr_chain_is_passive_and_unitary_when_lossless():
    wl = px.linspace(1.53, 1.57, 5)
    s = _psr_chain().compile()(wl=wl)
    assert set(s) >= {("in:TE0", "tm_out:TE0"), ("in:TM0", "te_out:TE0")}
    d = analyze_sparameters(s, lossless_atol=1e-12)
    assert d.passive and d.reciprocal and d.lossless
    assert float(np.max(d.unitarity_error)) <= 1e-12
    # The half-wave rotator swaps TE and TM, so TM input exits the TE arm.
    p = np.abs(np.asarray(s[("in:TM0", "te_out:TE0")])) ** 2
    assert np.allclose(p, 1 - 10 ** (-2.0), atol=1e-12)
    lossy = analyze_sparameters(_psr_chain(loss=0.4, xt=-25.0).compile()(wl=wl))
    assert lossy.passive and not lossy.lossless


def test_mode_mismatch_terminates_or_raises():
    nl = Netlist()
    nl.add("psr", "polarization_splitter_rotator").add("wg", "straight")
    nl.connect(("psr", "o2"), ("wg", "o1"))
    nl.expose("in", ("psr", "o1")).expose("out", ("wg", "o2")).expose("x", ("psr", "o3"))
    f = ModalNetlist(nl, modes={"wg": ("TE0", "TM0")}).compile()
    with pytest.warns(ModeMismatchWarning, match=r"\['TM0'\] only on wg.o1"):
        s = f(wl=1.55)
    assert ("in:TE0", "out:TM0") in s and abs(complex(s[("in:TE0", "out:TM0")])) == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # warned once per compiled model
        f(wl=1.56)
    with pytest.raises(ValueError, match="only on"):
        ModalNetlist(nl, modes={"wg": ("TE0", "TM0")}, on_mismatch="raise").compile()(wl=1.55)


def test_lift_model_coupling_matrix_is_applied_reciprocally():
    th = 0.3
    C = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    f = lift_model(comp.straight, ("TE0", "TM0"), per_mode={"TM0": {"neff": 1.9}}, coupling=lambda wl: C,
                   coupling_port="o2")
    s = f(wl=1.55, length=15.0)
    te = complex(comp.straight(wl=1.55, length=15.0)[("o1", "o2")])
    tm = complex(comp.straight(wl=1.55, length=15.0, neff=1.9)[("o1", "o2")])
    J = jones_from_sdict(s)
    assert np.max(np.abs(J - C @ np.diag([te, tm]))) <= TOL
    d = analyze_sparameters(s)
    assert d.reciprocal and d.lossless
    with pytest.raises(ValueError, match="coupling_port"):
        lift_model(comp.straight, ("TE0", "TM0"), coupling=C, coupling_port="o9")(wl=1.55)
    wl = px.linspace(1.5, 1.6, 3)
    sb = lift_model(comp.straight, ("TE0", "TM0"), coupling=lambda w: np.broadcast_to(C, (3, 2, 2)))(wl=wl)
    assert np.asarray(sb[("o1:TE0", "o2:TM0")]).shape == (3,)


# --------------------------------------------------------------------------- #
# (4) Asymmetric mode converter (coupled-mode theory)
# --------------------------------------------------------------------------- #
def test_mode_converter_full_and_detuned_conversion():
    kappa = 0.04
    s = comp.asymmetric_mode_converter(length=math.pi / (2 * kappa), kappa_per_um=kappa, delta_beta_per_um=0.0)
    assert abs(abs(complex(s[("o1:TE0", "o3:TE1")])) ** 2 - 1.0) <= TOL
    assert abs(complex(s[("o1:TE0", "o4:TE0")])) <= 1e-12
    for dbeta in (0.01, 0.05, 0.12):
        delta = dbeta / 2
        S = math.hypot(kappa, delta)
        peak = comp.asymmetric_mode_converter(length=math.pi / (2 * S), kappa_per_um=kappa, delta_beta_per_um=dbeta)
        assert abs(abs(complex(peak[("o1:TE0", "o3:TE1")])) ** 2 - kappa**2 / (kappa**2 + delta**2)) <= TOL
        d = analyze_sparameters(peak, lossless_atol=1e-12)
        assert d.lossless and d.reciprocal
    L = np.linspace(0, 80, 9)
    p = [abs(complex(comp.asymmetric_mode_converter(length=x, kappa_per_um=kappa, delta_beta_per_um=0.03)[
        ("o1:TE0", "o3:TE1")])) ** 2 for x in L]
    S = math.hypot(kappa, 0.015)
    assert np.allclose(p, kappa**2 / S**2 * np.sin(S * L) ** 2, atol=1e-12)


def test_multimode_straight_coupling_conserves_power_and_reduces_to_straights():
    wl = px.linspace(1.5, 1.6, 21)
    s = comp.multimode_straight(wl=wl, length=50.0, coupling_per_um=0.02)
    J = jones_from_sdict(s, modes=("TE0", "TE1"))
    gram = np.conj(np.swapaxes(J, -1, -2)) @ J
    assert np.max(np.abs(gram - np.eye(2))) <= 1e-12
    s0 = comp.multimode_straight(wl=wl, length=50.0, coupling_per_um=1e-300)       # coupled path, kappa -> 0
    s1 = comp.multimode_straight(wl=wl, length=50.0)
    for m in ("TE0", "TE1"):
        assert np.max(np.abs(np.asarray(s0[(f"o1:{m}", f"o2:{m}")]) - np.asarray(s1[(f"o1:{m}", f"o2:{m}")]))) <= TOL
    with pytest.raises(ValueError, match="exactly two modes"):
        comp.multimode_straight(modes={"TE0": {}, "TE1": {}, "TE2": {}}, coupling_per_um=0.1)


def test_mode_mux_routes_inputs_to_modes():
    s = comp.mode_mux(n_modes=3, loss_db=1.0, crosstalk_db=-20.0)
    T = np.array([[complex(s[(f"o{i + 1}:TE0", f"o4:TE{j}")]) for i in range(3)] for j in range(3)])
    sv = np.linalg.svd(T, compute_uv=False)
    a, x = 10 ** (-0.05), 0.01
    d = math.sqrt(1 - 2 * x)
    assert sorted(sv) == pytest.approx(sorted([a * math.hypot(d, math.sqrt(x))] * 2
                                              + [a * math.sqrt(d * d + 4 * x)]), abs=1e-12)


# --------------------------------------------------------------------------- #
# (5) Jones / Mueller / PDL
# --------------------------------------------------------------------------- #
def test_pdl_mueller_stokes():
    a, b = 0.9 * np.exp(0.3j), 0.4 * np.exp(-1.1j)
    assert abs(float(pdl_db(np.diag([a, b]))) - 20 * math.log10(abs(a) / abs(b))) <= 1e-12
    rng = np.random.default_rng(4)
    for _ in range(5):
        J = jones_from_sdict(comp.polarization_rotator(angle=rng.uniform(0, math.pi), retardance=rng.uniform(0, 6)))
        M = jones_to_mueller(J)
        assert abs(M[0, 0] - 1) <= 1e-12 and np.max(np.abs(M[0, 1:])) <= 1e-12
        R = M[1:, 1:]
        assert np.max(np.abs(R @ R.T - np.eye(3))) <= 1e-12 and abs(np.linalg.det(R) - 1) <= 1e-12
        G = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        E = rng.normal(size=2) + 1j * rng.normal(size=2)
        assert np.max(np.abs(stokes(G @ E) - jones_to_mueller(G) @ stokes(E))) <= 1e-12
        s = stokes(E)
        assert abs(s[0] ** 2 - np.sum(s[1:] ** 2)) <= 1e-12          # fully polarized
    hwp = jones_from_sdict(comp.polarization_rotator())
    assert np.allclose(np.abs(hwp), [[0, 1], [1, 0]], atol=1e-15)
    pbs = comp.polarization_beam_splitter(te_extinction_db=27.0, tm_extinction_db=27.0, loss_db=0.0)
    assert float(polarization_extinction_db(pbs, "o1", "o2")) == pytest.approx(
        10 * math.log10((1 - 10 ** -2.7) / 10 ** -2.7), abs=1e-9)


# --------------------------------------------------------------------------- #
# (6) DGD and PMD statistics
# --------------------------------------------------------------------------- #
def test_dgd_of_single_birefringent_element():
    tau = 3.7     # ps
    wl = np.array([1.55, 1.55002, 1.55004])
    w = 2 * np.pi * px.components.polarization.C_UM_PER_PS / wl
    th = 0.4
    R = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
    J = np.array([R.T @ np.diag([np.exp(-0.5j * x * tau), np.exp(0.5j * x * tau)]) @ R for x in w])
    assert np.allclose(dgd(wl, J), tau, rtol=0, atol=1e-9)
    psp = principal_states(wl, J)[0]
    axes = R.T                        # columns: slow/fast axes in the TE/TM basis
    overlap = np.abs(axes.conj().T @ psp)
    assert np.allclose(np.sort(overlap, axis=1), [[0, 1], [0, 1]], atol=1e-9)
    one = px.components.pmd_fiber_jones(wl, length_m=4000.0, pmd_ps_per_sqrt_km=0.5, n_sections=1, seed=9)
    assert np.allclose(dgd(wl, one), 0.5 * 2 / math.sqrt(8 / (3 * math.pi)), atol=1e-9)


def test_pmd_emulator_mean_dgd_is_maxwellian():
    wl = np.array([1.55, 1.550001])
    pmd, L = 0.2, 25_000.0
    J = px.components.pmd_fiber_jones(wl, length_m=L, pmd_ps_per_sqrt_km=pmd, n_sections=50, seed=range(2000))
    tau = dgd(wl, J)[:, 0]
    expected = pmd * math.sqrt(L / 1000)
    assert abs(tau.mean() / expected - 1) <= 0.03
    # Maxwellian second moment: <tau^2> = 3 pi / 8 <tau>^2.
    assert abs(np.mean(tau**2) / (3 * math.pi / 8 * expected**2) - 1) <= 0.06
    s = comp.jones_fiber(wl=wl, length_m=L, pmd_ps_per_sqrt_km=pmd, seed=17)
    assert analyze_sparameters(s).lossless


# --------------------------------------------------------------------------- #
# (7) Meep-style modal dataset cascaded with a lifted multimode straight
# --------------------------------------------------------------------------- #
def test_meep_style_dataset_cascades_with_multimode_straight():
    wl = np.array([1.54, 1.55, 1.56])
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    T = 0.9 * q                                      # forward modal transfer [out, in]
    ports = ("o1:m1", "o1:m2", "o2:m1", "o2:m2")
    S = np.zeros((3, 4, 4), complex)
    S[:, 2:, :2] = T                                 # s[:, out, in]
    S[:, :2, 2:] = T.T
    ds = meep_dataset_to_modal(SParameterDataset(wl, ports, S), {"m1": "TE0", "m2": "TE1"})
    assert ds.ports == ("o1:TE0", "o1:TE1", "o2:TE0", "o2:TE1")
    nl = Netlist()
    nl.add("dev", "dev").add("mm", "mm", length=30.0, coupling_per_um=0.01)
    nl.connect(("dev", "o2"), ("mm", "o1"))
    nl.expose("in", ("dev", "o1")).expose("out", ("mm", "o2"))
    f = ModalNetlist(nl).compile({"dev": ds, "mm": comp.multimode_straight})
    for w in wl:
        s = f(wl=float(w))
        got = jones_from_sdict(s, "in", "out", modes=("TE0", "TE1"))
        mm = jones_from_sdict(comp.multimode_straight(wl=float(w), length=30.0, coupling_per_um=0.01),
                              modes=("TE0", "TE1"))
        assert np.max(np.abs(got - mm @ T)) <= TOL


# --------------------------------------------------------------------------- #
# (8) EME: a taper cut at its midpoint and recombined equals one EME run
# --------------------------------------------------------------------------- #
def test_eme_taper_cut_and_recombined_through_modal_netlist():
    from photonix.em.eme import Section, eme_smatrix

    x = np.linspace(-3.0, 3.0, 241)
    dx = float(x[1] - x[0])
    widths = np.linspace(0.5, 1.4, 12)
    eps = [np.where(np.abs(x) < w / 2, 2.85**2, 1.444**2) for w in widths]
    seg = 1.5
    secs = [Section(e, seg) for e in eps]
    k = 6
    first = [*secs[:k], Section(eps[k], seg / 2)]
    second = [Section(eps[k], seg / 2), *secs[k + 1:]]
    nm = 4
    modes = tuple(f"TE{i}" for i in range(nm))
    full = eme_to_modal(eme_smatrix(secs, dx, 1.55, nm, pml=(0.8, 1.0)), modes_in=modes, modes_out=modes)
    a = eme_to_modal(eme_smatrix(first, dx, 1.55, nm, pml=(0.8, 1.0)), modes_in=modes, modes_out=modes)
    b = eme_to_modal(eme_smatrix(second, dx, 1.55, nm, pml=(0.8, 1.0)), modes_in=modes, modes_out=modes)
    nl = Netlist()
    nl.add("a", "a").add("b", "b")
    nl.connect(("a", "o2"), ("b", "o1"))
    nl.expose("o1", ("a", "o1")).expose("o2", ("b", "o2"))
    s = ModalNetlist(nl).compile({"a": lambda wl=1.55: a, "b": lambda wl=1.55: b})(wl=1.55)
    assert _max_diff(s, full) <= 1e-9
    assert 0.5 < abs(full[("o1:TE0", "o2:TE0")]) ** 2 <= 1.0


# --------------------------------------------------------------------------- #
# Gradients
# --------------------------------------------------------------------------- #
@requires_jax
def test_gradient_through_modal_netlist_matches_fd():
    import jax

    nl = Netlist()
    nl.add("conv", "asymmetric_mode_converter", length=25.0).add("wg", "multimode_straight", length=10.0)
    nl.connect(("conv", "o3"), ("wg", "o1"))
    nl.expose("bus", ("conv", "o1")).expose("mm", ("wg", "o2"))
    mn = ModalNetlist(nl, on_mismatch="raise")
    with pytest.raises(ValueError, match="only on"):
        mn.compile()(wl=1.55)
    f = ModalNetlist(nl).compile()

    def fom(kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ModeMismatchWarning)
            s = f(wl=1.55, conv={"kappa_per_um": kappa})
        return px.xp.abs(s[("bus:TE0", "mm:TE1")]) ** 2

    k0 = 0.031
    g = float(jax.grad(fom)(k0))
    fd = float(finite_diff(lambda k: float(fom(k)), k0, h=1e-6))
    assert abs(g - fd) <= 1e-6 * max(1.0, abs(g))
    assert abs(float(fom(k0)) - math.sin(k0 * 25.0) ** 2) <= 1e-12
