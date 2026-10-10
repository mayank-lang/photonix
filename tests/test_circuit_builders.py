"""Circuit builders (lattice, CROW, Sagnac, Clements mesh) and netlist serialization."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from conftest import finite_diff, requires_jax
from scipy.stats import unitary_group

import photonix as px
from photonix.circuit import (
    Netlist,
    clements_decompose,
    crow,
    lattice_filter,
    mesh_matrix,
    mzi_mesh,
    sagnac_mirror,
)


def _coupler_matrix(k):
    t, x = math.sqrt(1 - k), -1j * math.sqrt(k)
    return np.array([[t, x], [x, t]])   # (upper, lower) in -> (upper, lower) out


def test_lattice_matches_transfer_matrix_product():
    wl = np.linspace(1.53, 1.57, 9)
    ks, dls, phs = (0.5, 0.2, 0.7, 0.35), (15.0, 30.0, -12.0), (0.3, 0.0, 1.1)
    s = lattice_filter(couplings=ks, delta_lengths=dls, phases=phs, length=8.0)(wl=wl)
    n = np.asarray(px.components.neff_linear(wl, 2.4, 4.2, 1.55))
    for i, w in enumerate(wl):
        beta = 2 * np.pi * n[i] / w
        T = _coupler_matrix(ks[0])
        for k, dl, ph in zip(ks[1:], dls, phs, strict=True):
            delay = np.diag([np.exp(-1j * (beta * (8.0 + dl) + ph)), np.exp(-1j * beta * 8.0)])
            T = _coupler_matrix(k) @ delay @ T
        assert complex(s[("o1", "o4")][i]) == pytest.approx(T[0, 0], abs=1e-12)
        assert complex(s[("o1", "o3")][i]) == pytest.approx(T[1, 0], abs=1e-12)
        assert complex(s[("o2", "o4")][i]) == pytest.approx(T[0, 1], abs=1e-12)


def test_lattice_validates_lengths():
    with pytest.raises(ValueError, match="one more coupler"):
        lattice_filter(couplings=(0.5,), delta_lengths=(10.0,))


def test_single_ring_crow_is_the_analytic_add_drop_ring():
    wl = px.linspace(1.545, 1.555, 1001)
    s = crow(couplings=(0.15, 0.25), radius=10.0, loss_db_cm=3.0)(wl=wl)
    ref = px.components.add_drop_ring(wl=wl, coupling1=0.15, coupling2=0.25, radius=10.0, loss_db_cm=3.0)
    for key in (("o1", "o2"), ("o1", "o3"), ("o4", "o3"), ("o4", "o2")):
        assert np.allclose(np.asarray(px.power(s[key])), np.asarray(px.power(ref[key])), atol=1e-10)


@pytest.mark.parametrize("couplings", [(0.3, 0.05, 0.3), (0.4, 0.06, 0.04, 0.06, 0.4)])
def test_crow_lossless_energy_and_drop_side(couplings):
    wl = px.linspace(1.54, 1.56, 6001)
    s = crow(couplings=couplings, loss_db_cm=0.0)(wl=wl)
    thru = np.asarray(px.power(s[("o1", "o2")]))
    drop = np.asarray(px.power(s[("o1", "o3")]))
    assert np.allclose(thru + drop, 1.0, atol=1e-10)
    assert drop.max() > 0.99
    assert np.asarray(px.power(s[("o1", "o4")])).max() < 1e-20   # nothing exits the add port


def _shape_factor(wl, drop):
    peak = int(np.argmax(drop))
    win = np.abs(wl - wl[peak]) < 0.003
    w, d = wl[win], drop[win] / drop[peak]

    def width(level):
        above = w[d >= level]
        return above[-1] - above[0]

    return width(0.01) / width(0.5)


def test_higher_order_crow_has_sharper_skirts():
    wl = np.linspace(1.545, 1.555, 40001)
    one = np.asarray(px.power(crow(couplings=(0.1, 0.1), loss_db_cm=0.0)(wl=wl)[("o1", "o3")]))
    three = np.asarray(px.power(crow(couplings=(0.4, 0.06, 0.06, 0.4), loss_db_cm=0.0)(wl=wl)[("o1", "o3")]))
    assert _shape_factor(wl, three) < 0.5 * _shape_factor(wl, one)


@pytest.mark.parametrize("k", [0.1, 0.3, 0.5, 0.8])
def test_sagnac_mirror_reflection(k):
    s = sagnac_mirror(coupling=k)(wl=1.55)
    assert float(px.power(s[("o1", "o1")])) == pytest.approx(4 * k * (1 - k), abs=1e-12)
    assert float(px.power(s[("o1", "o2")])) == pytest.approx((1 - 2 * k) ** 2, abs=1e-12)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 7])
def test_clements_reconstructs_haar_unitaries(n):
    for seed in range(3):
        U = unitary_group.rvs(n, random_state=seed)
        prog = clements_decompose(U)
        assert prog.n_mzis == n * (n - 1) // 2
        assert np.allclose(mesh_matrix(prog), U, atol=1e-10)


@pytest.mark.parametrize("U", [np.eye(4), np.eye(4)[[1, 0, 3, 2]], np.eye(3)[[2, 0, 1]],
                               np.diag(np.exp(1j * np.array([0.3, -1.0, 2.0])))])
def test_clements_handles_zero_elements(U):
    assert np.allclose(mesh_matrix(clements_decompose(U)), U, atol=1e-10)


def test_clements_rejects_non_unitary():
    with pytest.raises(ValueError, match="not unitary"):
        clements_decompose(np.array([[1.0, 1.0], [0.0, 1.0]]))


@pytest.mark.parametrize("n", [3, 5])
def test_component_level_mesh_implements_the_unitary(n):
    U = unitary_group.rvs(n, random_state=11)
    s = mzi_mesh(clements_decompose(U))(wl=1.55)
    S = np.array([[complex(s[(f"o{i + 1}", f"o{2 * n - j}")]) for i in range(n)] for j in range(n)])
    assert np.allclose(S, U, atol=1e-10)


def test_imperfect_splitters_and_loss_degrade_fidelity():
    n = 4
    U = unitary_group.rvs(n, random_state=5)
    prog = clements_decompose(U)

    def realized(**kw):
        s = mzi_mesh(prog, **kw)(wl=1.55)
        return np.array([[complex(s[(f"o{i + 1}", f"o{2 * n - j}")]) for i in range(n)] for j in range(n)])

    def fidelity(S):
        return abs(np.trace(U.conj().T @ S)) ** 2 / (n * np.trace(S.conj().T @ S).real)

    assert fidelity(realized()) == pytest.approx(1.0, abs=1e-12)
    assert fidelity(realized(splitter_coupling=0.45)) < 0.999
    lossy = realized(phase_loss_db=0.1)
    assert np.max(np.linalg.svd(lossy, compute_uv=False)) < 1.0


@requires_jax
def test_mesh_gradient_wrt_phase_override():
    U = unitary_group.rvs(3, random_state=2)
    model = mzi_mesh(clements_decompose(U))

    def f(theta):
        return px.power(model(wl=1.55, int1={"phase0": theta})[("o1", "o6")])

    assert float(px.grad(f)(0.8)) == pytest.approx(float(finite_diff(f, 0.8, 1e-6)), rel=1e-6)


# --------------------------------------------------------------------------- #
# Netlist serialization
# --------------------------------------------------------------------------- #
def test_netlist_json_round_trip_simulates_identically(tmp_path):
    import photonix.layout as lay

    nl = lay.extract_netlist(lay.components.mzi(delta_length=12.0))
    path = tmp_path / "mzi.json"
    nl.to_json(path)
    loaded = Netlist.from_json(path)
    assert loaded == nl
    a = px.circuit.circuit_from_netlist(nl)(wl=1.551)
    b = px.circuit.circuit_from_netlist(loaded)(wl=1.551)
    assert complex(a[("o1", "o2")]) == pytest.approx(complex(b[("o1", "o2")]), abs=0)
    assert json.loads(path.read_text())["format"] == "photonix.netlist"


def test_netlist_serialization_converts_arrays_and_rejects_callables():
    nl = Netlist(name="g")
    nl.add("g", "bragg_grating", kappa=np.array([0.01, 0.02]), n_sections=np.int64(2))
    d = nl.to_dict()
    assert d["instances"]["g"]["settings"] == {"kappa": [0.01, 0.02], "n_sections": 2}
    nl.set("g", neff=lambda wl: 2.4)
    with pytest.raises(TypeError, match="callable"):
        nl.to_dict()
    with pytest.raises(ValueError, match="not a photonix netlist"):
        Netlist.from_dict({"format": "other"})
