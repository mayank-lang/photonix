"""Package-level coherence checks.

These guard structural invariants that unit tests of individual modules cannot
see: that there is one implementation per concept, that the documented
layout -> netlist -> simulation loop actually runs, and that a broken subpackage
fails loudly instead of quietly disappearing from `photonix.*`.
"""
from __future__ import annotations

import numpy as np
from conftest import requires_jax

import photonix as px
import photonix.em as em
import photonix.modes as modes
from photonix.circuit import Netlist


def test_no_duplicate_mode_solver():
    """`photonix.modes` must forward to `photonix.em`, not reimplement it.

    Two independent scalar solvers under the same names returned 2.644 vs 2.612
    for the same SOI strip. Identity -- not merely 'close enough' -- is the only
    check that cannot silently drift.
    """
    for name in ("solve_modes", "n_eff", "group_index", "rectangular_waveguide",
                 "silicon", "silica", "silicon_nitride"):
        assert getattr(modes, name) is getattr(em, name), name
    assert modes.ModeResult is em.ModeData
    assert modes.CrossSection is em.CrossSection


def test_em_is_self_sufficient():
    """`photonix.em` must not need `photonix.modes` for geometry or materials."""
    for name in ("geometry", "materials", "CrossSection", "rectangular_waveguide",
                 "Material", "silicon", "silica", "silicon_nitride"):
        assert hasattr(em, name), name


def test_all_subpackages_import_and_are_attached():
    """A subpackage must never silently vanish from the top-level namespace."""
    for name in ("core", "components", "circuit", "modes", "em", "pdk", "optim", "multiphysics"):
        assert hasattr(px, name), name
    assert px.UNAVAILABLE == {}, f"subpackages skipped: {px.UNAVAILABLE}"


def test_public_dir_is_clean():
    """`dir(photonix)` should show the documented API, not import machinery."""
    names = dir(px)
    assert "annotations" not in names
    assert set(names) == set(px.__all__)


def test_layout_to_circuit_round_trip():
    """The flow advertised in docs/ARCHITECTURE.md must run with no extra wiring.

    `circuit_from_netlist` used to require a models registry positionally, so
    this raised TypeError despite `components.MODELS` existing for the purpose.
    """
    from photonix.layout import Cell, extract_netlist
    from photonix.layout import components as lc

    top = Cell("top")
    top.add_ref(lc.straight(10.0), origin=(0, 0), name="a")
    top.add_ref(lc.straight(10.0), origin=(10, 0), name="b")

    nl = extract_netlist(top)
    assert isinstance(nl, Netlist)                       # never a bare dict
    assert nl.connections == {("a", "o2"): ("b", "o1")}

    s = px.circuit.circuit_from_netlist(nl)(wl=1.55)     # models default to MODELS
    assert abs(float(px.power(s[("a_o1", "b_o2")])) - 1.0) < 1e-9


def test_layout_cell_ports_match_component_models():
    """Extraction pairs layout cells with models by name, so ports must line up."""
    from photonix.layout import components as lc

    for cell_fn, model_name in ((lc.straight, "straight"), (lc.bend_circular, "bend"),
                                (lc.mmi1x2, "mmi1x2")):
        cell = cell_fn()
        assert cell.name == model_name, (cell.name, model_name)
        model_ports = set(px.core.ports_of(px.components.MODELS[model_name](wl=1.55)))
        assert set(cell.ports) <= model_ports, (model_name, set(cell.ports), model_ports)


def test_scalar_solver_overestimates_high_contrast_index():
    """Pins the guidance in the docs: use the full-vector solver for SOI strips."""
    kw = dict(wl=1.55, width=0.5, thickness=0.22, resolution=20)
    scalar = em.n_eff(**kw)
    fullvec = em.n_eff_fullvector(**kw)
    assert scalar > fullvec > 1.444
    assert 2.3 < fullvec < 2.55          # literature TE0 for 500x220 nm SOI


@requires_jax
def test_gradients_flow_through_the_whole_stack():
    """The package's central claim: one grad call across circuit + components."""
    wl = 1.55

    def fom(dl):
        return px.power(px.circuit.mzi(delta_length=dl)(wl=wl)[("o1", "o4")])

    g = float(px.grad(fom)(20.0))
    h = 1e-5
    fd = (float(fom(20.0 + h)) - float(fom(20.0 - h))) / (2 * h)
    assert abs(g - fd) / abs(fd) < 1e-6
    assert not np.isclose(g, 0.0)
