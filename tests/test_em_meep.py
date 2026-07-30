"""Tests for the Meep backend (:mod:`photonix.em.meep`).

The backend's import contract is that it **requires Meep**: ``from photonix.em
import meep`` raises ImportError (with an install hint) when Meep is absent, while
``import photonix.em`` keeps working. So the tests split in two:

* **Always-on contract tests** -- ``import photonix.em`` succeeds with or without
  Meep, and the Meep backend raises a helpful ImportError when Meep is missing.
* **Backend tests** -- guarded by ``skipif(not MEEP_PRESENT)``; they exercise the
  pure translation layer (unit conversions, MaterialGrid weights, grid<->coordinate
  mapping, the epsilon-lookup closure) and the Meep-requiring solvers (MPB n_eff vs
  the in-house FDE; a straight-waveguide FDTD transmission ~ 1).
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from photonix.em import meep

MEEP_PRESENT = importlib.util.find_spec("meep") is not None
needs_meep = pytest.mark.skipif(not MEEP_PRESENT, reason="Meep is not installed")
def _has_mpb() -> bool:
    if not MEEP_PRESENT:
        return False
    try:
        import meep.mpb  # noqa: F401
    except Exception:
        return False
    return True


needs_mpb = pytest.mark.skipif(not _has_mpb(), reason="Meep/MPB is not installed")

# --------------------------------------------------------------------------- #
# Always-on: import contract / graceful but loud degradation
# --------------------------------------------------------------------------- #
def test_em_imports_without_meep():
    # The rest of photonix never requires Meep.
    import photonix.em as em

    assert hasattr(em, "solve_modes_fullvector")


def test_meep_backend_is_import_safe_but_runtime_is_guarded():
    if MEEP_PRESENT:
        assert meep.HAS_MEEP
        return
    assert not meep.HAS_MEEP
    with pytest.raises(ImportError, match="conda"):
        meep.require_meep()


def _load_meep_backend_standalone(*names):
    """Load ``photonix.em.meep`` submodules without importing Meep.

    The package ``__init__`` calls ``require_meep()`` at import time, so the
    modules are loaded under a synthetic package instead. The MPB/Meep-facing
    entry points take ``mp``/``mpb`` as arguments (or never touch Meep at
    module level), which lets these Meep-free contract tests exercise the real
    backend code with stub objects.
    """
    import importlib.util
    import pathlib
    import sys
    import types

    import photonix

    pkgdir = pathlib.Path(photonix.__file__).parent / "em" / "meep"
    pkgname = "_photonix_meep_stub_pkg"
    pkg = types.ModuleType(pkgname)
    pkg.__path__ = [str(pkgdir)]
    sys.modules[pkgname] = pkg
    loaded = [pkgname]
    try:
        mods = {}
        for name in ("_guard",) + names:   # _guard first: others import it
            spec = importlib.util.spec_from_file_location(
                f"{pkgname}.{name}", pkgdir / f"{name}.py"
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"{pkgname}.{name}"] = mod
            loaded.append(f"{pkgname}.{name}")
            spec.loader.exec_module(mod)
            mods[name] = mod
        return mods
    finally:
        for k in loaded:
            sys.modules.pop(k, None)


def _load_modes_standalone():
    return _load_meep_backend_standalone("modes")["modes"]


class _StubV3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)


class _StubLattice:
    def __init__(self, size=None):
        self.size = size


class _StubMedium:
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon


class _StubBlock:
    def __init__(self, size=None, center=None, material=None):
        self.size, self.center, self.material = size, center, material


class _StubModeSolver:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _stub_mp_mpb():
    import types

    mp = types.SimpleNamespace(
        Vector3=_StubV3, Lattice=_StubLattice, Medium=_StubMedium, Block=_StubBlock
    )
    mpb = types.SimpleNamespace(ModeSolver=_StubModeSolver)
    return mp, mpb


def test_mpb_eps_grid_becomes_supported_sampled_material_not_materialgrid():
    """Integration contract with Meep's MPB (libpympb).

    MaterialGrid is not supported by MPB. The arbitrary grid is represented by
    a supported position-dependent material callback on a block spanning the
    lattice, with cell-centred (piecewise-constant) sampling.
    """
    modes = _load_modes_standalone()
    mp, mpb = _stub_mp_mpb()

    ny, nx = 4, 6
    eps = np.arange(ny * nx, dtype=float).reshape(ny, nx) + 2.0
    x = np.linspace(-1.5, 1.5, nx)
    y = np.linspace(-1.0, 1.0, ny)

    ms, gx, gy = modes._build_solver(
        mpb, mp, wl=1.55, width=0.5, thickness=0.22, n_core=3.4757, n_clad=1.444,
        margin=1.5, eps=eps, grid=(x, y), resolution=20, num_modes=2,
    )

    default = ms.kwargs["default_material"]
    assert isinstance(default, _StubMedium)
    (block,) = ms.kwargs["geometry"]
    assert callable(block.material)
    assert block.material(_StubV3(x[0], y[0])).epsilon == pytest.approx(eps[0, 0])
    assert block.material(_StubV3(x[-1], y[-1])).epsilon == pytest.approx(eps[-1, -1])
    lat = ms.kwargs["geometry_lattice"]
    assert lat.size.x == pytest.approx(nx * float(x[1] - x[0]))
    assert lat.size.y == pytest.approx(ny * float(y[1] - y[0]))
    assert ms.kwargs["num_bands"] == 2
    assert np.array_equal(gx, x) and np.array_equal(gy, y)


def test_parity_labels_match_photonix_field_families():
    """Cross-backend polarization contract (photonix labels, not Meep's).

    photonix's in-plane solvers define "te" as the out-of-plane-Ez scalar
    family and "tm" as the Hz family; the Meep wrapper must select the same
    physics for the same label (Meep's own TE/TM naming is the opposite), or
    cross-backend validation compares opposite polarizations.
    """
    import types

    mods = _load_meep_backend_standalone("materials", "geometry", "fdtd")
    fdtd = mods["fdtd"]
    mp = types.SimpleNamespace(EVEN_Z=1, ODD_Z=2)
    assert fdtd.parity_for(mp, "te") == mp.ODD_Z    # Ez family, like em.fdfd "te"
    assert fdtd.parity_for(mp, "ez") == mp.ODD_Z
    assert fdtd.parity_for(mp, "tm") == mp.EVEN_Z   # Hz family, like em.fdfd "tm"
    assert fdtd.parity_for(mp, "hz") == mp.EVEN_Z
    with pytest.raises(ValueError):
        fdtd.parity_for(mp, "circular")


def test_mpb_parametric_path_uses_block_and_medium():
    """The rectangle path must stay on Medium/Block (both MPB-supported)."""
    modes = _load_modes_standalone()
    mp, mpb = _stub_mp_mpb()

    ms, gx, gy = modes._build_solver(
        mpb, mp, wl=1.55, width=0.5, thickness=0.22, n_core=3.4757, n_clad=1.444,
        margin=1.5, eps=None, grid=None, resolution=20, num_modes=1,
    )

    (block,) = ms.kwargs["geometry"]
    assert isinstance(block, _StubBlock)
    assert isinstance(block.material, _StubMedium)
    assert block.material.epsilon == pytest.approx(3.4757**2)
    assert isinstance(ms.kwargs["default_material"], _StubMedium)
    assert ms.kwargs["default_material"].epsilon == pytest.approx(1.444**2)


# --------------------------------------------------------------------------- #
# Backend: unit bridge
# --------------------------------------------------------------------------- #
def test_meep_frequency_is_inverse_wavelength():
    assert meep.meep_frequency(1.55) == pytest.approx(1.0 / 1.55)
    with pytest.raises(ValueError):
        meep.meep_frequency(0)


def test_neff_k_roundtrip():
    f = meep.meep_frequency(1.55)
    k = meep.k_from_n_eff(2.45, f)
    assert meep.n_eff_from_k(k, f) == pytest.approx(2.45)
    assert k == pytest.approx(2.45 * f)


# --------------------------------------------------------------------------- #
# Backend: MaterialGrid weights
# --------------------------------------------------------------------------- #
def test_material_grid_weights_two_material():
    eps = np.array([[2.0, 12.0], [12.0, 2.0]])
    w, lo, hi = meep.material_grid_weights(eps)
    assert (lo, hi) == (2.0, 12.0)
    assert np.allclose(w, [[0.0, 1.0], [1.0, 0.0]])


def test_material_grid_weights_uniform_collapses():
    eps = np.full((3, 3), 5.3)
    w, lo, hi = meep.material_grid_weights(eps)
    assert lo == hi == pytest.approx(5.3)
    assert np.all(w == 0.0)


def test_material_grid_weights_linear_and_clipped():
    eps = np.array([[1.0, 2.0, 3.0]])
    w, lo, hi = meep.material_grid_weights(eps, eps_low=1.0, eps_high=3.0)
    assert np.allclose(w, [[0.0, 0.5, 1.0]])
    eps2 = np.array([[0.5, 4.0]])
    w2, _, _ = meep.material_grid_weights(eps2, eps_low=1.0, eps_high=3.0)
    assert np.allclose(w2, [[0.0, 1.0]])


def test_index_grid():
    assert np.allclose(meep.index_grid(np.array([4.0, 9.0])), [2.0, 3.0])
    with pytest.raises(ValueError):
        meep.index_grid(np.array([-1.0]))


# --------------------------------------------------------------------------- #
# Backend: geometry / coordinate mapping
# --------------------------------------------------------------------------- #
def test_cell_size_matches_pixel_count():
    eps = np.zeros((10, 20))
    sx, sy = meep.cell_size(eps, 0.05, 0.04)
    assert sx == pytest.approx(20 * 0.05)
    assert sy == pytest.approx(10 * 0.04)


def test_col_row_centered_on_origin():
    assert meep.col_to_x(0, 100, 0.05) == pytest.approx(-2.5 + 0.025)
    assert meep.col_to_x(99, 100, 0.05) == pytest.approx(2.5 - 0.025)
    assert meep.row_to_y(0, 80, 0.05) == pytest.approx(-2.0 + 0.025)


def test_devicegrid_properties():
    eps = np.ones((50, 120))
    d = meep.DeviceGrid(eps, 0.05, 0.025)
    assert d.shape == (50, 120)
    assert d.size == (pytest.approx(6.0), pytest.approx(1.25))
    assert d.resolution == pytest.approx(1.0 / 0.025)
    assert d.x_of_col(60) == pytest.approx(meep.col_to_x(60, 120, 0.05))


def test_epsilon_lookup_nearest_cell():
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 3)
    eps = np.arange(15, dtype=float).reshape(3, 5) + 1.0
    g = meep.epsilon_lookup(eps, x, y)
    assert g(-1.0, -1.0) == eps[0, 0]
    assert g(1.0, 1.0) == eps[2, 4]
    assert g(0.0, 0.0) == eps[1, 2]
    assert g(10.0, 10.0) == eps[2, 4]


def test_pml_padding_uses_ceil_and_zero_is_zero():
    mods = _load_meep_backend_standalone("materials", "geometry", "fdtd")
    eps = np.full((2, 3), 2.0)
    padded, npx, npy = mods["fdtd"].pad_for_pml(eps, 0.3, 0.4, 1.0)
    assert (npx, npy) == (4, 3)
    assert padded.shape == (2 + 2 * npy, 3 + 2 * npx)
    same, npx0, npy0 = mods["fdtd"].pad_for_pml(eps, 0.3, 0.4, 0.0)
    assert (npx0, npy0) == (0, 0)
    assert np.array_equal(same, eps) and same is not eps


def test_mpb_fields_match_native_yx_dominant_component():
    modes = _load_modes_standalone()

    class Solver:
        def get_efield(self, _band):
            field = np.zeros((3, 2, 1, 3), dtype=complex)
            field[:, :, 0, 0] = np.arange(6).reshape(3, 2)
            return field

    fields, fractions = modes._extract_fields(Solver(), 1, np.arange(3), np.arange(2))
    assert fields.shape == (1, 2, 3)
    assert np.array_equal(fields[0], np.arange(6).reshape(3, 2).T)
    assert fractions[0] == pytest.approx(1.0)


def test_full_two_port_assembly_uses_reverse_run(monkeypatch):
    mods = _load_meep_backend_standalone("materials", "geometry", "fdtd")
    fdtd = mods["fdtd"]
    calls = []

    def fake_one_way(grid, **kwargs):
        calls.append((np.asarray(grid).copy(), kwargs))
        return ((0.1 + 0.01j, 0.8 + 0.02j) if len(calls) == 1
                else (0.2 + 0.03j, 0.7 + 0.04j))

    monkeypatch.setattr(fdtd, "_one_way_sparams", fake_one_way)
    eps = np.arange(24, dtype=float).reshape(4, 6) + 2.0
    result = fdtd.waveguide_sparams(
        eps,
        dx=0.1,
        dy=0.1,
        wl=1.55,
        src_col=0,
        in_mon_col=1,
        out_mon_col=5,
    )
    assert result == {
        ("o1", "o1"): 0.1 + 0.01j,
        ("o1", "o2"): 0.8 + 0.02j,
        ("o2", "o2"): 0.2 + 0.03j,
        ("o2", "o1"): 0.7 + 0.04j,
    }
    assert np.array_equal(calls[1][0], eps[:, ::-1])


def test_waveguide_spectrum_preserves_wavelength_shape(monkeypatch):
    mods = _load_meep_backend_standalone("materials", "geometry", "fdtd")
    fdtd = mods["fdtd"]
    monkeypatch.setattr(
        fdtd,
        "_one_way_sparams",
        lambda _eps, **kwargs: (0j, complex(kwargs["wl"])),
    )
    wavelengths = np.array([[1.5, 1.55], [1.6, 1.65]])
    result = fdtd.waveguide_spectrum(
        np.ones((3, 5)),
        wavelengths=wavelengths,
        dx=0.1,
        dy=0.1,
        src_col=0,
        in_mon_col=1,
        out_mon_col=4,
        bidirectional=False,
    )
    assert result[("o1", "o2")].shape == wavelengths.shape
    assert np.allclose(result[("o1", "o2")].real, wavelengths)

    dataset = fdtd.waveguide_dataset(
        np.ones((3, 5)), wavelengths=np.array([1.5, 1.6]),
        dx=0.1, dy=0.1, src_col=0, in_mon_col=1, out_mon_col=4,
        bidirectional=False, metadata={"device": "straight"},
    )
    assert dataset.metadata["solver"] == "meep-fdtd"
    assert dataset.metadata["device"] == "straight"
    assert dataset.s.shape == (2, 2, 2)


def test_prepare_layout_centres_polygons_and_ports():
    from photonix.layout import Cell

    cell = Cell("straight")
    cell.add_polygon([(2, -0.25), (6, -0.25), (6, 0.25), (2, 0.25)], layer=(1, 0))
    cell.add_port("o1", center=(2, 0), orientation=180, width=0.5, layer=(1, 0))
    cell.add_port("o2", center=(6, 0), orientation=0, width=0.5, layer=(1, 0))
    cell.add_port("diag", center=(4, 0), orientation=45, width=0.5, layer=(1, 0))
    prepared = meep.prepare_layout(cell, [meep.LayerSpec((1, 0), epsilon=12.0)], margin=1.0)
    assert prepared.dimensions == 2
    assert prepared.origin == pytest.approx((4.0, 0.0))
    assert prepared.cell_size == pytest.approx((6.0, 2.5, 0.0))
    assert prepared.ports["o1"].center == pytest.approx((-2.0, 0.0))
    assert prepared.ports["o1"].outward_normal == pytest.approx((-1.0, 0.0))
    assert np.min(prepared.polygons[0].vertices[:, 0]) == pytest.approx(-2.0)
    with pytest.raises(ValueError, match="axis-aligned"):
        meep.port_region(prepared.ports["diag"])


def test_prepare_layout_builds_finite_3d_stack_metadata():
    import types

    import photonix.em.meep.layout as meep_layout
    from photonix.layout import Cell

    cell = Cell("stack").add_polygon([(0, 0), (1, 0), (1, 1)], layer=(1, 0))
    spec = meep.LayerSpec((1, 0), epsilon=4.0, thickness=0.22, z_center=10.0)
    prepared = meep.prepare_layout(cell, [spec], margin=(0.5, 0.5, 0.8))
    assert prepared.dimensions == 3
    assert prepared.cell_size == pytest.approx((2.0, 2.0, 1.82))
    assert prepared.z_origin == pytest.approx(10.0)

    class Prism:
        def __init__(self, vertices, height, axis, **kwargs):
            self.vertices, self.height, self.axis = vertices, height, axis
            self.kwargs = kwargs

    stub = types.SimpleNamespace(
        Vector3=_StubV3,
        Medium=_StubMedium,
        Prism=Prism,
        inf=float("inf"),
    )
    original = meep_layout.require_meep
    meep_layout.require_meep = lambda: stub
    try:
        built = meep_layout.build_layout_geometry(prepared)
    finally:
        meep_layout.require_meep = original
    prism = built.geometry[0]
    assert [(v.x, v.y) for v in prism.vertices] == [
        tuple(point) for point in prepared.polygons[0].vertices
    ]
    assert all(v.z == pytest.approx(-0.11) for v in prism.vertices)
    assert prism.height == pytest.approx(0.22)


# --------------------------------------------------------------------------- #
# Backend: Meep-requiring solvers
# --------------------------------------------------------------------------- #
@needs_mpb
def test_mpb_neff_matches_fde():
    import photonix.em as em

    n_mpb = meep.n_eff(width=0.5, thickness=0.22, resolution=32)
    n_fde = em.n_eff_fullvector(width=0.5, thickness=0.22, resolution=40)
    assert abs(n_mpb - n_fde) < 0.03
    assert 1.444 < n_mpb < 3.4757


@needs_meep
def test_meep_material_grid_constructs():
    eps = np.full((8, 8), 4.0)
    eps[2:6, 2:6] = 12.0
    grid = meep.to_material_grid(eps)
    assert tuple(grid.grid_size)[:2] == (8, 8)


@needs_mpb
def test_meep_fdtd_straight_waveguide_transmits():
    dx = dy = 0.025
    ny, nx = int(3.0 / dy), int(4.0 / dx)
    y = (np.arange(ny) - ny / 2 + 0.5) * dy
    eps = np.full((ny, nx), 1.444**2)
    eps[np.abs(y) < 0.25, :] = 3.4757**2
    s = meep.waveguide_sparams(
        eps, dx=dx, dy=dy, wl=1.55,
        src_col=int(0.5 / dx), in_mon_col=int(1.0 / dx),
        out_mon_col=nx - int(0.5 / dx), polarization="te",
    )
    t = abs(s[("o1", "o2")]) ** 2
    assert 0.9 < t <= 1.02
