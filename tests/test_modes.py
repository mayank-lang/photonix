"""Tests for `photonix.modes` -- the compatibility facade over `photonix.em`.

The numerical validation of the solver itself (analytic-slab anchor, O(h^2)
convergence, differentiability) lives in ``test_em_fde.py``, against the
canonical implementation. Duplicating it here is what let the two solvers drift
apart in the first place, so this file tests the *facade contract* instead:
that `photonix.modes` really is `photonix.em`, plus the material models it
re-exports.
"""
from __future__ import annotations

import numpy as np

import photonix.em as em
import photonix.modes as m


def test_facade_forwards_to_em_not_a_copy():
    """The whole point of the facade: one implementation, two import paths.

    If someone reintroduces a second solver under `photonix.modes`, these
    identity checks fail immediately rather than silently returning a different
    n_eff under the same name.
    """
    assert m.n_eff is em.n_eff
    assert m.solve_modes is em.solve_modes
    assert m.group_index is em.group_index
    assert m.ModeResult is em.ModeData
    assert m.CrossSection is em.CrossSection
    assert m.rectangular_waveguide is em.rectangular_waveguide
    assert m.silicon is em.silicon


def test_mode_is_guided():
    ne = m.n_eff(wl=1.55, width=0.5, thickness=0.22, resolution=20)
    assert 1.444 < ne < 3.4757


def test_normal_dispersion():
    kw = dict(width=0.5, thickness=0.22, resolution=18, richardson=False)
    assert m.n_eff(wl=1.50, **kw) > m.n_eff(wl=1.60, **kw)


def test_group_index_exceeds_neff():
    kw = dict(width=0.5, thickness=0.22, resolution=18, richardson=False)
    assert m.group_index(wl=1.55, **kw) > m.n_eff(wl=1.55, **kw)


def test_solve_modes_result_shape():
    r = m.solve_modes(wl=1.55, width=0.5, thickness=0.22, resolution=20)
    assert r.fields.shape[0] == 1
    assert r.fields.shape[1:] == (len(r.y), len(r.x))
    assert r.wl == 1.55
    assert 1.444 < r.neff0 < 3.4757


def test_scalar_overestimates_fullvector_te():
    """Scalar FDE is an upper bound on the vectorial TE index for an SOI strip.

    Documented in `photonix.modes.solver`; asserted so the guidance to use
    `em.n_eff_fullvector` for high-contrast geometry stays true.
    """
    kw = dict(wl=1.55, width=0.5, thickness=0.22, resolution=20)
    assert m.n_eff(**kw) > em.n_eff_fullvector(**kw) > 1.444


def test_overlap_is_normalized():
    rng = np.random.default_rng(0)
    f = rng.random((12, 12))
    g = rng.random((12, 12))
    assert abs(m.overlap(f, f) - 1.0) < 1e-12
    assert 0.0 <= m.overlap(f, g) <= 1.0


def test_materials_literature_values():
    assert abs(float(m.silicon(1.55)) - 3.4757) < 3e-3
    assert abs(float(m.silica(1.55)) - 1.444) < 2e-3
    assert abs(float(m.silicon_nitride(1.55)) - 1.996) < 3e-3
