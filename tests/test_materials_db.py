"""Validation of photonix.materials against analytic limits and independent sources.

Reference data in ``tests/data/refractiveindex_info/`` are verbatim entries of the
refractiveindex.info database (public domain, CC0 1.0). They are *independent
transcriptions* of the cited papers: the library coefficients were entered from
the original publications (or, for Li 1980 and Rakic 1998, from the papers'
formulas/parameter tables), so agreement checks the transcription, the formula
algebra and the evaluation code together.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from conftest import finite_diff, requires_jax

import photonix as px
import photonix.em as em
import photonix.materials as pm
from photonix.materials import (
    MaterialRangeError,
    MaterialRangeWarning,
    MissingPropertyError,
    get_material,
)
from photonix.materials.io import DATA_DIR

REF = Path(__file__).parent / "data" / "refractiveindex_info"


def ref_model(name: str, *, to_vacuum: bool = True) -> pm.OpticalModel:
    return pm.load_refractiveindex_yaml(REF / name, to_vacuum=to_vacuum)


@pytest.fixture(autouse=True)
def _no_range_warnings_by_default():
    """Every test runs with range warnings as errors unless it expects them explicitly."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", MaterialRangeWarning)
        yield


def re(x) -> np.ndarray:
    return np.asarray(x).real


# =========================================================================== #
# 1. Backward compatibility: em.materials stays the canonical Sellmeier home
# =========================================================================== #
@pytest.mark.parametrize("name, fn, lo, hi", [
    ("Si", em.silicon, 1.36, 11.0),
    ("Si3N4", em.silicon_nitride, 0.31, 5.5),
])
def test_library_wraps_em_sellmeier_bit_for_bit(name, fn, lo, hi):
    spec = get_material(name)
    assert spec.optical.coeffs["fn"] is fn                    # referenced, not copied
    wl = np.linspace(lo, hi, 257)
    np.testing.assert_array_equal(re(spec.index(wl)), np.asarray(fn(wl)))
    np.testing.assert_array_equal(np.asarray(spec(wl)), np.asarray(fn(wl)))   # LayerLevel callable
    t_ref = spec.reference_temperature_k                       # explicit T_ref adds exactly zero
    np.testing.assert_array_equal(re(spec.index(wl, t_ref)), np.asarray(fn(wl)))


def test_sio2_is_em_silica_converted_from_air_relative_to_absolute():
    """Malitson (1965) is relative to air at 20 degC on an air-wavelength axis: N = N_rel(l/n_air) n_air."""
    spec = get_material("SiO2")
    assert spec.optical.coeffs["fn"] is em.silica              # still referenced, not copied
    wl = np.linspace(0.25, 6.5, 257)
    n_air = np.asarray(pm.formulas.ciddor1996_air(wl, 293.15))
    expected = np.asarray(em.silica(wl / n_air)) * n_air
    np.testing.assert_allclose(re(spec.index(wl)), expected, rtol=0, atol=1e-15)
    shift = float(spec.index(1.55).real) - float(em.silica(1.55))
    assert shift == pytest.approx(3.93e-4, abs=1e-5)          # ~ (n_air - 1) n
    # the Malitson reference file of the database, converted by the YAML loader, agrees
    np.testing.assert_allclose(re(spec.index(wl)), re(ref_model("SiO2_Malitson.yml").n(wl)), rtol=0, atol=1e-12)


def test_materials_subpackage_is_attached():
    assert px.materials is pm
    assert "materials" in px.__all__
    # em keeps its own API untouched
    for name in ("silicon", "silica", "silicon_nitride", "Material"):
        assert hasattr(em, name)


# =========================================================================== #
# 2. Li (1980) n(wl, T) against Li's own tabulated recommended values
# =========================================================================== #
@pytest.mark.parametrize("temp, tol", [(100, 1e-4), (200, 1e-4), (293, 1e-4), (350, 1e-4), (450, 1e-4),
                                       (600, 1.5e-4)])
def test_li1980_silicon_reproduces_li_tables(temp, tol):
    tab = ref_model(f"Si_Li-{temp}K.yml").coeffs
    n = np.asarray(pm.li1980_silicon(tab["wl"], float(temp)))
    assert np.max(np.abs(n - tab["n"])) < tol                  # tables are rounded to 1e-4


@pytest.mark.parametrize("temp", [100, 200, 293, 400, 500])
def test_li1980_germanium_reproduces_li_tables(temp):
    tab = ref_model(f"Ge_Li-{temp}K.yml").coeffs
    n = np.asarray(pm.li1980_germanium(tab["wl"], float(temp)))
    assert np.max(np.abs(n - tab["n"])) < 1e-4


def test_silicon_thermo_optic_coefficient():
    """d n_Si / dT at 1.55 um, 300 K: Li formula vs Li's tables vs the photonix active-device constant."""
    dndt = float(get_material("Si").dn_dt(1.55, 300.0))
    # independent of the formula: a finite difference of Li's tabulated values at 293 and 350 K
    n293 = np.interp(1.55, *[ref_model("Si_Li-293K.yml").coeffs[k] for k in ("wl", "n")])
    n350 = np.interp(1.55, *[ref_model("Si_Li-350K.yml").coeffs[k] for k in ("wl", "n")])
    from_tables = (n350 - n293) / 57.0
    dndt_mid = float(get_material("Si_Li1980").dn_dt(1.55, 321.5))
    assert abs(dndt_mid - from_tables) / from_tables < 0.03
    # DN_DT_SI = 1.86e-4 /K is the bulk value used by the thermo-optic phase-shifter model
    assert abs(dndt - 1.86e-4) / 1.86e-4 < 0.05
    dn_dt_si = getattr(px.components, "DN_DT_SI", None)
    if dn_dt_si is not None:
        assert abs(dndt - dn_dt_si) / dn_dt_si < 0.05
    # both silicon entries share the Li temperature model
    assert float(get_material("Si_Li1980").dn_dt(1.55, 300.0)) == pytest.approx(dndt, rel=1e-9)


def test_li1980_silicon_is_continuous_at_its_293k_reference():
    """Regression: the two dL/L_293 branches did not meet at 293 K (jump 7.5e-7 in n), so
    the central-difference dn/dT at the default temperature was ~4 % too large."""
    wl = np.array([1.31, 1.55, 5.0])
    jump = np.asarray(pm.li1980_silicon(wl, 293.0 + 1e-9)) - np.asarray(pm.li1980_silicon(wl, 293.0))
    assert np.max(np.abs(jump)) < 1e-11
    # dL/L_293 vanishes at 293 K by definition: L(293) = 1, n^2 = eps + (A0 + A1 T + A2 T^2) / l^2
    t = 293.0
    eps = 11.4445 + 2.7739e-4 * t + 1.7050e-6 * t**2 - 8.1347e-10 * t**3
    n_ref = np.sqrt(eps + (0.8948 + 4.3977e-4 * t + 7.3835e-8 * t**2) / wl**2)
    assert np.max(np.abs(np.asarray(pm.li1980_silicon(wl, t)) - n_ref)) < 1e-14
    # dn/dT at 293 K: the mean of the two one-sided derivatives (one per branch)
    h = 1e-3
    n0 = float(pm.li1980_silicon(1.55, t))
    one_sided = 0.5 * ((float(pm.li1980_silicon(1.55, t + h)) - n0) + (n0 - float(pm.li1980_silicon(1.55, t - h)))) / h
    si = get_material("Si_Li1980")
    assert float(si.dn_dt(1.55)) == pytest.approx(one_sided, rel=1e-5)
    assert float(si.dn_dt(1.55)) == pytest.approx(1.8186e-4, rel=2e-4)
    # and it lies between its neighbours (no spike at the branch point)
    assert float(si.dn_dt(1.55, 292.5)) < float(si.dn_dt(1.55)) < float(si.dn_dt(1.55, 293.5))


def test_germanium_li_agrees_with_nist_burnett_2016():
    """Cross-source: Li (1980) at 22 degC vs Burnett et al. (NIST, 2016; uncertainty 2-8e-5) for 2.5-12 um."""
    wl = np.linspace(2.5, 12.0, 120)
    # The published numbers as stored (Burnett's file is relative to air; Li's compilation states no
    # medium), so the ~1e-3 air correction does not enter this source-to-source comparison.
    burnett = re(ref_model("Ge_Burnett.yml", to_vacuum=False).n(wl))
    li = re(get_material("Ge_Li1980").index(wl, 295.15))
    assert np.max(np.abs(li - burnett)) < 1e-3


def test_silicon_sources_scatter_is_documented():
    """Salzberg-Villa (em.silicon, 26 degC) and Li (1980) at 299 K differ by < 2e-3 over 1.4-11 um."""
    wl = np.linspace(1.4, 11.0, 200)
    diff = re(get_material("Si").index(wl)) - re(get_material("Si_Li1980").index(wl, 299.15))
    assert np.max(np.abs(diff)) < 2e-3


# =========================================================================== #
# 3. Library anchors against independent transcriptions / sources
# =========================================================================== #
@pytest.mark.parametrize("material, axis, ref_file, wl_lo, wl_hi", [
    ("LiNbO3", "o", "LiNbO3_Zelmon-o.yml", 0.4, 5.0),
    ("LiNbO3", "e", "LiNbO3_Zelmon-e.yml", 0.4, 5.0),
    ("Al2O3", "o", "Al2O3_Malitson-o.yml", 0.2, 5.0),
    ("Al2O3", "e", "Al2O3_Malitson-e.yml", 0.2, 5.0),
    ("TiO2", "o", "TiO2_Devore-o.yml", 0.43, 1.53),
    ("TiO2", "e", "TiO2_Devore-e.yml", 0.43, 1.53),
    ("GaAs", None, "GaAs_Skauli.yml", 0.97, 17.0),
    ("InP", None, "InP_Pettit.yml", 0.95, 10.0),
    ("SU-8", None, "SU8_3000.yml", 0.32, 1.7),
    ("air", None, "air_Ciddor.yml", 0.23, 1.69),
    ("Si", None, "Si_Salzberg.yml", 1.357, 11.04),
    ("SiO2", None, "SiO2_Malitson.yml", 0.21, 6.7),
    ("Si3N4", None, "Si3N4_Luke.yml", 0.31, 5.504),
])
def test_library_matches_refractiveindex_info_transcription(material, axis, ref_file, wl_lo, wl_hi):
    spec = get_material(material)
    # The coefficients as published; SiO2 and Al2O3 (Malitson) are relative to air and the library
    # entries are absolute, so those files are compared after the loader's own air-to-vacuum conversion.
    ref = ref_model(ref_file, to_vacuum=material in ("SiO2", "Al2O3"))
    model = spec.optical.axis_model(axis) if axis else spec.optical
    lo, hi = model.wl_range_um            # (the air-to-vacuum conversion shifts the window by n_air)
    wl = np.linspace(lo, hi, 200)
    # GaAs: the database rounds the converted Sellmeier constants to 7 digits (Skauli's form is used here)
    tol = 2e-7 if material == "GaAs" else 1e-10
    assert np.max(np.abs(re(spec.index(wl, axis=axis)) - re(ref.n(wl)))) < tol
    assert model.wl_range_um == pytest.approx(ref.wl_range_um)
    if material not in ("SiO2", "Al2O3"):
        assert (wl_lo, wl_hi) == pytest.approx(model.wl_range_um, rel=1e-3)


def test_lithium_niobate_published_values():
    """Zelmon et al. (1997): congruent LN no(1.55) ~ 2.211, ne(1.55) ~ 2.138 (21 degC)."""
    ln = get_material("LiNbO3")
    assert abs(float(ln.index(1.55, axis="o").real) - 2.211) < 5e-4
    assert abs(float(ln.index(1.55, axis="e").real) - 2.138) < 5e-4
    assert ln.pockels_pm_per_v["r33"] == 30.8


def test_air_ciddor_vs_peck_reeder():
    """Two independent standard-air dispersion formulas agree to 2e-8 in n - 1."""
    wl = np.linspace(0.4, 1.69, 200)
    assert np.max(np.abs(re(get_material("air").index(wl)) - re(ref_model("air_Peck.yml").n(wl)))) < 2e-8


def test_sapphire_malitson_dodge_vs_malitson_1962():
    """Independent sapphire measurements (1962 vs 1972) agree to 2e-4 for the ordinary ray."""
    wl = np.linspace(0.3, 5.0, 200)
    # Both papers report the index relative to air versus the wavelength in air; compare on the
    # library's absolute, vacuum-wavelength basis by converting the 1962 fit the same way.
    m1962 = ref_model("Al2O3_Malitson1962.yml")
    absolute_1962, _ = pm.io._to_absolute_vacuum(m1962.evaluate, m1962.wl_range_um, n_absolute=False,
                                                  wavelength_vacuum=False, t_air=293.15)
    diff = re(get_material("Al2O3").index(wl, axis="o")) - re(absolute_1962(wl, None))
    assert np.max(np.abs(diff)) < 2e-4


# Rows of the refractiveindex.info "Rakic-LD" tables (computed by the database from Rakic et al. 1998):
# (wavelength um, n, k). They check the Lorentz-Drude parameters entered in photonix.materials.library.
RAKIC_LD_ROWS = {
    "Au": [(0.50523, 0.74535, 1.941), (1.0129, 0.2597, 6.0788), (1.5676, 0.58129, 9.7744), (3.0426, 1.7932, 19.004)],
    "Al": [(0.5023, 0.7699, 5.9264), (1.0005, 1.4765, 9.2887), (1.5536, 1.5399, 15.368), (3.0185, 4.3917, 29.559)],
    "Cu": [(0.50054, 0.84204, 2.3418), (1.0075, 0.25557, 6.5372), (1.552, 0.47077, 10.467), (3.0602, 1.6332, 20.6)],
    "Ag": [(0.50321, 0.13208, 2.7927), (1.0013, 0.21994, 6.4341), (1.5737, 0.40905, 10.333), (3.0107, 1.2442, 19.905)],
    "W": [(0.50178, 3.3901, 2.6208), (1.0035, 3.027, 3.488), (1.556, 2.3486, 4.8411), (3.0042, 1.4288, 13.081)],
    "Ti": [(0.50217, 1.5969, 2.9068), (1.0023, 2.9718, 3.8098), (1.556, 3.8312, 4.4235), (3.0025, 4.57, 5.9674)],
    "Cr": [(0.50031, 2.533, 4.0857), (1.0039, 4.457, 4.2431), (1.5535, 4.2698, 4.9327), (3.0154, 3.596, 9.2491)],
    "Pt": [(0.50178, 1.9736, 3.4311), (1.0035, 3.4607, 5.7449), (1.556, 5.3712, 7.0502), (3.0042, 3.1396, 10.458)],
    "Ni": [(0.50056, 1.7168, 2.9755), (1.0007, 2.8011, 5.0401), (1.551, 3.4427, 6.7572), (3.0025, 4.0592, 11.184)],
}


@pytest.mark.parametrize("metal", sorted(RAKIC_LD_ROWS))
def test_rakic_lorentz_drude_parameters(metal):
    rows = np.array(RAKIC_LD_ROWS[metal])
    N = np.asarray(get_material(metal).index(rows[:, 0]))
    assert np.max(np.abs(N.real - rows[:, 1]) / rows[:, 1]) < 2e-4   # tables carry 5 significant digits
    assert np.max(np.abs(N.imag - rows[:, 2]) / rows[:, 2]) < 2e-4
    eps = np.asarray(get_material(metal).epsilon(rows[:, 0]))
    assert np.all(eps.imag > 0)                                     # absorbing, physics sign


@pytest.mark.parametrize("name, file", [("Ge", "Ge_Nunley2016.yml"), ("TiN", "TiN_Beliaev2023.yml"),
                                        ("Si_Green2008", "Si_Green2008.yml")])
def test_tabulated_entries_reproduce_their_data_rows(name, file):
    spec = get_material(name)
    raw = pm.load_refractiveindex_yaml(DATA_DIR / file).coeffs
    inside = raw["wl"][1:-1]
    N = np.asarray(spec.index(inside))
    np.testing.assert_allclose(N.real, raw["n"][1:-1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(N.imag, raw["k"][1:-1], rtol=0, atol=1e-15)


def test_absorption_anchors():
    """Green (2008): alpha_Si(1.0 um) = 64 /cm. Ge (Nunley 2016) absorbs strongly in the O band."""
    alpha = lambda spec, wl: 4 * np.pi * float(spec.index(wl).imag) / (wl * 1e-4)  # noqa: E731
    assert alpha(get_material("Si_Green2008"), 1.0) == pytest.approx(64.0, rel=0.01)
    a_ge = alpha(get_material("Ge"), 1.305)
    assert 7000 < a_ge < 9000          # interband absorption of bulk Ge at 1.3 um


# =========================================================================== #
# 4. refractiveindex.info YAML reader
# =========================================================================== #
def test_yaml_parser_matches_pyyaml():
    yaml = pytest.importorskip("yaml")
    files = sorted(REF.glob("*.yml")) + sorted(DATA_DIR.glob("*.yml"))
    assert len(files) >= 30
    for path in files:
        text = path.read_text(encoding="utf-8")
        mine = pm.parse_refractiveindex_yaml(text)
        theirs = yaml.safe_load(text)
        assert " ".join(str(mine["REFERENCES"]).split()) == " ".join(str(theirs["REFERENCES"]).split()), path
        assert len(mine["DATA"]) == len(theirs["DATA"])
        for a, b in zip(mine["DATA"], theirs["DATA"], strict=True):
            assert a["type"] == b["type"]
            for key in ("coefficients", "wavelength_range"):
                if key in b:
                    np.testing.assert_allclose([float(v) for v in a[key].split()],
                                               [float(v) for v in str(b[key]).split()])
            if "data" in b:
                assert a["data"].split() == str(b["data"]).split()


def _explicit(number: int, c: list[float], lam: np.ndarray) -> np.ndarray:
    """Formulas restated directly from the refractiveindex.info documentation, term by term."""
    l2 = lam ** 2
    if number == 3:      # PMMA (Beadie 2015)
        return np.sqrt(c[0] + c[1] * lam ** c[2] + c[3] * lam ** c[4] + c[5] * lam ** c[6] + c[7] * lam ** c[8]
                       + c[9] * lam ** c[10] + c[11] * lam ** c[12])
    if number == 5:      # SU-8 3000 Cauchy
        return c[0] + c[1] * lam ** c[2] + c[3] * lam ** c[4]
    if number == 6:      # He (Mansfield & Peck 1969)
        return 1 + c[0] + c[1] / (c[2] - lam ** -2.0)
    if number == 7:      # Si Herzberger (Edwards & Ochoa 1980)
        L = 1 / (l2 - 0.028)
        return c[0] + c[1] * L + c[2] * L ** 2 + c[3] * l2 + c[4] * l2 ** 2
    if number == 8:      # AgBr retro (Schroter)
        a = c[0] + c[1] * l2 / (l2 - c[2]) + c[3] * l2
        return np.sqrt((1 + 2 * a) / (1 - a))
    if number == 9:      # urea (Rosker 1985)
        return np.sqrt(c[0] + c[1] / (l2 - c[2]) + c[3] * (lam - c[4]) / ((lam - c[4]) ** 2 + c[5]))
    raise AssertionError(number)


@pytest.mark.parametrize("file, number, check", [
    ("SiO2_Malitson.yml", 1, "em.silica"),
    ("LiNbO3_Zelmon-o.yml", 2, "library"),
    ("PMMA_Beadie.yml", 3, "explicit"),
    ("TiO2_Devore-o.yml", 4, "library"),
    ("SU8_3000.yml", 5, "explicit"),
    ("He_Mansfield.yml", 6, "explicit"),
    ("Si_Edwards.yml", 7, "explicit"),
    ("AgBr_Schroter.yml", 8, "explicit"),
    ("urea_Rosker-e.yml", 9, "explicit"),
])
def test_yaml_formula_types_1_to_9(file, number, check):
    model = ref_model(file, to_vacuum=False)            # the formula as stored
    assert model.coeffs["formula"] == number
    lo, hi = model.wl_range_um
    wl = np.linspace(lo, hi, 101)
    got = re(model.n(wl))
    if check == "em.silica":
        expected = np.asarray(em.silica(wl))
    elif check == "library":
        mat = "LiNbO3" if number == 2 else "TiO2"
        expected = re(get_material(mat).index(wl, axis="o"))
    else:
        expected = _explicit(number, model.coeffs["coefficients"], wl)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_yaml_air_relative_data_are_converted_to_absolute_vacuum(tmp_path):
    """Regression: SPECS n_is_absolute/wavelength_is_vacuum = false were ignored, so air-relative
    glass data (Sellmeier of N-BK7 type, 20 degC) were off by ~4e-4."""
    from photonix.materials.formulas import ciddor1996_air

    body = (
        'REFERENCES: "test glass, catalogue Sellmeier relative to air"\n'
        "DATA:\n"
        "  - type: formula 2\n"
        "    wavelength_range: 0.3 2.5\n"
        "    coefficients: 0 1.03961212 0.00600069867 0.231792344 0.0200179144 1.01046945 103.560653\n"
        "SPECS:\n"
        "    n_is_absolute: {abs}\n"
        "    wavelength_is_vacuum: {vac}\n"
        "    temperature: 20 °C\n"
    )
    paths = {}
    for a in ("true", "false"):
        for v in ("true", "false"):
            paths[a, v] = tmp_path / f"glass_{a}_{v}.yml"
            paths[a, v].write_text(body.format(abs=a, vac=v), encoding="utf-8")
    c = [1.03961212, 0.00600069867, 0.231792344, 0.0200179144, 1.01046945, 103.560653]

    def n_rel(lam):                                        # the stored formula, evaluated by hand
        lam2 = lam * lam
        return np.sqrt(1 + sum(c[i] * lam2 / (lam2 - c[i + 1]) for i in (0, 2, 4)))

    t_air = 293.15
    wl = np.linspace(0.4, 2.0, 41)                         # vacuum wavelengths
    n_air = np.asarray(ciddor1996_air(wl, t_air))
    lam_air = wl / n_air
    assert np.max(np.abs(lam_air * np.asarray(ciddor1996_air(wl, t_air)) - wl)) < 1e-15
    expected = {("true", "true"): n_rel(wl), ("false", "true"): n_rel(wl) * n_air,
                ("true", "false"): n_rel(lam_air), ("false", "false"): n_rel(lam_air) * n_air}
    for key, path in paths.items():
        m = pm.load_refractiveindex_yaml(path)
        assert np.max(np.abs(re(m.n(wl)) - expected[key])) < 1e-13, key
        raw = pm.load_refractiveindex_yaml(path, to_vacuum=False)
        assert np.max(np.abs(re(raw.n(wl)) - n_rel(wl))) < 1e-13, key
    m = pm.load_refractiveindex_yaml(paths["false", "false"])
    assert 3.5e-4 < float(m.n(1.55).real) - n_rel(1.55) < 4.5e-4          # the size of the correction
    # the air-wavelength range 0.3-2.5 um maps to vacuum wavelengths: l_vac = n_air(l_vac) l_air
    lo, hi = m.wl_range_um
    assert lo / float(ciddor1996_air(lo, t_air)) == pytest.approx(0.3, abs=1e-14)
    assert hi / float(ciddor1996_air(hi, t_air)) == pytest.approx(2.5, abs=1e-14)
    assert m.coeffs["air_conversion"]["air_temperature_k"] == pytest.approx(t_air)
    # a malitson-type reference file of the database is converted too (fused silica, relative to air)
    silica = ref_model("SiO2_Malitson.yml")
    assert float(silica.n(1.55).real) - float(em.silica(1.55)) == pytest.approx(3.94e-4, abs=2e-5)
    bad = tmp_path / "bad.yml"
    bad.write_text(body.format(abs="maybe", vac="true"), encoding="utf-8")
    with pytest.raises(ValueError, match="n_is_absolute"):
        pm.load_refractiveindex_yaml(bad)


def test_yaml_formula_anchor_helium():
    """Helium at 0 degC, 1 atm: n - 1 ~ 3.49e-5 at 632.8 nm (Mansfield & Peck 1969)."""
    assert float(ref_model("He_Mansfield.yml").n(0.6328).real) - 1 == pytest.approx(3.49e-5, rel=0.01)


def test_yaml_formula_plus_tabulated_k():
    """Water (Kedenburg 2012): formula-2 n with a tabulated k block; range is the overlap."""
    m = ref_model("H2O_Kedenburg.yml")
    assert m.wl_range_um == (0.5, 1.6)
    wl_k, k = m.coeffs["wl_k"], m.coeffs["k"]
    sel = (wl_k > 0.5) & (wl_k < 1.6)
    np.testing.assert_allclose(np.asarray(m.n(wl_k[sel])).imag, k[sel], rtol=1e-12)
    # the file lists 1.150 um twice (two stitched spectrometer ranges): the rows are averaged
    assert float(m.n(1.15).imag) == pytest.approx(0.5 * (8.95923e-06 + 8.64808e-06), rel=1e-12)
    np.testing.assert_allclose(re(m.n(np.array([0.6, 1.2]))),
                               np.sqrt(1 + 0.75831 * np.array([0.36, 1.44]) / (np.array([0.36, 1.44]) - 0.01007)
                                       + 0.08495 * np.array([0.36, 1.44]) / (np.array([0.36, 1.44]) - 8.91377)),
                               rtol=1e-14)
    assert m.reference_temperature_k == pytest.approx(293.15)


def test_yaml_tabulated_n_and_k_on_different_grids():
    m = ref_model("Si_Green-1995.yml")
    assert m.wl_range_um == (0.25, 1.0)
    assert m.reference_temperature_k == pytest.approx(300.0)


def test_yaml_reference_metadata():
    m = ref_model("LiNbO3_Zelmon-o.yml")
    assert m.reference.doi == "10.1364/JOSAB.14.003319"
    assert "Zelmon" in m.reference.citation and "<" not in m.reference.citation
    assert m.reference_temperature_k == pytest.approx(294.15)          # "21 °C" in COMMENTS
    assert ref_model("TiO2_Devore-o.yml").temperature_range_k == pm.ROOM_TEMPERATURE_K


def test_yaml_errors(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text("DATA:\n  - type: formula 12\n    wavelength_range: 1 2\n    coefficients: 0 1 1\n")
    with pytest.raises(ValueError):
        pm.load_refractiveindex_yaml(bad).n(1.5)
    only_k = tmp_path / "k.yml"
    only_k.write_text("DATA:\n  - type: tabulated k\n    data: |\n        1.0 0.1\n        2.0 0.2\n")
    with pytest.raises(ValueError, match="no block defines"):
        pm.load_refractiveindex_yaml(only_k)
    weird = tmp_path / "w.yml"
    weird.write_text("DATA:\n  - type: spline n\n")
    with pytest.raises(ValueError, match="unsupported"):
        pm.load_refractiveindex_yaml(weird)


def test_nk_csv_loader(tmp_path):
    p = tmp_path / "film.csv"
    p.write_text("# my film\nE_eV;n;k\n0.8;2.00;0.010\n1.0;2.05;0.020\n")
    m = pm.load_nk_csv(p, wl_unit="eV")
    assert m.wl_range_um == pytest.approx((1.2398419843320025, 1.549802480415003))
    assert complex(m.n(1.2398419843320025)) == pytest.approx(2.05 + 0.02j)
    neg = tmp_path / "neg.csv"
    neg.write_text("1500 2.0 -0.1\n1600 2.0 0.0\n")
    with pytest.raises(ValueError, match="k must be >= 0"):
        pm.load_nk_csv(neg, wl_unit="nm")


# =========================================================================== #
# 5. Validity-range guard
# =========================================================================== #
def test_range_guard_wavelength():
    si = get_material("Si")                      # em.silicon-backed: 1.357-11.04 um
    with pytest.warns(MaterialRangeWarning, match="1.31"):
        si.index(1.31)
    with pytest.raises(MaterialRangeError):
        si.index(1.31, strict=True)
    with pytest.warns(MaterialRangeWarning):     # the LayerLevel-style call is guarded too
        si(1.31)
    # Li (1980) covers the O band cleanly (warnings are errors in this module)
    assert 3.50 < float(get_material("Si_Li1980").index(1.31).real) < 3.51


def test_range_guard_temperature():
    with pytest.warns(MaterialRangeWarning, match="temperature"):
        get_material("Si").index(1.55, 900.0)
    with pytest.raises(MaterialRangeError):
        get_material("SiO2").index(1.55, 500.0, strict=True)
    # temperature-independent data: within +-5 K of the data temperature is accepted, beyond warns
    get_material("Ge").index(1.55, 303.0)
    with pytest.warns(MaterialRangeWarning):
        get_material("Ge").index(1.55, 350.0)
    with pytest.raises(MissingPropertyError):
        get_material("Ge").dn_dt(1.55)


@requires_jax
def test_range_guard_under_jax_transforms():
    import jax

    si = get_material("Si")
    with pytest.warns(MaterialRangeWarning):     # primal values are checked under grad ...
        jax.grad(lambda w: si.index(w).real)(1.31)
    with pytest.warns(MaterialRangeWarning):     # ... and under vmap
        jax.vmap(lambda w: si.index(w).real)(np.array([1.30, 1.55]))
    jax.jit(lambda w: si.index(w).real)(1.31)    # abstract under jit: check skipped, no error


# =========================================================================== #
# 6. Differentiability (JAX) and backend agreement
# =========================================================================== #
@requires_jax
@pytest.mark.parametrize("name, kwargs, wl0", [
    ("Si", {}, 1.55), ("Si_Li1980", {}, 1.31), ("SiO2", {}, 1.55), ("LiNbO3", {"axis": "e"}, 1.55),
    ("Au", {}, 1.55), ("GaAs", {}, 1.55), ("air", {}, 1.55),
])
def test_grad_wrt_wavelength_matches_finite_difference(name, kwargs, wl0):
    spec = get_material(name)

    def n_of(w):
        return spec.index(w, **kwargs).real

    g = float(px.grad(n_of)(wl0))
    fd = float(finite_diff(lambda w: float(n_of(w)), wl0, h=1e-5))
    assert abs(g - fd) < 1e-8
    assert (g > 0) if name == "Au" else (g < 0)   # dielectrics: normal dispersion; Au: n rises in the IR


@requires_jax
@pytest.mark.parametrize("name, t0", [("Si", 300.0), ("Si_Li1980", 150.0), ("SiO2", 320.0), ("Si3N4", 300.0),
                                      ("air", 300.0), ("Ge_Li1980", 400.0)])
def test_grad_wrt_temperature_matches_finite_difference(name, t0):
    spec = get_material(name)
    wl0 = 2.5 if name == "Ge_Li1980" else 1.55

    def n_of(t):
        return spec.index(wl0, t).real

    g = float(px.grad(n_of)(t0))
    fd = float(finite_diff(lambda t: float(n_of(t)), t0, h=1e-3))
    assert abs(g - fd) < 1e-8 * max(1.0, abs(fd) * 1e4)
    assert g == pytest.approx(float(spec.dn_dt(wl0, t0)), rel=1e-6)


@requires_jax
def test_grad_of_free_carrier_model():
    g = float(px.grad(lambda nh: pm.nedeljkovic_2011(0.0, nh, 1.55)[0])(1e18))
    expected = -1.53e-18 * 0.838 * (1e18) ** (0.838 - 1.0)     # analytic derivative of the fit
    assert g == pytest.approx(expected, rel=1e-10)


@requires_jax
def test_numpy_backend_returns_the_same_values():
    code = (
        "import json, warnings; warnings.simplefilter('ignore')\n"
        "import photonix as px, photonix.materials as pm\n"
        "assert not px.HAS_JAX\n"
        "m = pm.get_material\n"
        "vals = [complex(m(n).index(1.55, axis=a)) for n, a in "
        "[('Si', None), ('Si_Li1980', None), ('SiO2', None), ('LiNbO3', 'e'), ('Au', None), ('Ge', None), "
        "('TiN', None), ('GaAs', None)]]\n"
        "vals += [complex(m('Si').index(1.55, 350.0)), complex(pm.doped_si(1e18, 1e18).index(1.55))]\n"
        "print(json.dumps([[v.real, v.imag] for v in vals]))\n"
    )
    src = str(Path(px.__file__).resolve().parents[1])
    env = dict(os.environ, PHOTONIX_BACKEND="numpy", PYTHONPATH=src)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    numpy_vals = np.array(json.loads(out.stdout.strip().splitlines()[-1]))
    m = get_material
    jax_vals = [complex(m(n).index(1.55, axis=a)) for n, a in
                [("Si", None), ("Si_Li1980", None), ("SiO2", None), ("LiNbO3", "e"), ("Au", None), ("Ge", None),
                 ("TiN", None), ("GaAs", None)]]
    jax_vals += [complex(m("Si").index(1.55, 350.0)), complex(pm.doped_si(1e18, 1e18).index(1.55))]
    np.testing.assert_allclose(numpy_vals[:, 0], [v.real for v in jax_vals], rtol=1e-14, atol=0)
    np.testing.assert_allclose(numpy_vals[:, 1], [v.imag for v in jax_vals], rtol=1e-12, atol=1e-18)


# =========================================================================== #
# 7. Passive Lorentz-Drude fits and the Meep round trip
# =========================================================================== #
def _meep_formula(params: dict, wl: np.ndarray) -> np.ndarray:
    """Meep's documented susceptibility sum, re-implemented independently (a = 1 um, f = 1/wl)."""
    f = 1.0 / wl
    eps = np.full(f.shape, params["epsilon"], dtype=complex)
    for s in params["E_susceptibilities"]:
        fn, g, sig = s["frequency"], s["gamma"], s["sigma"]
        if s["type"] == "lorentzian":
            eps += sig * fn ** 2 / (fn ** 2 - f ** 2 - 1j * f * g)
        else:
            eps += sig * fn ** 2 / (-f ** 2 - 1j * f * g)
    return eps


def _assert_passive_everywhere(fit):
    assert fit.is_passive()
    wl = 1.0 / np.geomspace(1e-3, 1e3, 4001)       # six decades of frequency, not only the band
    assert np.all(fit.epsilon(wl).imag >= -1e-12)


def test_lorentz_fit_recovers_exact_lossy_oscillator():
    """Analytic limit: a target that *is* a damped Lorentz oscillator is recovered exactly."""
    def target(wl):
        f = 1.0 / np.asarray(wl)
        return 2.25 + 1.3 * 0.7 ** 2 / (0.7 ** 2 - f ** 2 - 1j * f * 0.05)

    fit = pm.fit_lorentz(target, 1.2, 1.8, n_poles=1)
    assert fit.max_rel_error < 1e-8
    (p,) = fit.poles
    assert (fit.eps_inf, p.delta_eps, p.frequency, p.gamma) == pytest.approx((2.25, 1.3, 0.7, 0.05), rel=1e-6)
    _assert_passive_everywhere(fit)


@pytest.mark.parametrize("name, lo, hi, kwargs, tol", [
    ("Si3N4", 1.2, 2.0, dict(n_poles=2), 2e-6),
    ("SiO2", 1.2, 1.7, dict(n_poles=2), 2e-6),
    ("Si", 1.4, 1.7, dict(n_poles=3), 1e-6),
    ("LiNbO3", 1.2, 1.7, dict(n_poles=2, axis="e"), 2e-6),
])
def test_lorentz_fit_of_sellmeier_dielectrics(name, lo, hi, kwargs, tol):
    """A Sellmeier medium is a sum of lossless Lorentz poles, so a few poles fit it to ~1e-6."""
    fit = pm.fit_lorentz(get_material(name), lo, hi, **kwargs)
    assert fit.max_rel_error < tol
    _assert_passive_everywhere(fit)


@pytest.mark.parametrize("metal", ["Au", "Al"])
def test_lorentz_drude_fit_of_metals(metal):
    fit = pm.fit_lorentz(get_material(metal), 1.2, 1.7, n_poles=2, drude=True)
    assert fit.max_rel_error < 1e-3
    assert fit.drude is not None
    _assert_passive_everywhere(fit)
    wl = np.linspace(1.2, 1.7, 7)
    # Meep round trip: the exported parameters, put through Meep's own formula, give the same eps
    np.testing.assert_allclose(_meep_formula(fit.meep_parameters(), wl), fit.epsilon(wl), rtol=1e-12)
    np.testing.assert_allclose(fit.epsilon(wl, convention="engineering"), np.conj(fit.epsilon(wl)))


def test_meep_medium_round_trip_if_meep_installed():
    pytest.importorskip("meep")
    fit = pm.fit_lorentz(get_material("Au"), 1.3, 1.7, n_poles=1, drude=True)
    medium = fit.to_meep_medium()
    for wl in (1.3, 1.5, 1.7):
        assert complex(medium.epsilon(1.0 / wl)[0, 0]) == pytest.approx(complex(fit.epsilon(wl)), rel=1e-9)


def test_to_meep_medium_without_meep_raises_import_error():
    from photonix.em.meep._guard import HAS_MEEP

    if HAS_MEEP:
        pytest.skip("Meep is installed")
    fit = pm.fit_lorentz(get_material("SiO2"), 1.3, 1.6, n_poles=1)
    with pytest.raises(ImportError, match="Meep"):
        fit.to_meep_medium()


# =========================================================================== #
# 8. Kramers-Kronig consistency
# =========================================================================== #
def test_kramers_kronig_analytic_lorentz():
    f = np.geomspace(0.01, 100.0, 6000)
    def N(ff):
        return np.sqrt(1 + 2.0 / (1 - ff ** 2 - 0.2j * ff))
    n_kk = pm.kramers_kronig_n(1 / f, N(f).imag, wl_ref=10.0, n_ref=float(N(0.1).real),
                               wl_eval=1 / np.array([0.3, 0.8, 1.0, 1.2, 3.0]))
    np.testing.assert_allclose(n_kk, N(np.array([0.3, 0.8, 1.0, 1.2, 3.0])).real, atol=2e-3)


def test_kramers_kronig_rakic_aluminium():
    """Rakic's Al model spans 0.062-248 um, so KK of its k reproduces its n to ~0.5 % in the NIR."""
    wl = np.geomspace(0.0621, 247.0, 3000)
    N = np.asarray(get_material("Al").index(wl))
    ev = np.array([0.5, 1.0, 1.55, 3.0])
    n_kk = pm.kramers_kronig_n(wl, N.imag, wl_ref=1.0, n_ref=float(np.asarray(get_material("Al").index(1.0)).real),
                               wl_eval=ev)
    np.testing.assert_allclose(n_kk, np.asarray(get_material("Al").index(ev)).real, rtol=1e-2)


@pytest.mark.parametrize("name, wl_ref, wl_eval, tol", [
    ("Ge", 1.8, [1.0, 1.31, 1.55, 2.0, 2.3], 1.5e-2),
    ("Si_Green2008", 1.2, [1.0, 1.1, 1.3, 1.4], 1.5e-2),
])
def test_kramers_kronig_consistency_of_measured_tables(name, wl_ref, wl_eval, tol):
    """Measured n,k sets are KK-consistent to ~1e-2 (ellipsometric n uncertainty and UV truncation)."""
    c = get_material(name).optical.coeffs
    n_kk = pm.kramers_kronig_n(c["wl"], c["k"], wl_ref=wl_ref, n_ref=float(np.interp(wl_ref, c["wl"], c["n"])),
                               wl_eval=wl_eval)
    assert np.max(np.abs(n_kk - np.interp(wl_eval, c["wl"], c["n"]))) < tol


def test_kramers_kronig_input_validation():
    with pytest.raises(ValueError):
        pm.kramers_kronig_n([1.0, 2.0, 3.0], [0.1, -0.1, 0.0], wl_ref=2.0, n_ref=1.5)
    with pytest.raises(ValueError, match="inside"):
        pm.kramers_kronig_n([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], wl_ref=5.0, n_ref=1.5)


# =========================================================================== #
# 9. Free carriers in silicon
# =========================================================================== #
def test_soref_bennett_drude_scaling_matches_their_1310nm_fit():
    """S&B (1987) publish separate 1.3 um fits; the (wl/1.55)^2 scaling of the 1.55 um fit reproduces them."""
    s = (1.31 / 1.55) ** 2
    for scaled, published in ((8.8e-22 * s, 6.2e-22), (8.5e-18 * s, 6.0e-18),
                              (8.5e-18 * s, 6.0e-18), (6.0e-18 * s, 4.0e-18)):
        assert abs(scaled - published) / published < 0.08
    dn, da = pm.soref_bennett_1987(1e18, 0.0, 1.31)
    assert float(dn) == pytest.approx(-8.8e-22 * 1e18 * s, rel=1e-12)
    assert float(da) == pytest.approx(8.5e-18 * 1e18 * s, rel=1e-12)


def test_soref_bennett_matches_components_plasma_dispersion_if_present():
    fn = getattr(px.components, "plasma_dispersion", None)
    if fn is None:
        pytest.skip("photonix.components.plasma_dispersion not in this tree")
    for wl in (1.31, 1.55):
        a, b = fn(3e17, 7e17, wl=wl), pm.soref_bennett_1987(3e17, 7e17, wl)
        assert float(a[0]) == pytest.approx(float(b[0]), rel=1e-12)
        assert float(a[1]) == pytest.approx(float(b[1]), rel=1e-12)


def test_nedeljkovic_coefficients_and_doped_si():
    dn, da = pm.nedeljkovic_2011(1e18, 2e18, 1.55)
    assert float(dn) == pytest.approx(-5.4e-22 * 1e18 ** 1.011 - 1.53e-18 * 2e18 ** 0.838, rel=1e-12)
    assert float(da) == pytest.approx(8.88e-21 * 1e18 ** 1.167 + 5.84e-20 * 2e18 ** 1.109, rel=1e-12)
    d = pm.doped_si(1e18, 2e18)
    base = get_material("Si_Li1980")
    N = complex(d.index(1.55))
    assert N.real - float(base.index(1.55).real) == pytest.approx(float(dn), rel=1e-10)
    assert 4 * np.pi * N.imag / 1.55e-4 == pytest.approx(float(da), rel=1e-10)    # k <-> alpha [1/cm]
    assert "doped" in d.tags and d.thermal_conductivity == base.thermal_conductivity
    with pytest.warns(MaterialRangeWarning):                    # Nedeljkovic fit window is 1.5-1.6 um
        d.index(1.31)
    pm.doped_si(1e18, 0.0, model="soref_bennett1987").index(1.31)   # S&B window reaches 1.3 um
    with pytest.raises(ValueError):
        pm.doped_si(-1.0)
    with pytest.raises(ValueError):
        pm.doped_si(1e18, model="drude")


# =========================================================================== #
# 10. Protocol, anisotropy, registry, provenance
# =========================================================================== #
def test_every_registered_material_satisfies_the_protocol():
    for name in pm.list_materials():
        spec = get_material(name)
        assert isinstance(spec, pm.MaterialLike)
        models = [spec.optical] if spec.isotropic else [spec.optical.ordinary, spec.optical.extraordinary]
        for model in models:
            assert model.reference.citation and model.reference.citation != "user supplied", name
            if name != "vacuum":
                assert model.wl_range_um is not None, name
                assert model.temperature_range_k is not None, name
        lo, hi = models[0].wl_range_um or (1.0, 2.0)
        wl = 1.55 if lo <= 1.55 <= hi else 0.5 * (lo + hi)
        axis = None if spec.isotropic else "o"
        N = complex(spec.index(wl, axis=axis))
        assert N.imag >= 0, name                               # physics sign: k >= 0
        assert complex(spec.epsilon(wl, axis=axis)) == pytest.approx(N * N, rel=1e-12)


def test_every_non_optical_value_has_a_citation():
    cited_by = {"thermal_conductivity": ("thermal",), "heat_capacity": ("thermal",),
                "density": ("thermal", "density"), "relative_permittivity_static": ("static_permittivity",),
                "resistivity": ("electrical",), "resistivity_tcr": ("electrical", "electrical_tcr"),
                "pockels_pm_per_v": ("electro_optic",)}
    for name in pm.list_materials():
        spec = get_material(name)
        for field, keys in cited_by.items():
            if getattr(spec, field) is None or (field == "relative_permittivity_static" and name in ("air", "vacuum")):
                continue                    # vacuum/air static permittivity 1.0 is exact / an approximation
            assert any(k in spec.references for k in keys), (name, field)


def test_wiedemann_franz_consistency_of_metal_data():
    """Independent check of the handbook values: k_th * rho / T ~ Lorenz number 2.44e-8 W ohm / K^2."""
    for name in ("Al", "Cu", "Au", "Ag", "W", "Pt"):
        spec = get_material(name)
        lorenz = spec.thermal_conductivity * spec.resistivity / 300.0
        assert 0.8 < lorenz / 2.44e-8 < 1.35, (name, lorenz)


def test_uniaxial_orientation_and_scalar_views():
    ln = get_material("LiNbO3")
    eo, ee = complex(ln.epsilon(1.55, axis="o")), complex(ln.epsilon(1.55, axis="e"))
    cases = {("z", "x"): (eo, ee, eo), ("z", "y"): (eo, ee, eo), ("x", "y"): (ee, eo, eo),
             ("x", "z"): (eo, eo, ee), ("y", "x"): (ee, eo, eo), ("y", "z"): (eo, eo, ee)}
    for (cut, prop), expected in cases.items():
        got = tuple(complex(v) for v in ln.eps_tensor(1.55, cut=cut, propagation_axis=prop))
        assert got == pytest.approx(expected), (cut, prop)
    with pytest.raises(ValueError):
        ln.eps_tensor(1.55, cut="z", propagation_axis="z")
    with pytest.raises(ValueError, match="uniaxial"):
        ln.index(1.55)
    with pytest.raises(TypeError):
        ln(1.55)
    view = ln.component("e")
    assert float(np.asarray(view(1.55))) == pytest.approx(float(ln.index(1.55, axis="e").real))
    assert ln.static_permittivity_tensor(cut="x", propagation_axis="y") == (27.9, 44.3, 44.3)
    si_t = get_material("Si").eps_tensor(1.55)
    assert si_t[0] == si_t[1] == si_t[2]


def test_engineering_sign_convention():
    au = get_material("Au")
    n_phys = complex(au.index(1.55))
    assert complex(au.index(1.55, convention="engineering")) == n_phys.conjugate()
    assert complex(pm.to_engineering_sign(au.epsilon(1.55))).imag < 0     # photonix.em loss sign
    with pytest.raises(ValueError):
        au.index(1.55, convention="optics")


def test_registry_lookup_and_user_materials():
    assert get_material("silicon") is get_material("Si") is get_material("SI")
    assert get_material("SU8") is get_material("su-8")
    assert get_material(get_material("Cu")) is get_material("Cu")
    with pytest.raises(KeyError, match="Did you mean"):
        get_material("Silicn")
    spec = pm.MaterialSpec("TestGlass", pm.constant_model(1.5, reference=pm.Reference("test")))
    pm.register(spec, aliases=["tg"])
    try:
        assert get_material("TG") is spec
        with pytest.raises(ValueError, match="already registered"):
            pm.register(spec)
    finally:
        pm.unregister("TestGlass")
    with pytest.raises(KeyError):
        get_material("tg")
    assert set(pm.list_materials(tag="metal")) >= {"Al", "Cu", "Au", "TiN"}


def test_as_material_and_replace():
    assert complex(pm.as_material(2.0 + 0.1j).index(1.0)) == 2.0 + 0.1j
    assert float(pm.as_material(em.materials.SI).index(1.55).real) == float(em.silicon(1.55))
    with pytest.raises(TypeError):
        pm.as_material("Si")
    film = get_material("Si").replace(thermal_conductivity=60.0)
    assert film.thermal_conductivity == 60.0 and get_material("Si").thermal_conductivity == 148.0
    assert get_material("Cu").resistivity_at(393.15) == pytest.approx(1.7241e-8 * 1.393)
    with pytest.raises(MissingPropertyError):
        get_material("Au").resistivity_at(400.0)               # no TCR recorded for Au
    with pytest.raises(MissingPropertyError):
        get_material("SU-8").static_permittivity_tensor()


def test_layerstack_compatibility_if_available():
    """A MaterialSpec is a plain ``n(wl)`` callable, so it drops into ``LayerLevel(material=...)``."""
    stack_mod = pytest.importorskip("photonix.pdk.stack")
    level = stack_mod.LayerLevel(name="core", layer=(1, 0), thickness=0.22, z_min=0.0, material=get_material("Si"))
    assert stack_mod._index(level.material, 1.55) == pytest.approx(float(em.silicon(1.55)))


# =========================================================================== #
# 11. Lorentz-Lorenz mixing (SiON)
# =========================================================================== #
def test_lorentz_lorenz_mixing_limits_and_identity():
    sio2, sin = get_material("SiO2"), get_material("Si3N4")
    wl = np.linspace(1.2, 1.7, 11)
    np.testing.assert_array_equal(re(pm.sion(0.0).index(wl)), re(sio2.index(wl)))
    np.testing.assert_array_equal(re(pm.sion(1.0).index(wl)), re(sin.index(wl)))
    x = 0.37
    n = np.asarray(pm.sion(x).index(wl))
    def ll(e):
        return (e - 1) / (e + 2)
    rhs = (1 - x) * ll(np.asarray(sio2.epsilon(wl))) + x * ll(np.asarray(sin.epsilon(wl)))
    np.testing.assert_allclose(ll(n ** 2), rhs, rtol=1e-13)
    n155 = [float(pm.sion(f).index(1.55).real) for f in np.linspace(0, 1, 11)]
    assert np.all(np.diff(n155) > 0)                              # monotonic between the end members
    # the temperature dependence of both constituents is carried through: differentiating the mixing rule,
    # q'(n) dn/dT = (1 - x) q'(n_a) dn_a/dT + x q'(n_b) dn_b/dT with q'(n) = 6 n / (n^2 + 2)^2
    def dq(nn):
        return 6 * nn / (nn ** 2 + 2) ** 2
    na, nb = float(sio2.index(1.55, 300.0).real), float(sin.index(1.55, 300.0).real)
    nm = float(pm.sion(0.5).index(1.55, 300.0).real)
    expected = (0.5 * dq(na) * float(sio2.dn_dt(1.55, 300.0)) + 0.5 * dq(nb) * float(sin.dn_dt(1.55, 300.0))) / dq(nm)
    assert float(pm.sion(0.5).dn_dt(1.55, 300.0)) == pytest.approx(expected, rel=1e-6)
    with pytest.raises(ValueError):
        pm.sion(1.5)
    with pytest.raises(ValueError):
        pm.lorentz_lorenz_mix(get_material("LiNbO3"), sio2, 0.5)


def test_lorentz_lorenz_mixing_keeps_loss_sign():
    mix = pm.lorentz_lorenz_mix(get_material("SiO2"), get_material("Ge"), 0.1)
    assert float(mix.index(1.31).imag) > 0
