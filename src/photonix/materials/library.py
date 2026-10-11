"""Curated, cited material library.

Every entry records where each number comes from (:meth:`MaterialSpec.citations`)
and the wavelength/temperature window its source supports; evaluating outside
that window warns (:class:`~photonix.materials.MaterialRangeWarning`). The
Sellmeier functions of :mod:`photonix.em.materials` remain the canonical
implementations for Si, SiO2 and Si3N4 -- the entries below *reference* them,
so ``SI.index(wl).real`` is bit-identical to :func:`photonix.em.silicon`.

Optical anchors (each value is checked in ``tests/test_materials_db.py``
against an independent source; see ``docs/MATERIALS.md`` for the evidence level
of every entry):

>>> from photonix.materials import get_material as m
>>> round(float(m("Si").index(1.55).real), 4)              # Salzberg & Villa / Tatian Sellmeier (em.silicon)
3.4777
>>> round(float(m("Si_Li1980").index(1.55, 293.0).real), 4)  # Li (1980) recommended table: 3.4757
3.4757
>>> round(float(m("SiO2").index(1.55).real), 4)            # Malitson (1965)
1.444
>>> round(float(m("Si3N4").index(1.55).real), 4)           # Luke et al. (2015)
1.9963
>>> ln = m("LiNbO3")                                       # Zelmon et al. (1997): no ~ 2.211, ne ~ 2.138
>>> round(float(ln.index(1.55, axis="o").real), 3), round(float(ln.index(1.55, axis="e").real), 3)
(2.211, 2.138)
>>> round(float(m("InP").index(1.55).real), 4)             # Pettit & Turner (1965) / Handbook of Optics
3.1649
>>> round(float(m("GaAs").index(1.55).real), 4)            # Skauli et al. (2003)
3.3702
>>> N = complex(m("Ge").index(1.305))                      # Nunley et al. (2016) table row: 4.279 + 0.08699j
>>> round(N.real, 3), round(N.imag, 5)
(4.279, 0.08699)
>>> round(float(m("Al2O3").index(1.55, axis="o").real), 4)  # Malitson & Dodge (1972), sapphire
1.7462
>>> round(float(m("TiO2").index(1.0, axis="o").real), 4)    # Devore (1951), rutile
2.4856
>>> round(float(m("SU-8").index(1.55).real), 4)            # MicroChem SU-8 3000 data sheet Cauchy
1.5552
>>> round(float(m("air").index(1.55).real) - 1.0, 8)        # Ciddor (1996), standard air
0.00027326
>>> N = complex(m("Au").index(1.5676))                     # Rakic et al. (1998) LD table: 0.58129 + 9.7744j
>>> round(N.real, 3), round(N.imag, 2)
(0.581, 9.77)
>>> N = complex(m("TiN").index(1.55))                      # Beliaev et al. (2023), sputtered film
>>> round(N.real, 2), round(N.imag, 2)
(2.17, 8.0)
"""
from __future__ import annotations

from typing import Any

from photonix.core.backend import xp
from photonix.em import materials as _em

from .formulas import HC_EV_UM, li1980_germanium, li1980_silicon, lorentz_drude_epsilon, refractiveindex_formula
from .io import DATA_DIR, load_refractiveindex_yaml
from .model import (
    ROOM_TEMPERATURE_K,
    MaterialSpec,
    OpticalModel,
    Reference,
    Uniaxial,
    callable_model,
    constant_model,
    with_linear_thermo_optic,
    with_temperature_shift,
)
from .registry import register

__all__ = [
    "SI", "SI_LI1980", "SI_GREEN2008", "SIO2", "SI3N4", "AIR", "VACUUM",
    "GE", "GE_LI1980", "INP", "GAAS", "LINBO3", "AL2O3", "TIO2", "SU8",
    "AL", "CU", "AU", "AG", "W", "TI", "CR", "PT", "NI", "TIN",
    "REFERENCES",
]


def _window(t_k: float, half_width: float = 5.0) -> tuple[float, float]:
    return (t_k - half_width, t_k + half_width)


# =========================================================================== #
# References
# =========================================================================== #
REFERENCES: dict[str, Reference] = {
    "salzberg1957": Reference(
        "C. D. Salzberg and J. J. Villa, Infrared refractive indexes of silicon, germanium and modified selenium "
        "glass, J. Opt. Soc. Am. 47, 244-246 (1957); Sellmeier fit by B. Tatian, Appl. Opt. 23, 4477-4485 (1984)",
        doi="10.1364/JOSA.47.000244", wl_range_um=(1.357, 11.04), temperature_range_k=(299.15, 299.15),
        notes="26 degC, unknown purity; the fit is photonix.em.silicon"),
    "li1980": Reference(
        "H. H. Li, Refractive index of silicon and germanium and its wavelength and temperature derivatives, "
        "J. Phys. Chem. Ref. Data 9, 561-658 (1980)",
        doi="10.1063/1.555624",
        notes="analytic n(wl, T); enforced windows are where photonix verified it against Li's tables"),
    "green2008": Reference(
        "M. A. Green, Self-consistent optical parameters of intrinsic silicon at 300 K including temperature "
        "coefficients, Sol. Energy Mater. Sol. Cells 92, 1305-1310 (2008)",
        doi="10.1016/j.solmat.2008.06.009", wl_range_um=(0.25, 1.45), temperature_range_k=(300.0, 300.0),
        notes="Kramers-Kronig-consistent compilation; k is band-to-band absorption only (no free carriers)"),
    "malitson1965": Reference(
        "I. H. Malitson, Interspecimen comparison of the refractive index of fused silica, J. Opt. Soc. Am. 55, "
        "1205-1208 (1965); validity to 6.7 um per C. Z. Tan, J. Non-Cryst. Solids 223, 158-163 (1998)",
        doi="10.1364/JOSA.55.001205", wl_range_um=(0.21, 6.7), temperature_range_k=(293.15, 293.15),
        notes="fused silica, 20 degC; the fit is photonix.em.silica"),
    "luke2015": Reference(
        "K. Luke, Y. Okawachi, M. R. E. Lamont, A. L. Gaeta and M. Lipson, Broadband mid-infrared frequency comb "
        "generation in a Si3N4 microresonator, Opt. Lett. 40, 4823-4826 (2015)",
        doi="10.1364/OL.40.004823", wl_range_um=(0.31, 5.504),
        notes="LPCVD stoichiometric Si3N4 film (340 nm on thermal oxide), room temperature; the fit is "
              "photonix.em.silicon_nitride"),
    "arbabi2013": Reference(
        "A. Arbabi and L. L. Goddard, Measurements of the refractive indices and thermo-optic coefficients of "
        "Si3N4 and SiOx using microring resonances, Opt. Lett. 38, 3878-3881 (2013)",
        doi="10.1364/OL.38.003878",
        notes="dn/dT = 2.45e-5 /K (LPCVD Si3N4) and 0.95e-5 /K (PECVD SiOx) near 1.55 um and room temperature"),
    "ciddor1996": Reference(
        "P. E. Ciddor, Refractive index of air: new equations for the visible and near infrared, Appl. Opt. 35, "
        "1566-1573 (1996)", doi="10.1364/AO.35.001566", wl_range_um=(0.23, 1.69),
        temperature_range_k=(288.15, 288.15),
        notes="standard air: dry, 15 degC, 101.325 kPa, 450 ppm CO2"),
    "nunley2016": Reference(
        "T. N. Nunley, N. S. Fernando, N. Samarasingha, J. M. Moya, C. M. Nelson, A. A. Medina and S. Zollner, "
        "Optical constants of germanium and thermally grown germanium dioxide from 0.5 to 6.6 eV via a "
        "multi-sample ellipsometry investigation, J. Vac. Sci. Technol. B 34, 061205 (2016)",
        doi="10.1116/1.4963075", wl_range_um=(0.1879, 2.48), temperature_range_k=(300.0, 300.0),
        notes="bulk Ge, 300 K, spectroscopic ellipsometry; k near the direct edge (1.5-1.6 um) is small and has "
              "a large relative uncertainty"),
    "pettit1965": Reference(
        "G. D. Pettit and W. J. Turner, Refractive index of InP, J. Appl. Phys. 36, 2081 (1965); Sellmeier from "
        "Handbook of Optics, 2nd ed., Vol. 2 (McGraw-Hill, 1994), based on Pettit & Turner and A. N. Pikhtin and "
        "A. D. Yas'kov, Sov. Phys. Semicond. 12, 622-626 (1978)",
        doi="10.1063/1.1714410", wl_range_um=(0.95, 10.0), notes="room temperature"),
    "skauli2003": Reference(
        "T. Skauli, P. S. Kuo, K. L. Vodopyanov, T. J. Pinguet, O. Levi, L. A. Eyres, J. S. Harris, M. M. Fejer, "
        "B. Gerard, L. Becouarn and E. Lallier, Improved dispersion relations for GaAs and applications to "
        "nonlinear optics, J. Appl. Phys. 94, 6447-6455 (2003)",
        doi="10.1063/1.1621740", wl_range_um=(0.97, 17.0), temperature_range_k=(295.15, 295.15),
        notes="22 degC model; the paper's temperature dependence is not implemented"),
    "zelmon1997": Reference(
        "D. E. Zelmon, D. L. Small and D. Jundt, Infrared corrected Sellmeier coefficients for congruently grown "
        "lithium niobate and 5 mol.% magnesium oxide-doped lithium niobate, J. Opt. Soc. Am. B 14, 3319-3322 "
        "(1997)", doi="10.1364/JOSAB.14.003319", wl_range_um=(0.4, 5.0), temperature_range_k=(294.15, 294.15),
        notes="congruent LiNbO3, 21 degC"),
    "weis1985": Reference(
        "R. S. Weis and T. K. Gaylord, Lithium niobate: summary of physical properties and crystal structure, "
        "Appl. Phys. A 37, 191-203 (1985)", doi="10.1007/BF00614817",
        notes="constant-strain (clamped) electro-optic coefficients at 633 nm and dielectric constants"),
    "malitson_dodge1972": Reference(
        "I. H. Malitson and M. J. Dodge, Refractive index and birefringence of synthetic sapphire, J. Opt. Soc. "
        "Am. 62, 1405 (1972) (abstract); dispersion formula in M. J. Dodge, Refractive Index, Handbook of Laser "
        "Science and Technology Vol. IV, Optical Materials Part 2 (CRC Press, 1986), p. 30",
        wl_range_um=(0.2, 5.0), temperature_range_k=(293.15, 293.15), notes="synthetic sapphire, 20 degC"),
    "devore1951": Reference(
        "J. R. Devore, Refractive indices of rutile and sphalerite, J. Opt. Soc. Am. 41, 416-419 (1951)",
        doi="10.1364/JOSA.41.000416", wl_range_um=(0.43, 1.53), notes="single-crystal rutile, room temperature"),
    "su8_3000": Reference(
        "MicroChem SU-8 3000 permanent epoxy negative photoresist data sheet (2011), Cauchy coefficients as "
        "transcribed in the refractiveindex.info database", wl_range_um=(0.32, 1.7),
        notes="manufacturer data, cure conditions unspecified; evidence level: data sheet only"),
    "rakic1998": Reference(
        "A. D. Rakic, A. B. Djurisic, J. M. Elazar and M. L. Majewski, Optical properties of metallic films for "
        "vertical-cavity optoelectronic devices, Appl. Opt. 37, 5271-5283 (1998)",
        doi="10.1364/AO.37.005271",
        notes="Lorentz-Drude fit (Table 1 parameters) to evaporated-film data, room temperature"),
    "beliaev2023": Reference(
        "L. Yu. Beliaev, E. Shkondin, A. V. Lavrinenko and O. Takayama, Optical properties of plasmonic titanium "
        "nitride thin films from ultraviolet to mid-infrared wavelengths deposited by pulsed-DC sputtering, "
        "thermal and plasma-enhanced atomic layer deposition, Opt. Mater. 143, 114237 (2023)",
        doi="10.1016/j.optmat.2023.114237", wl_range_um=(0.2114, 20.0),
        notes="50 nm pulsed-DC sputtered film on Si; TiN optics depend strongly on the deposition process"),
    "incropera": Reference(
        "F. P. Incropera, D. P. DeWitt, T. L. Bergman and A. S. Lavine, Fundamentals of Heat and Mass Transfer, "
        "6th ed. (Wiley, 2007), Tables A.1 (metallic solids), A.3 (nonmetallic solids) and A.4 (air), 300 K",
        temperature_range_k=(300.0, 300.0),
        notes="nominal bulk values; thin films are commonly several times less conductive"),
    "ioffe": Reference(
        "Ioffe Physico-Technical Institute, NSM Archive - Physical Properties of Semiconductors, basic parameters "
        "at 300 K (http://www.ioffe.ru/SVA/NSM/Semicond/)", temperature_range_k=(300.0, 300.0)),
    "sze2007": Reference(
        "S. M. Sze and K. K. Ng, Physics of Semiconductor Devices, 3rd ed. (Wiley, 2007), Appendix "
        "(properties of SiO2 and Si3N4 at 300 K)", notes="amorphous thermal oxide / CVD nitride films"),
    "crc_resistivity": Reference(
        "CRC Handbook of Chemistry and Physics, 95th ed. (CRC Press, 2014), Electrical Resistivity of Pure "
        "Metals, 293 K", temperature_range_k=(293.0, 293.0),
        notes="bulk, high-purity metals; deposited films are higher by grain-boundary and surface scattering"),
    "iacs": Reference(
        "International Annealed Copper Standard, IEC 60028 (1925): 1.7241e-8 ohm m and 0.00393 /K at 20 degC",
        temperature_range_k=(293.15, 293.15), notes="standard annealed copper (not high-purity bulk)"),
    "iec60751": Reference(
        "IEC 60751:2022, Industrial platinum resistance thermometers: mean temperature coefficient 0.00385 /K "
        "over 0-100 degC", notes="thermometer-grade platinum; pure Pt is ~0.0039 /K"),
}


# =========================================================================== #
# Helpers
# =========================================================================== #
def _formula(number: int, coeffs: list[float], ref: Reference, *, label: str,
             wl_range: tuple[float, float] | None = None, t_ref: float | None = None,
             t_range: tuple[float, float] | None = None, kind: str = "sellmeier") -> OpticalModel:
    c = list(coeffs)

    def evaluate(wl: Any, T: Any) -> Any:
        return refractiveindex_formula(number, c, wl)

    return OpticalModel(kind, evaluate, (ref,), wl_range or ref.wl_range_um, t_range, t_ref, False,
                        {"formula": number, "coefficients": tuple(c)}, label)


def _thermal(density: float, heat_capacity: float, conductivity: float) -> dict[str, Any]:
    return {"density": density, "heat_capacity": heat_capacity, "thermal_conductivity": conductivity}


# =========================================================================== #
# Silicon
# =========================================================================== #
_SI_SALZBERG = callable_model(
    _em.silicon, reference=REFERENCES["salzberg1957"], wl_range_um=(1.357, 11.04),
    temperature_range_k=_window(299.15), reference_temperature_k=299.15, kind="sellmeier",
    label="Si (Salzberg & Villa 1957)")

_SI_LI_T_RANGE = (100.0, 750.0)     # verified against Li's tables 100-750 K (formula stated to 1600 K)
_SI_LI_WL_RANGE = (1.2, 14.0)

_SI_COMMON: dict[str, Any] = dict(
    **_thermal(2330.0, 712.0, 148.0),
    relative_permittivity_static=11.7,
    references={"thermal": REFERENCES["incropera"], "static_permittivity": REFERENCES["ioffe"]},
    tags=("semiconductor", "crystal"),
)

#: Crystalline silicon: :func:`photonix.em.silicon` (Salzberg & Villa 1957, 26 degC, 1.357-11.04 um) with
#: the Li (1980) temperature shift ``n_Li(wl, T) - n_Li(wl, 299.15 K)`` (100-750 K). ``temperature=None``
#: returns em.silicon bit-for-bit. For the O band (below 1.357 um) use :data:`SI_LI1980`.
SI = MaterialSpec(
    "Si",
    with_temperature_shift(_SI_SALZBERG, li1980_silicon, reference_temperature_k=299.15,
                           temperature_range_k=_SI_LI_T_RANGE, shift_wl_range_um=_SI_LI_WL_RANGE,
                           reference=REFERENCES["li1980"]),
    description="Crystalline Si. Optical: em.silicon (Salzberg-Villa/Tatian) + Li (1980) thermal shift.",
    **_SI_COMMON,
)


def _li_si(wl: Any, T: Any) -> Any:
    return li1980_silicon(wl, 293.0 if T is None else T)


#: Crystalline silicon from the Li (1980) analytic n(wl, T): 1.2-14 um, 100-750 K, reference 293 K.
SI_LI1980 = MaterialSpec(
    "Si_Li1980",
    OpticalModel("li1980", _li_si, (REFERENCES["li1980"],), _SI_LI_WL_RANGE, _SI_LI_T_RANGE, 293.0, True,
                 {"formula": "li1980_silicon"}, "Si (Li 1980)"),
    description="Crystalline Si, Li (1980) n(wl, T); covers the O band.",
    **_SI_COMMON,
)

#: Intrinsic silicon n, k at 300 K, 0.25-1.45 um (Green 2008): band-to-band absorption below ~1.1 um.
SI_GREEN2008 = MaterialSpec(
    "Si_Green2008",
    load_refractiveindex_yaml(DATA_DIR / "Si_Green2008.yml", label="Si (Green 2008)",
                              reference=REFERENCES["green2008"], reference_temperature_k=300.0,
                              temperature_range_k=_window(300.0)),
    description="Intrinsic Si at 300 K with band-to-band absorption (visible/NIR).",
    **_SI_COMMON,
)

# =========================================================================== #
# Dielectrics
# =========================================================================== #
_SIO2_BASE = callable_model(_em.silica, reference=REFERENCES["malitson1965"], wl_range_um=(0.21, 6.7),
                            temperature_range_k=_window(293.15), reference_temperature_k=293.15,
                            kind="sellmeier", label="SiO2 (Malitson 1965)")

#: Silicon dioxide: :func:`photonix.em.silica` (Malitson 1965 fused silica, 20 degC) with a linear
#: thermo-optic coefficient of 0.95e-5 /K (Arbabi & Goddard 2013, PECVD SiOx at 1.55 um).
SIO2 = MaterialSpec(
    "SiO2",
    with_linear_thermo_optic(_SIO2_BASE, 0.95e-5, reference_temperature_k=293.15,
                             temperature_range_k=(273.15, 373.15), reference=REFERENCES["arbabi2013"]),
    **_thermal(2220.0, 745.0, 1.38),
    relative_permittivity_static=3.9,
    references={"thermal": REFERENCES["incropera"], "static_permittivity": REFERENCES["sze2007"]},
    tags=("dielectric", "cladding", "amorphous"),
    description="SiO2: fused-silica dispersion + deposited-oxide thermo-optic coefficient; thermal data for "
                "fused silica.",
)

_SIN_BASE = callable_model(_em.silicon_nitride, reference=REFERENCES["luke2015"], wl_range_um=(0.31, 5.504),
                           temperature_range_k=ROOM_TEMPERATURE_K, reference_temperature_k=295.15,
                           kind="sellmeier", label="Si3N4 (Luke 2015)")

#: Stoichiometric LPCVD silicon nitride: :func:`photonix.em.silicon_nitride` (Luke et al. 2015) with
#: dn/dT = 2.45e-5 /K (Arbabi & Goddard 2013).
SI3N4 = MaterialSpec(
    "Si3N4",
    with_linear_thermo_optic(_SIN_BASE, 2.45e-5, reference_temperature_k=295.15,
                             temperature_range_k=(273.15, 373.15), reference=REFERENCES["arbabi2013"]),
    **_thermal(2400.0, 691.0, 16.0),
    relative_permittivity_static=7.5,
    references={"thermal": REFERENCES["incropera"], "static_permittivity": REFERENCES["sze2007"]},
    tags=("dielectric", "core", "amorphous"),
    description="Stoichiometric LPCVD Si3N4. Thermal values are for bulk silicon nitride ceramic.",
)


def _air(wl: Any, T: Any) -> Any:
    sigma2 = 1.0 / (xp.asarray(wl, dtype=float) ** 2)
    refractivity = 0.05792105 / (238.0185 - sigma2) + 0.00167917 / (57.362 - sigma2)
    if T is None:
        return 1.0 + refractivity
    return 1.0 + refractivity * (288.15 / xp.asarray(T, dtype=float))


#: Standard air (Ciddor 1996, 15 degC, 101.325 kPa, 450 ppm CO2); the refractivity ``n - 1`` scales with the
#: ideal-gas density ``288.15 K / T`` at constant pressure (humidity and compressibility are neglected).
AIR = MaterialSpec(
    "air",
    OpticalModel("ciddor1996+ideal_gas", _air,
                 (REFERENCES["ciddor1996"],
                  Reference("ideal-gas density scaling of the refractivity at constant pressure",
                            notes="Lorentz-Lorenz in the dilute limit; ~1e-7 absolute accuracy for 233-373 K")),
                 (0.23, 1.69), (233.15, 373.15), 288.15, True,
                 {"coefficients": (0.05792105, 238.0185, 0.00167917, 57.362)}, "air (Ciddor 1996)"),
    **_thermal(1.1614, 1007.0, 0.0263),
    relative_permittivity_static=1.0,
    references={"thermal": REFERENCES["incropera"]},
    tags=("gas", "cladding"),
    description="Dry standard air.",
)

#: Vacuum: n = 1 exactly, at every wavelength and temperature.
VACUUM = MaterialSpec(
    "vacuum", constant_model(1.0, reference=Reference("exact")), relative_permittivity_static=1.0,
    tags=("gas",), description="n = 1 exactly.")

# =========================================================================== #
# Germanium and III-V
# =========================================================================== #
_GE_COMMON: dict[str, Any] = dict(
    **_thermal(5360.0, 322.0, 59.9),
    relative_permittivity_static=16.2,
    references={"thermal": REFERENCES["incropera"], "static_permittivity": REFERENCES["ioffe"]},
    tags=("semiconductor", "crystal", "absorber"),
)

#: Germanium n, k at 300 K, 0.188-2.48 um (Nunley et al. 2016): includes the absorption used by Ge
#: photodetectors (k = 0.087 at 1.31 um).
GE = MaterialSpec(
    "Ge",
    load_refractiveindex_yaml(DATA_DIR / "Ge_Nunley2016.yml", label="Ge (Nunley 2016)",
                              reference=REFERENCES["nunley2016"], reference_temperature_k=300.0,
                              temperature_range_k=_window(300.0)),
    description="Bulk Ge at 300 K (ellipsometry), with interband absorption.",
    **_GE_COMMON,
)


def _li_ge(wl: Any, T: Any) -> Any:
    return li1980_germanium(wl, 293.0 if T is None else T)


#: Germanium from the Li (1980) analytic n(wl, T) in the transparent region: 1.9-18 um, 100-550 K.
GE_LI1980 = MaterialSpec(
    "Ge_Li1980",
    OpticalModel("li1980", _li_ge, (REFERENCES["li1980"],), (1.9, 18.0), (100.0, 550.0), 293.0, True,
                 {"formula": "li1980_germanium"}, "Ge (Li 1980)"),
    description="Ge, Li (1980) n(wl, T), transparent region only.",
    **_GE_COMMON,
)

#: Indium phosphide, room temperature, 0.95-10 um (Pettit & Turner 1965 via Handbook of Optics).
INP = MaterialSpec(
    "InP",
    _formula(1, [6.255, 2.316, 0.6263, 2.765, 32.935], REFERENCES["pettit1965"], label="InP (Pettit 1965)",
             t_ref=295.65, t_range=ROOM_TEMPERATURE_K),
    density=4810.0, heat_capacity=310.0, thermal_conductivity=68.0, relative_permittivity_static=12.5,
    references={"thermal": REFERENCES["ioffe"], "static_permittivity": REFERENCES["ioffe"]},
    tags=("semiconductor", "crystal", "iii-v"),
    description="InP, room temperature.",
)

# Skauli et al. (2003), Eq. (1) at 22 degC: n^2 = A + sum_i g_i / (lambda_i^-2 - lambda^-2).
_SKAULI_A = 5.372514
_SKAULI_G = (27.83972, 0.031764, 0.00143636)
_SKAULI_LAM = (0.4431307, 0.8746453, 36.9166)


def _gaas(wl: Any, T: Any) -> Any:
    inv2 = 1.0 / (xp.asarray(wl, dtype=float) ** 2)
    n2 = _SKAULI_A + sum(g / (lam ** -2 - inv2) for g, lam in zip(_SKAULI_G, _SKAULI_LAM, strict=True))
    return xp.sqrt(n2)


#: Gallium arsenide at 22 degC, 0.97-17 um (Skauli et al. 2003).
GAAS = MaterialSpec(
    "GaAs",
    OpticalModel("skauli2003", _gaas, (REFERENCES["skauli2003"],), (0.97, 17.0), _window(295.15), 295.15, False,
                 {"A": _SKAULI_A, "g": _SKAULI_G, "lam": _SKAULI_LAM}, "GaAs (Skauli 2003)"),
    density=5320.0, heat_capacity=330.0, thermal_conductivity=55.0, relative_permittivity_static=12.9,
    references={"thermal": REFERENCES["ioffe"], "static_permittivity": REFERENCES["ioffe"]},
    tags=("semiconductor", "crystal", "iii-v"),
    description="GaAs at 22 degC.",
)


# =========================================================================== #
# Crystals: LiNbO3, sapphire, rutile
# =========================================================================== #
def _zelmon(a: float, b: float, c: float, d: float, e: float, f: float, label: str) -> OpticalModel:
    # n^2 - 1 = A l^2/(l^2 - B) + C l^2/(l^2 - D) + E l^2/(l^2 - F)   (Zelmon et al. 1997, Table 1)
    def evaluate(wl: Any, T: Any) -> Any:
        l2 = xp.asarray(wl, dtype=float) ** 2
        return xp.sqrt(1.0 + a * l2 / (l2 - b) + c * l2 / (l2 - d) + e * l2 / (l2 - f))

    return OpticalModel("sellmeier", evaluate, (REFERENCES["zelmon1997"],), (0.4, 5.0), _window(294.15), 294.15,
                        False, {"A": a, "B": b, "C": c, "D": d, "E": e, "F": f}, label)


#: Congruent lithium niobate (uniaxial, optic axis = crystal z), 0.4-5 um at 21 degC (Zelmon et al. 1997);
#: clamped electro-optic and dielectric constants from Weis & Gaylord (1985).
LINBO3 = MaterialSpec(
    "LiNbO3",
    Uniaxial(_zelmon(2.6734, 0.01764, 1.2290, 0.05914, 12.614, 474.60, "LiNbO3 o (Zelmon 1997)"),
             _zelmon(2.9804, 0.02047, 0.5981, 0.0666, 8.9543, 416.08, "LiNbO3 e (Zelmon 1997)")),
    density=4640.0,
    relative_permittivity_static=(44.3, 44.3, 27.9),
    pockels_pm_per_v={"r13": 8.6, "r22": 3.4, "r33": 30.8, "r51": 28.0},
    references={"electro_optic": REFERENCES["weis1985"], "static_permittivity": REFERENCES["weis1985"],
                "density": REFERENCES["weis1985"]},
    tags=("crystal", "electro-optic", "uniaxial"),
    description="Congruent LiNbO3. Static permittivity and r_ij are constant-strain (clamped) values.",
)


def _sapphire(coeffs: list[float], label: str) -> OpticalModel:
    return _formula(1, [0.0] + coeffs, REFERENCES["malitson_dodge1972"], label=label, t_ref=293.15,
                    t_range=_window(293.15))


#: Synthetic sapphire (uniaxial, crystal z = c axis), 0.2-5 um at 20 degC (Malitson & Dodge 1972).
#: Amorphous ALD/sputtered alumina films have a markedly lower index (~1.6 at 1.55 um).
AL2O3 = MaterialSpec(
    "Al2O3",
    Uniaxial(_sapphire([1.4313493, 0.0726631, 0.65054713, 0.1193242, 5.3414021, 18.028251], "sapphire o"),
             _sapphire([1.5039759, 0.0740288, 0.55069141, 0.1216529, 6.5927379, 20.072248], "sapphire e")),
    **_thermal(3970.0, 765.0, 46.0),
    references={"thermal": REFERENCES["incropera"]},
    tags=("crystal", "dielectric", "uniaxial"),
    description="Single-crystal sapphire.",
)


def _devore(a: float, b: float, c: float, label: str) -> OpticalModel:
    # n^2 = A + B / (lambda^2 - C)   (Devore 1951)
    def evaluate(wl: Any, T: Any) -> Any:
        l2 = xp.asarray(wl, dtype=float) ** 2
        return xp.sqrt(a + b / (l2 - c))

    return OpticalModel("sellmeier", evaluate, (REFERENCES["devore1951"],), (0.43, 1.53), ROOM_TEMPERATURE_K,
                        295.65, False, {"A": a, "B": b, "C": c}, label)


#: Rutile TiO2 (uniaxial, crystal z = c axis), 0.43-1.53 um, room temperature (Devore 1951). Thin-film
#: (amorphous/anatase) TiO2 has a lower, process-dependent index; 1.55 um is a slight extrapolation.
TIO2 = MaterialSpec(
    "TiO2",
    Uniaxial(_devore(5.913, 0.2441, 0.0803, "TiO2 rutile o (Devore 1951)"),
             _devore(7.197, 0.3322, 0.0843, "TiO2 rutile e (Devore 1951)")),
    **_thermal(4157.0, 710.0, 8.4),
    references={"thermal": REFERENCES["incropera"]},
    tags=("crystal", "dielectric", "uniaxial"),
    description="Rutile TiO2. Thermal data are for polycrystalline TiO2.",
)

#: SU-8 3000 epoxy (manufacturer Cauchy fit, 0.32-1.7 um).
SU8 = MaterialSpec(
    "SU-8",
    _formula(5, [1.5525, 0.00629, -2.0, 0.0004, -4.0], REFERENCES["su8_3000"], label="SU-8 3000 (data sheet)",
             t_ref=295.65, t_range=ROOM_TEMPERATURE_K, kind="cauchy"),
    tags=("polymer", "cladding"),
    description="SU-8 3000 photoresist (manufacturer data only).",
)

# =========================================================================== #
# Metals (Rakic et al. 1998 Lorentz-Drude)
# =========================================================================== #
# plasma energy wp [eV], (f0, Gamma0) [-, eV], oscillators (f_j, Gamma_j, w_j) [-, eV, eV]; fit range [eV]
_RAKIC_LD: dict[str, tuple[float, tuple[float, float], list[tuple[float, float, float]], tuple[float, float]]] = {
    "Ag": (9.01, (0.845, 0.048), [(0.065, 3.886, 0.816), (0.124, 0.452, 4.481), (0.011, 0.065, 8.185),
                                   (0.840, 0.916, 9.083), (5.646, 2.419, 20.29)], (0.1, 5.0)),
    "Au": (9.03, (0.760, 0.053), [(0.024, 0.241, 0.415), (0.010, 0.345, 0.830), (0.071, 0.870, 2.969),
                                   (0.601, 2.494, 4.304), (4.384, 2.214, 13.32)], (0.2, 5.0)),
    "Cu": (10.83, (0.575, 0.030), [(0.061, 0.378, 0.291), (0.104, 1.056, 2.957), (0.723, 3.213, 5.300),
                                    (0.638, 4.305, 11.18)], (0.1, 6.0)),
    "Al": (14.98, (0.523, 0.047), [(0.227, 0.333, 0.162), (0.050, 0.312, 1.544), (0.166, 1.351, 1.808),
                                    (0.030, 3.382, 3.473)], (0.005, 20.0)),
    "W": (13.22, (0.206, 0.064), [(0.054, 0.530, 1.004), (0.166, 1.281, 1.917), (0.706, 3.332, 3.580),
                                   (2.590, 5.836, 7.498)], (0.1, 5.0)),
    "Ti": (7.29, (0.148, 0.082), [(0.899, 2.276, 0.777), (0.393, 2.518, 1.545), (0.187, 1.663, 2.509),
                                   (0.001, 1.762, 19.43)], (0.04, 5.0)),
    "Cr": (10.75, (0.168, 0.047), [(0.151, 3.175, 0.121), (0.150, 1.305, 0.543), (1.149, 2.676, 1.970),
                                    (0.825, 1.335, 8.775)], (0.02, 5.0)),
    "Pt": (9.59, (0.333, 0.080), [(0.191, 0.517, 0.780), (0.659, 1.838, 1.314), (0.547, 3.668, 3.141),
                                   (3.576, 8.517, 9.249)], (0.1, 5.0)),
    "Ni": (15.92, (0.096, 0.048), [(0.100, 4.511, 0.174), (0.135, 1.334, 0.582), (0.106, 2.178, 1.597),
                                    (0.729, 6.292, 6.089)], (0.2, 5.0)),
}


def _rakic(metal: str) -> OpticalModel:
    wp, drude, osc, (e_lo, e_hi) = _RAKIC_LD[metal]

    def evaluate(wl: Any, T: Any) -> Any:
        return xp.sqrt(lorentz_drude_epsilon(wl, wp, drude, osc))

    return OpticalModel("lorentz_drude", evaluate, (REFERENCES["rakic1998"],), (HC_EV_UM / e_hi, HC_EV_UM / e_lo),
                        ROOM_TEMPERATURE_K, 295.65, False,
                        {"plasma_ev": wp, "drude": drude, "oscillators": tuple(osc)}, f"{metal} (Rakic 1998 LD)")


def _metal(name: str, thermal: tuple[float, float, float], resistivity: float | None = None,
           tcr: float | None = None, electrical_ref: str = "crc_resistivity", **extra: Any) -> MaterialSpec:
    refs = {"thermal": REFERENCES["incropera"]}
    t_rho = None
    if resistivity is not None:
        refs["electrical"] = REFERENCES[electrical_ref]
        t_rho = 293.15
    if tcr is not None and electrical_ref != "iacs":
        refs["electrical_tcr"] = REFERENCES["iec60751"]
    return MaterialSpec(name, _rakic(name), density=thermal[0], heat_capacity=thermal[1],
                        thermal_conductivity=thermal[2], resistivity=resistivity, resistivity_tcr=tcr,
                        resistivity_temperature_k=t_rho, references=refs, tags=("metal",),
                        description=f"{name}: Rakic (1998) Lorentz-Drude optics; bulk thermal/electrical data.",
                        **extra)


#: Aluminium (Rakic LD 0.062-248 um; Incropera 300 K; CRC 2.65e-8 ohm m at 293 K).
AL = _metal("Al", (2702.0, 903.0, 237.0), 2.65e-8)
#: Copper (Rakic LD 0.207-12.4 um; Incropera 300 K; IACS 1.7241e-8 ohm m and 0.00393 /K at 20 degC).
CU = _metal("Cu", (8933.0, 385.0, 401.0), 1.7241e-8, 0.00393, electrical_ref="iacs")
#: Gold (Rakic LD 0.248-6.2 um; Incropera 300 K; CRC 2.214e-8 ohm m at 293 K).
AU = _metal("Au", (19300.0, 129.0, 317.0), 2.214e-8)
#: Silver (Rakic LD 0.248-12.4 um; Incropera 300 K; CRC 1.587e-8 ohm m at 293 K).
AG = _metal("Ag", (10500.0, 235.0, 429.0), 1.587e-8)
#: Tungsten (Rakic LD 0.248-12.4 um; Incropera 300 K; CRC 5.28e-8 ohm m at 293 K).
W = _metal("W", (19300.0, 132.0, 174.0), 5.28e-8)
#: Titanium (Rakic LD 0.248-31 um; Incropera 300 K).
TI = _metal("Ti", (4500.0, 522.0, 21.9))
#: Chromium (Rakic LD 0.248-62 um; Incropera 300 K).
CR = _metal("Cr", (7160.0, 449.0, 93.7))
#: Platinum (Rakic LD 0.248-12.4 um; Incropera 300 K; CRC 10.5e-8 ohm m at 293 K; IEC 60751 TCR 0.00385 /K).
PT = _metal("Pt", (21450.0, 133.0, 71.6), 10.5e-8, 0.00385)
#: Nickel (Rakic LD 0.248-6.2 um; Incropera 300 K).
NI = _metal("Ni", (8900.0, 444.0, 90.7))

#: Titanium nitride, 50 nm pulsed-DC sputtered film, 0.21-20 um (Beliaev et al. 2023). Optical constants and
#: resistivity of TiN depend strongly on the deposition process; no electrical or thermal data are asserted.
TIN = MaterialSpec(
    "TiN",
    load_refractiveindex_yaml(DATA_DIR / "TiN_Beliaev2023.yml", label="TiN (Beliaev 2023)",
                              reference=REFERENCES["beliaev2023"], reference_temperature_k=295.65,
                              temperature_range_k=ROOM_TEMPERATURE_K),
    tags=("metal", "heater", "conductive-nitride"),
    description="Sputtered TiN film (one process); supply measured resistivity for heater design.",
)

# =========================================================================== #
# Registration
# =========================================================================== #
_ALIASES: dict[str, tuple[str, ...]] = {
    "Si": ("silicon", "c-Si"),
    "SiO2": ("silica", "silicon dioxide", "oxide", "fused silica"),
    "Si3N4": ("silicon nitride", "SiN", "nitride", "Si3N4_LPCVD"),
    "air": (),
    "vacuum": (),
    "Ge": ("germanium",),
    "InP": ("indium phosphide",),
    "GaAs": ("gallium arsenide",),
    "LiNbO3": ("lithium niobate", "LN", "TFLN"),
    "Al2O3": ("sapphire", "alumina"),
    "TiO2": ("rutile", "titanium dioxide"),
    "SU-8": ("SU8 3000",),
    "Al": ("aluminium", "aluminum"),
    "Cu": ("copper",),
    "Au": ("gold",),
    "Ag": ("silver",),
    "W": ("tungsten",),
    "Ti": ("titanium",),
    "Cr": ("chromium",),
    "Pt": ("platinum",),
    "Ni": ("nickel",),
    "TiN": ("titanium nitride",),
}

for _spec in (SI, SI_LI1980, SI_GREEN2008, SIO2, SI3N4, AIR, VACUUM, GE, GE_LI1980, INP, GAAS, LINBO3, AL2O3,
              TIO2, SU8, AL, CU, AU, AG, W, TI, CR, PT, NI, TIN):
    register(_spec, aliases=_ALIASES.get(_spec.name, ()), overwrite=True)
del _spec
