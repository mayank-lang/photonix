# Materials database (`photonix.materials`)

`photonix.materials` holds cited, range-checked, differentiable material models
for PIC design: complex refractive index (with temperature dependence and
uniaxial anisotropy where the literature supports it), plus the thermal,
electrical and electro-optic data that heaters, modulators and multiphysics
adapters need. Every entry carries its literature reference and validity range.

```python
from photonix.materials import get_material, list_materials

si = get_material("Si")
si.index(1.55)                 # n + i k (physics sign: k >= 0 is loss)
si.dn_dt(1.55, 300.0)          # thermo-optic coefficient, 1/K (Li 1980)
list_materials()               # registered names
```

## Material protocol

Other packages (for example `photonix.xsection`) accept any object with:

* `index(wl, temperature=None) -> n + i k`, wavelength in µm, temperature in K;
* `epsilon(wl, temperature=None)`;
* optional `thermal_conductivity` [W/(m K)], `heat_capacity` [J/(kg K)],
  `density` [kg/m^3], `relative_permittivity_static` and `dn_dt(wl, T)`.

**Sign convention.** The protocol returns the *physics* sign used by tabulated
n-k data (`n + i k`, `k >= 0` for loss). The `photonix.em` solvers propagate as
`exp(-j beta z)` and treat `Im(eps) < 0` as loss. Convert before handing a
lossy index to a solver, either with `to_engineering_sign` (complex
conjugation) or by passing `convention="engineering"` to
`MaterialSpec.index`, `epsilon` or `eps_tensor`.

**Range checks.** Evaluating outside a model's stated wavelength or temperature
range raises `MaterialRangeWarning` (or `MaterialRangeError` in strict mode).

## Library

| Name | Optical model (reference) |
|---|---|
| `Si` | Salzberg & Villa (1957) / Tatian Sellmeier via `photonix.em.silicon`; thermo-optic data from Li (1980) |
| `Si_Li1980`, `Ge_Li1980` | H. H. Li (1980), temperature-dependent n(λ, T) |
| `Si_Green2008` | M. A. Green (2008), tabulated n-k, 0.25-1.45 µm |
| `SiO2` | Malitson (1965) via `photonix.em.silica` |
| `Si3N4` | Luke et al. (2015) via `photonix.em.silicon_nitride` |
| `Ge` | Nunley et al. (2016), tabulated n-k |
| `InP` | Pettit & Turner (1965) |
| `GaAs` | Skauli et al. (2003) |
| `LiNbO3` | Zelmon et al. (1997), ordinary/extraordinary (`Uniaxial`) |
| `Al2O3`, `TiO2` | Malitson & Dodge (1972); DeVore (1951) |
| `SU-8` | MicroChem SU-8 3000 data sheet |
| `Al`, `Cu`, `Au`, `Ag`, `W`, `Ti`, `Cr`, `Pt`, `Ni` | Rakic et al. (1998) Lorentz-Drude |
| `TiN` | Beliaev et al. (2023) |
| `air`, `vacuum` | Ciddor (1996); exact |

`photonix.em.silicon`, `silica` and `silicon_nitride` remain the canonical
Sellmeier implementations; the library entries wrap them (one implementation
per concept).

## Tools

* `load_refractiveindex_yaml`, `parse_refractiveindex_yaml`, `load_nk_csv` --
  import refractiveindex.info YAML (formula types 1-9, tabulated n/k/nk) or CSV.
  Loaded models always use vacuum wavelengths and absolute indices. Files whose
  `SPECS` say `n_is_absolute: false` or `wavelength_is_vacuum: false` (for
  example catalogue glasses and Malitson-type fits) are converted with
  Ciddor's standard-air model at the data temperature:
  `N(l) = N_rel(l / n_air(l)) n_air(l)`. The conversion is a 3e-4 to 1e-3
  change. Pass `to_vacuum=False` to use the stored numbers as written.
* `register`, `unregister`, `get_material`, `list_materials` -- registry.
* `doped_si`, `soref_bennett_1987`, `nedeljkovic_2011` -- free-carrier index
  and absorption changes of doped silicon.
* `lorentz_lorenz_mix`, `sion` -- effective-medium mixtures (SiON).
* `fit_lorentz` -> `LorentzFit` -- passive multi-pole Lorentz-Drude fits over a
  band, exportable to Meep media. Meep's `exp(-i omega t)` media already treat
  `Im(eps) > 0` as loss, the same sign as this protocol, so the parameters are
  passed through unchanged.
* `kramers_kronig_n` -- consistency check of tabulated n against k.

## Validation

`tests/test_materials_db.py` checks every entry against values published in
its source (and against refractiveindex.info fixture files in
`tests/data/refractiveindex_info/`). It also covers the range warnings,
temperature derivatives against finite differences, Lorentz fits for passivity
and accuracy, Kramers-Kronig consistency, and JAX differentiability.

## Limits

The data are literature values for bulk or reference films. Deposited films
(PECVD nitride, sputtered oxides, thin silicon) differ by process. Use foundry
or ellipsometry data for sign-off.
