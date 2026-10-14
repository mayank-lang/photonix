# Command-line interface

`photonix` (or `python -m photonix`) runs the common design and hand-off tasks
without a script: a quick mode solve, sweeping a model to Touchstone, gating a
vendor `.sNp` file on passivity, inspecting and pre-checking a GDS file,
simulating a netlist, and certifying an installation against analytic
references. The commands are thin wrappers over library calls (named below) and
add no numerics of their own; the tests check each one against the direct
library call on identical arguments.

```bash
photonix COMMAND [options] [--json]
photonix COMMAND --help          # every option of one command
```

## Exit codes and JSON

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | the command ran and a check it performs failed (non-passive data, DRC violations, a failed self-check, an unconverged study with `--require-converged`) |
| 2 | usage or input error: bad arguments, an unreadable or malformed file, an unknown model or PDK |
| 3 | an optional dependency the command needs is missing (the message names the `pip install` extra) |
| 4 | internal error: an unexpected exception, printed with its traceback. Please report it with `photonix info --json` |
| 141 | stdout was closed early by the reader (for example `photonix ... --json \| head`); the command stops quietly, like a process killed by `SIGPIPE` |

`--json` prints one JSON document (strict JSON, keys sorted) instead of text.
Its `schema` field is `photonix.cli/<command>/1`: `info`, `models`, `mode`,
`sweep`, `sparams-check`, `sparams-convert`, `gds-inspect`, `gds-precheck`,
`netlist-simulate` and `validate`. When a command fails under `--json`, stdout
carries a `photonix.cli/error/1` document with `exit_code` and the message;
the human-readable message always goes to stderr. Non-finite numbers are
encoded as the strings `"inf"`, `"-inf"` and `"nan"`, and complex numbers as
`[re, im]`. The version in a schema identifier changes only when a field is
removed or changes meaning. The frozen JSON Schema (draft 2020-12) of every
document is in
[`tests/data/cli_schema/`](../tests/data/cli_schema/), and the test suite
validates every document the commands produce against it.

## Commands

### `photonix info`

The numerical backend, precision, devices, optional Python packages and KLayout
discovery (`photonix.runtime_info()`). This is also what `photonix` prints with
no command, as in 0.1.

### `photonix models [NAME] [--no-ports]`

The compact-model registry `photonix.components.MODELS`: for each model its
function, one-line summary, ports (from one evaluation at the defaults and
`wl = 1.55`), keyword parameters with their defaults, and the waveguide
parameters a PDK cross-section binds. `NAME` shows one model with its full
docstring; an unknown name suggests close matches.

### `photonix mode`

Full-vector FDE modes (`photonix.em.solve_modes_fullvector`).

```bash
photonix mode --width 0.5 --thickness 0.22 --core Si --clad SiO2 --wl 1.55 --modes 2
photonix mode --pdk demo --xs strip --resolution 30
photonix mode --resolution 40 --converge --require-converged
```

`--core`/`--clad` take a number or `Si`, `SiO2`, `Si3N4`, `air` (Sellmeier fits
of `photonix.em.materials` evaluated at `--wl`). The defaults (3.4757 / 1.444,
0.5 x 0.22 µm, resolution 40, margin 1.5) are those of
`photonix.em.n_eff_fullvector`, so `photonix mode` with no options reproduces
it to machine precision. With `--pdk`, the geometry and materials come from the
kit's layer stack and cross-section. Each mode reports `n_eff`, its imaginary
part, the TE fraction (`> 0.5` is TE-like) and whether it is guided.
`--converge` adds a three-level grid study at `resolution x (1, 1.5, 2.25)`
through `photonix.em.estimate_convergence`: the extrapolated value, observed
order, grid-convergence index and whether the study is asymptotic and
converged within `--rtol`. A non-monotone sequence is reported as not
converged rather than extrapolated; `--require-converged` turns that into exit
code 1.

### `photonix sweep MODEL`

Evaluate a registered model over wavelength:

```bash
photonix sweep ring_single --wl 1.54:1.56:2001 -p radius=8 -p gap=0.22 -o ring.s2p
photonix sweep bragg_grating -p apodization=gaussian -p n_sections=40 -o bragg.csv
photonix sweep straight --pdk demo -p length=1000 --wl 1.55 --json
```

`--wl` is `START:STOP:COUNT` (both ends included), a comma list, or one value,
in µm. `-p NAME=VALUE` sets a parameter; values are JSON literals (`25`,
`2.45`, `true`, `null`, `[1, 2]`) or plain strings, and unknown names are
rejected with the list of valid ones. `--pdk demo [--xs NAME]` uses the kit's
model registry, which binds the cross-section's `neff`, `ng`, `loss_db_cm` and
`wl0`. The output reports every non-zero path's minimum and maximum `|S|²` in
dB and a physicality summary (largest singular value, worst reciprocity and
unitarity errors, and the wavelengths where they occur).

`-o FILE` writes the S-parameters in the format the suffix names. The suffix,
the target directory and whether the file already exists are checked before
any model is evaluated; an existing file is never replaced unless `--force` is
given (the same holds for the output of `sparams convert`).

| Suffix | Format |
|---|---|
| `.sNp` | Touchstone 1.0 `S RI` (`N` must equal the port count); port names and metadata in comments |
| `.npz` | `SParameterDataset.save_npz` (pickle-free, versioned) |
| `.json` | `photonix.sparameters` version 1: `wavelengths_um`, `ports`, `s_real`, `s_imag` with `s[k][out][in]`, `metadata` |
| `.csv` | `wl_um` then `IN->OUT:re`, `IN->OUT:im` columns for every port pair (output only) |

Without `-o`, `--json` includes the same `photonix.sparameters` document under
`data`.

### `photonix sparams check FILE`

Measure passivity, reciprocity and losslessness at every wavelength
(`photonix.analyze_sparameters`) of a `.sNp`, `.npz` or `.json` file, and exit
1 when a gated check fails. Passivity and reciprocity are gated, losslessness is
reported (gate it with `--lossless`; skip the reciprocity gate for
non-reciprocal devices with `--no-reciprocity`). `--atol` (default `1e-6`) sets
every tolerance; `--passivity-atol`, `--reciprocity-atol` and `--lossless-atol`
override one. The check assumes power-normalized waves with equal real reference
impedances, which is the photonic convention (see
[conventions](theory/conventions.md)); measured data usually needs a looser
tolerance such as `1e-4`.

### `photonix sparams convert IN OUT`

Convert between `.sNp`, `.npz` and `.json` (and write `.csv`). Port names and
metadata survive every conversion except to CSV. Values are written with 17
significant digits, so S-parameters round-trip exactly and wavelengths to the
rounding of the wavelength-frequency conversion.

### `photonix gds inspect FILE [--cell NAME]`

Every cell of a GDSII library (polygon, path, reference and label counts and
its bounding box) and, for the top cell or `--cell`, the flattened layers with
polygon counts and drawn area (the sum of polygon areas, so overlaps count
twice), the bounding box and the labels. Coordinates are converted from the
library's user unit to µm. A library with several top cells needs `--cell`.

### `photonix gds precheck FILE (--pdk NAME | --rules FILE)`

Flatten one cell and run the minimum width, space and area pre-check of
`photonix.layout.run_drc_precheck` with a PDK's design rules, a rules file, or
both (the file wins per layer and rule). Exit 1 on any violation; every
violation is listed with its bounding box (the first `--max-violations`, 100 by
default; the counts are always complete). Layers without rules are named as
unchecked. A rules file maps `LAYER/DATATYPE` (or `LAYER`) to rules in µm and
µm²:

```json
{"format": "photonix.drc_rules", "version": 1,
 "rules": {"1/0": {"min_width": 0.15, "min_space": 0.15}, "11/0": {"min_area": 4.0}}}
```

(the bare inner mapping is accepted as well). This is a fast pre-check, not
foundry sign-off; see `photonix.layout.run_drc` for running a foundry deck in
KLayout.

Both `gds` actions need the optional `gdstk` package and exit 3 without it.

### `photonix netlist simulate FILE`

Load a `Netlist.to_json` file, compile it with
`photonix.circuit.circuit_from_netlist` against `photonix.components.MODELS`
(or `--pdk NAME`'s registry), solve it at every `--wl` and report or save the
external-port S-parameters exactly like `sweep`. `-s INSTANCE.NAME=VALUE`
overrides one instance setting at call time; netlists that reference unknown
models are rejected before any simulation.

### `photonix validate [--quick] [--check NAME]`

Run the packaged self-checks and print them with `photonix info` as a
reproducibility manifest; exit 1 if any check fails. A check that raises is
reported as failed with the exception, never as a crash.

| Check | Reference | Tolerance | Evidence |
|---|---|---|---|
| `ring_allpass_closed_form` | netlist-solved all-pass ring equals `(t - a e^{-j phi}) / (1 - t a e^{-j phi})` over 401 wavelengths (Yariv, *Electron. Lett.* 36, 321 (2000); Bogaerts et al., *Laser Photonics Rev.* 6, 47 (2012)) | 1e-12 complex amplitude | 2 analytic |
| `mzi_transfer_matrix` | two-coupler MZI netlist equals the coupler-arm-coupler matrix product and conserves power | 1e-12 | 2 analytic |
| `bragg_peak_reflectance` | uniform-grating peak reflectance `tanh²(kappa L)` (Kogelnik and Shank, *J. Appl. Phys.* 43, 2327 (1972); Erdogan, *J. Lightwave Technol.* 15, 1277 (1997)) | 1e-6 relative | 2 analytic |
| `slab_te0_analytic`, `slab_tm0_analytic` | finite-difference slab TE0/TM0 `n_eff` at 80 points/µm against the exact dispersion relation | 1e-3 relative | 2 analytic |
| `models_passive_reciprocal` | every registry model at its defaults is passive and reciprocal at 5 wavelengths | 1e-9 | 1 physicality |
| `fullvector_soi_strip_te0` (full run only) | full-vector FDE TE0 of a 500 x 220 nm Si/SiO2 strip lies in 2.45 ± 0.05 (Chrostowski and Hochberg, *Silicon Photonics Design*, 2015) | inside the band | 4, coarse band |

`--quick` runs the closed-form checks only (a few seconds; well under one on
the NumPy backend). On the reference machine every error is at the 1e-15 level
except the slab checks (about 1.5e-4) and the full-vector strip (2.449 at
resolution 40).

## Validation of the command line itself

`tests/test_cli.py` checks, among other things, that

* `photonix mode` equals `photonix.em.n_eff_fullvector()` to 1e-12, and the
  Sellmeier and PDK variants equal the corresponding direct solves;
* `sweep` JSON equals the direct model evaluation to 1e-15, and every output
  format reloads to the same matrix;
* hand-written passive and non-passive Touchstone fixtures
  (`tests/data/cli/*.s2p`) exit 0 and 1 with the exact violation (0.05) at the
  right wavelength; `sweep straight -o x.s2p` followed by `sparams check x.s2p`
  exits 0;
* the DRC pre-check of a clean and a violating ring matches `Pdk.drc_precheck`,
  and GDS library units are converted to µm;
* with `gdstk` hidden, the `gds` commands exit 3 without a traceback;
* `python -m photonix` and the console-script entry point give identical
  output, and every JSON document validates against its frozen schema.

## Limits

* `photonix mode` solves rectangular strips and PDK layer-stack cross-sections
  only; ribs, slots and user geometries need the Python API.
* `sweep` and `netlist simulate` evaluate the compact models as they are: the
  physicality summary checks passivity and reciprocity, not that a model is
  calibrated for your process.
* The Touchstone reader is the internal Touchstone 1.0 `S RI` subset; MA/DB
  formats, Touchstone 2.x and noise data need scikit-rf
  (`SParameterDataset.from_skrf`).
* `gds precheck` checks minimum width, space and area only, on one flattened
  cell; it is not a substitute for a foundry DRC deck.
* `validate` checks anchors (evidence level 2, and a coarse level-4 band); it
  does not qualify a solver for a particular device.
* Heater, thermal and extraction commands will follow the corresponding library
  features.
