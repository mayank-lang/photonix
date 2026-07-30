# External integration guide

Photonix keeps optional and licensed software at explicit boundaries. Open-source
tools are used where they provide a strong, documented interface; proprietary
tools receive portable artifacts or an explicit user-configured adapter. No
foundry deck, vendor SDK, credential, license token, or calibration table is
bundled with the package.

## Capability map

| Need | Preferred integration | Photonix boundary | External prerequisite |
|---|---|---|---|
| GDSII/OASIS | gdstk | `write_gds`, `read_gds`, `write_oas`, `read_oas` | `photonix[layout]` |
| DRC/LVS | KLayout | `run_drc`, `run_lvs`, `run_klayout_deck` | KLayout executable and a user/foundry deck |
| Optical FDTD | MIT Meep + MPB | native polygons, modal ports, full modal S matrix | conda-forge `pymeep` with MPB |
| Sampled network data | Touchstone + scikit-rf | internal `S RI` reader/writer; optional `skrf.Network` | none for 1.0 RI; `photonix[rf]` for scikit-rf |
| Process variation | NumPy | absolute corners, covariance-validated Monte Carlo | calibrated parameters supplied by the PDK/user |
| Thermal/electrical FEM | Elmer | prepared `.sif` job and field-result contract | `ElmerSolver` and a validated deck |
| Carrier/TCAD | DEVSIM | prepared Python job and field-result contract | DEVSIM and a validated script |
| Licensed multiphysics | Ansys Lumerical DEVICE or site tool | import-safe license-aware adapter | vendor install, API, and license |
| Other licensed tools | vendor-neutral handoff/argv adapter | OASIS, Touchstone, NPZ fields, explicit metadata | user-configured executable/module and license |

## Mask verification

```python
import photonix.layout as lay

lay.write_oas(top, "chip.oas", compression_level=6, validation="crc32")
result = lay.run_drc(
    "chip.oas",
    "/secure/pdk/rules/foundry.lydrc",
    report_path="drc.lyrdb",
    variables={"threads": 8},
)
```

The KLayout runner uses an argv sequence and `shell=False`. It treats the deck as
opaque and passes absolute input/report paths through configurable runtime
variables. The same mechanism runs LVS, including a reference-netlist path when
the supplied deck exposes one as a runtime variable.

A zero return code means that KLayout executed the deck. It does not mean the
report contains zero violations or that a foundry accepts the result. Report
schema, waivers, density/fill, device recognition, connectivity, and the clean
criterion belong to the qualified deck and release procedure.

OASIS/GDSII preserve mask geometry and labels, not the complete Photonix port
model. The current reader returns a flattened top cell; label-derived ports do
not recover original width, orientation, datatype, hierarchy, or model metadata.
Keep the native design/manifest as the semantic source of truth.

## Complete modal matrices with Meep

```python
from photonix.em import meep

prepared = meep.prepare_layout(
    top,
    [
        meep.LayerSpec((1, 0), epsilon=3.48**2, thickness=0.22, z_center=0.0),
        meep.LayerSpec((2, 0), epsilon=1.44**2, thickness=2.0, z_center=-1.11),
    ],
    margin=(1.5, 1.5, 1.5),
)
plan = meep.plan_multiport(
    prepared,
    [1.50, 1.55, 1.60],
    port_modes={"o1": (1, 2), "o2": (1, 2), "drop": (1,)},
)
print(plan.terminal_names, plan.run_count)

dataset = meep.simulate_multiport_sparameters(
    prepared,
    wavelengths=plan.wavelengths,
    resolution=30,
    port_modes={
        "o1": meep.PortModeSpec((1, 2), monitor_offset=0.2),
        "o2": meep.PortModeSpec((1, 2), monitor_offset=0.2),
        "drop": meep.PortModeSpec((1,), monitor_offset=0.2),
    },
    reference=straight_reference,
    pml=1.0,
)
```

For a terminal `(p, m)`, let `a[p,m]` be the coefficient travelling inward
relative to physical port `p` and `b[q,n]` the coefficient travelling outward
at port `q`. The reported convention is

```text
S[(q,n),(p,m)] = b[q,n] / a[p,m]
dataset.s[:, outgoing, incoming]
```

With a reference run, the denominator is `a_ref[p,m]`. The reference's complete
backward modal vector is subtracted at the incident physical port before the
division. Every incoming modal terminal is simulated; reciprocity and geometric
symmetry are never used to fill missing columns.

MPB power-normalizes propagating modes, so `abs(S)**2` is a modal power fraction
only when both channels are propagating and the decomposition is well conditioned.
Near cutoff, evanescent modes, degenerate modes, and non-orthogonal lossy modes
need separate interpretation. MPB's Meep interface ignores magnetic,
conductive, nonlinear, and dispersive material terms during mode launch and
decomposition. Also converge resolution, PML, source/monitor offsets, modal basis,
reference planes, and decay tolerance. The layout adapter currently accepts only
axis-aligned mode planes.

## Touchstone and RF tooling

```python
dataset.save_touchstone("device.s5p", reference_impedance=50.0)
restored = type(dataset).load_touchstone("device.s5p")

# Optional richer analysis/calibration/de-embedding ecosystem:
network = dataset.to_skrf(name="device")
back = type(dataset).from_skrf(network)
```

The dependency-free implementation is intentionally Touchstone 1.0,
single-ended `S RI`. It writes increasing frequency, the legacy two-port order
`S11,S21,S12,S22`, and row-major matrices for three or more ports. Port names and
Photonix metadata live in ignorable comments. Touchstone 2.x, mixed-mode, noise,
MA/DB, per-port/frequency-dependent impedance, calibration, and de-embedding
belong in the optional scikit-rf path.

Touchstone's reference resistance is interoperability metadata here. A
power-normalized optical eigenmode is not a TEM voltage/current port, so a
nominal 50-ohm value must not be interpreted as a derived optical wave impedance.

## Process corners and Monte Carlo

```python
import photonix.em as em
from photonix.pdk import MonteCarloSpec, Pdk, ProcessCorner, ProcessStudy

nominal = ProcessCorner("nominal", {"width_um": 0.50, "height_um": 0.22})
study = ProcessStudy(
    nominal,
    corners=(
        ProcessCorner("narrow_thin", {"width_um": 0.48, "height_um": 0.21}),
        ProcessCorner("wide_thick", {"width_um": 0.52, "height_um": 0.23}),
    ),
    monte_carlo=MonteCarloSpec.independent(
        nominal.parameters,
        {"width_um": 0.005, "height_um": 0.003},
    ),
    units={"width_um": "um", "height_um": "um"},
)
cases = study.cases(monte_carlo_samples=100, seed=7)
neff_by_case = study.evaluate(
    lambda case: em.n_eff_fullvector(
        wl=1.55,
        width=case.parameters["width_um"],
        thickness=case.parameters["height_um"],
    ),
    monte_carlo_samples=100,
    seed=7,
)
pdk = Pdk("target-foundry")
pdk.add_process_study("waveguide-2026q3", study)
```

Corner values are absolute, not implicit deltas. Covariance must be finite,
symmetric, and positive semidefinite; seeded sampling does not modify NumPy's
global random state. `evaluate`/`map` preserve case names while an arbitrary
layout, compact-model, or external-solver callback interprets the absolute
parameters; a `Pdk` can register named studies. The Gaussian sampler is
mathematically unbounded and has no
spatial correlation, wafer hierarchy, truncation, or acceptance criterion.
Production distributions and correlation matrices must therefore come from a
qualified PDK or measured calibration.

## Electrothermal and carrier workflow

```python
import json
from pathlib import Path

from photonix.multiphysics import (
    ElmerAdapter,
    FieldDataset,
    LinearIndexModel,
    LinearResponseTerm,
    MultiphysicsJob,
    Physics,
    run_job,
)

calibration_record = json.loads(Path("qualified-index-calibration.json").read_text())
job = MultiphysicsJob(
    "heater",
    {Physics.ELECTROTHERMAL},
    "case.sif",
    required_outputs=("heater.vtu",),
    timeout_s=1800,
)
result = run_job(ElmerAdapter(), job)
fields = FieldDataset.from_meshio(
    result.output("heater.vtu"),
    fields=("temperature",),
    units={"temperature": "K"},
    coordinate_unit="m",
)
model = LinearIndexModel(
    reference_index=3.48,
    terms=(
        LinearResponseTerm(
            "temperature",
            reference=float(calibration_record["reference_temperature_K"]),
            coefficient=float(calibration_record["dn_dT_per_K"]),
            field_unit="K",
            coefficient_unit="1/K",
        ),
    ),
    provenance="replace-with-calibration-report-id",
)
optical_fields = model.apply(fields)
optical_fields.save_npz("heater-optical-fields.npz")
```

`DevsimAdapter` uses the same job/result contract for prepared DEVSIM Python
scripts. `LumericalDeviceAdapter` detects a user-installed `lumapi` module or
explicit module file without importing it or opening a licensed session during
capability checks. `ExternalSolverAdapter` accepts an explicit argv template and
can require a configured license environment variable for other site tools.

The field contract requires explicit coordinate and field units and performs no
hidden conversion. The linear response is

```text
n(x) = n_ref + sum_k c_k * (u_k(x) - u_ref,k)
epsilon_r(x) = n(x)^2
```

Calibration provenance is mandatory and no material coefficient is built in.
This is suitable for locally linear thermo-optic, electro-optic, or carrier
responses over a calibrated range. Nonlinear free-carrier dispersion,
temperature-dependent conductivity, self-heating feedback, stress, anisotropy,
gain, and hysteresis require a solver/model-specific coupling loop. Always make
the complex-index sign and harmonic time convention agree with the optical
solver before interpreting absorption or gain.

## Licensed handoffs

`photonix.interop.ExternalSolverHandoff` is a versioned JSON metadata record for
an external boundary. It records the tool, interface, artifact format, required
module/executable, whether a license is required, and workflow metadata. It is
not a command runner and deliberately cannot contain a license token.

For commercial verification or simulation, export OASIS/GDSII, Touchstone, or
the field NPZ; configure invocation in the tool-owning environment; retain tool
version, deck/version checksum, license feature, process corner, convergence
settings, and output checksum in the project manifest. That preserves
reproducibility without redistributing protected assets.
