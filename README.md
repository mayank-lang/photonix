# photonix

Photonix is an accuracy-first Python toolkit for photonic integrated circuit
(PIC) design. It connects differentiable compact models, S-parameter circuit
simulation, electromagnetic solvers, layout, process studies, and external
verification handoffs through one consistent port and data model.

| Layer | Delivered capability |
|---|---|
| Circuit | Differentiable named-port S-matrix assembly, including feedback networks |
| Electromagnetics | Scalar, semivectorial, and full-vector FDE; 2-D FDFD and EME; optional Meep/MPB FDTD |
| Accuracy & spectra | Adaptive Richardson/GCI studies; high-order group delay and GDD on nonuniform sweeps |
| Physicality QA | Quantitative passivity, reciprocity, and losslessness checks plus pointwise passive projection |
| Design | JAX gradients, inverse-design objectives, fabrication filters, and process variation studies |
| Layout & exchange | Hierarchical cells, routing, GDSII/OASIS, Touchstone, extraction, and KLayout handoff |

> Status: **v0.1.0 (beta).** MIT-licensed. Photonix is a simulation and design
> layer, not a substitute for a target foundry's qualified PDK, DRC/LVS decks,
> or tape-out sign-off.


## Install

```bash
pip install "photonix[all]"   # full feature set (jax, layout/viz, scikit-rf)
# or just the GDSII/OASIS serializer:
pip install "photonix[layout]"
# or the optional scikit-rf bridge (internal Touchstone RI needs no extra):
pip install "photonix[rf]"
# or install the minimal NumPy/SciPy core:
pip install photonix
```

For an editable checkout, clone the repository and run
`pip install -e ".[all]"` from its root.

photonix runs on a NumPy fallback backend when JAX is absent (autodiff disabled);
install the `jax` extra for the full differentiable experience.

Confirm the active backend, floating-point precision, devices, optional Python
packages, and KLayout discovery before a reproducibility-sensitive run:

```bash
photonix info
photonix info --json       # suitable for a simulation manifest or bug report
# `python -m photonix info` is equivalent
```

### Optional: FDTD via MEEP

photonix does not ship its own time-domain Maxwell solver. Workflows that need
FDTD are delegated to [MIT MEEP](https://meep.readthedocs.io) through the optional
`photonix.em.meep` backend. MEEP is conda-only (Linux/macOS; on Windows use WSL2),
so it is **not** a pip dependency:

```bash
conda install -c conda-forge pymeep        # or "pymeep=*=mpi_mpich_*" for the parallel build
```

Without MEEP, the package and the pure `photonix.em.meep` geometry/port
specifications remain importable. Only calls that construct MEEP objects or run
MPB/FDTD raise an `ImportError` with this install hint.

### Optional: process studies and external multiphysics

`photonix.pdk` provides validated absolute process corners and correlated
Gaussian Monte Carlo specifications. `photonix.multiphysics` provides import-safe
job, capability, command, and subprocess-result contracts for prepared Elmer
`.sif` cases, DEVSIM Python scripts, Ansys Lumerical/DEVICE `lumapi` scripts,
and configurable site/licensed solvers.
Photonix does not bundle those solvers, generate uncalibrated material decks, or
invent solver-specific observables or claim that unqualified results are calibrated:

```python
from photonix.multiphysics import ElmerAdapter, MultiphysicsJob, Physics, run_job

job = MultiphysicsJob("heater", {Physics.ELECTROTHERMAL}, "case.sif")
adapter = ElmerAdapter()
if adapter.capability().available:
    result = run_job(adapter, job)  # argv execution; shell=False
```

Solver fields can cross the tool boundary through `FieldDataset`: mesh coordinate
units and every scalar-field unit are mandatory, NPZ round-trips are pickle-free,
and optional `meshio` VTU/mesh reading happens only when called. A
`LinearIndexModel` requires caller-supplied coefficients and calibration
provenance before mapping temperature/carrier fields to scalar index and
permittivity; `FieldDataset.sample` then interpolates onto optical grid points
with an explicit out-of-domain policy.

The Lumerical adapter follows Ansys's documented external-Python boundary: the
user installation must make `lumapi` discoverable (or supply its installed
`lumapi.py` path), and executing a prepared script may check out a licensed
product session. Photonix does not ship the SDK or probe licenses by launching a
session. See the official [Ansys Python API overview](https://optics.ansys.com/hc/en-us/articles/360037824513-Python-API-overview)
and [installation guide](https://optics.ansys.com/hc/en-us/articles/39744901602707-Installation-and-Getting-Started-Python-API).

### Optional: OASIS and external KLayout verification

`photonix.layout.write_oas` and `read_oas` use the optional `gdstk` layout
extra. For DRC/LVS, Photonix can invoke a separately installed KLayout executable
with a deck supplied by the user or foundry:

```python
import photonix.layout as lay

lay.write_oas(top, "chip.oas", validation="crc32")
result = lay.run_drc(
    "chip.oas",
    "/secure/pdk/rules/foundry.lydrc",
    report_path="drc.lyrdb",
    variables={"threads": 8},
)
```

Set `KLAYOUT_EXECUTABLE` or pass `executable=` if KLayout is not on `PATH`.
Photonix never bundles, reads, copies, or rewrites the deck. A zero process exit
means the deck ran successfully; only that deck's report defines whether the
layout is sign-off clean.

## Example

```python
import photonix as px

wl = px.linspace(1.50, 1.60, 501)          # wavelength sweep [µm]

# Build a Mach–Zehnder interferometer and simulate it differentiably
mzi = px.circuit.mzi(delta_length=20.0)     # µm path imbalance
S = mzi(wl=wl)
T = px.power(S[("o1", "o4")])               # bar transmission spectrum

# Gradient of mean transmission w.r.t. the path imbalance — for free
def fom(dl):
    return px.power(px.circuit.mzi(delta_length=dl)(wl=wl)[("o1", "o4")]).mean()
g = px.grad(fom)(20.0)
```

Cross-section physics, and the layout → simulation loop:

```python
import photonix.em as em
import photonix.layout as lay
from photonix.layout import components as lc

em.n_eff_fullvector(wl=1.55, width=0.5, thickness=0.22)   # ≈2.45, SOI TE0

top = lay.Cell("top")
top.add_ref(lc.straight(10.0), origin=(0, 0), name="a")
top.add_ref(lc.straight(10.0), origin=(10, 0), name="b")
nl = lay.extract_netlist(top)               # coincident ports → connections
S  = px.circuit.circuit_from_netlist(nl)(wl=1.55)   # models default to px.components.MODELS
```

## Accuracy is a workflow

No single discretized result is "accurate" without evidence at the geometry and
operating point being studied. For engineering use, sweep grid resolution and
domain/PML size, sweep EME modal basis size where applicable, test S-parameter
passivity and reciprocity under the correct power-wave normalization, and retain
the runtime manifest with the result. Cross-check critical devices against an
independent solver or measurement.

[`docs/VALIDATION.md`](docs/VALIDATION.md) defines the evidence levels, solver
selection guidance, and a result-qualification checklist. The executable
[`examples/accuracy_workflow.py`](examples/accuracy_workflow.py) demonstrates an
analytic anchor, convergence check, S-parameter physicality report, and runtime
capture without claiming foundry qualification.

## Package layout

| Subpackage | Purpose |
|---|---|
| `photonix.core` | Backend, units, constants, the `SDict` type system, S-param utils |
| `photonix.em` | All EM physics: scalar/semivector/full-vector FDE, slab, EIM, FDFD, EME, bend loss, cross-section geometry, dispersive materials, plus an optional MEEP/MPB **FDTD backend** (`photonix.em.meep`) |
| `photonix.components` | Differentiable, parametric component models (waveguides, couplers, MZIs, rings, gratings) |
| `photonix.circuit` | Netlist + differentiable S-parameter circuit solver |
| `photonix.modes` | Compatibility facade re-exporting `photonix.em`'s scalar solver, geometry and materials — *deprecated*, prefer `photonix.em` |
| `photonix.layout` | Cells, routing, GDSII/OASIS I/O, netlist extraction, optional external KLayout DRC/LVS runs |
| `photonix.pdk` | PDK-agnostic interface + an open example PDK |
| `photonix.multiphysics` | Import-safe Elmer, DEVSIM, Lumerical/DEVICE, and external-solver adapters |
| `photonix.viz` | Plot spectra, mode profiles, layouts, circuit graphs |
| `photonix.optim` | Objectives, adjoint helpers, optimizers for inverse design |
| `photonix.diagnostics` | Reproducible backend, precision, device, dependency, and external-tool manifests |

**Port naming.** Optical ports are `o1, o2, ... oN` everywhere. The legacy
semantic names (`in0`/`out0`, `i1`/`t1`/`d2`) still work as lookup aliases; see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#port-naming).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design, and
[`BUILD_SPEC.md`](BUILD_SPEC.md) for the developer contract. The explicit
production-PIC boundary and remaining sign-off gaps are in
[`docs/PIC_COMPLETENESS.md`](docs/PIC_COMPLETENESS.md); executable open-source
and licensed-tool handoffs are collected in
[`docs/INTEGRATIONS.md`](docs/INTEGRATIONS.md), and the numerical evidence policy
is in [`docs/VALIDATION.md`](docs/VALIDATION.md).

## License

MIT — see [`LICENSE`](LICENSE).
