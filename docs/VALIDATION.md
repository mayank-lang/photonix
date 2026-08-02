# Numerical validation and result qualification

Photonix separates a calculation from the evidence supporting it. A solver can
implement the intended equations correctly and still produce a poor result when
the mesh, boundary, modal basis, material data, or port normalization is wrong
for a particular device. This document is the minimum validation workflow for a
result that will inform a PIC design decision.

## Evidence levels

Use the strongest available label; do not describe every passing calculation as
"validated."

| Level | Evidence | What it establishes |
|---|---|---|
| 0 — calculated | One run, input validation passed | The software completed; no accuracy claim |
| 1 — converged | Relevant discretization and domain/basis sweeps stabilize the quantity of interest | Numerical error is bounded for the chosen model |
| 2 — analytic | Agreement with an exact or asymptotic solution in its valid regime | Equations, units, and implementation agree at an anchor |
| 3 — cross-solver | Agreement with an independently formulated solver on the same geometry and materials | Reduced risk of shared implementation error |
| 4 — externally benchmarked | Agreement with a traceable external solver, publication, or open reference dataset | Reproducibility beyond Photonix |
| 5 — measurement calibrated | Agreement with versioned measurements for the target process and extraction flow | Fitness for that calibrated process window |

Foundry qualification and tape-out sign-off remain separate governance steps;
they are not implied by any numerical evidence level.

## Choose the model before choosing the mesh

| Method | Appropriate use | Required qualification | Important boundary |
|---|---|---|---|
| Compact component and circuit models | Fast sweeps, control design, circuit optimization | Check the model's calibration range and circuit S-matrix physicality | Does not infer geometry-level Maxwell physics |
| Scalar FDE | Weak-guidance modes, fast trends, initial estimates | Grid/domain convergence and an analytic slab anchor | Can overestimate confinement for high-index-contrast strips |
| Semivectorial/full-vector FDE | High-contrast cross-section modes and propagation constants | Grid/domain convergence; polarization/mode tracking; material-dispersion check | A 2-D cross-section solve is not a 3-D discontinuity solve |
| FDFD | Frequency-domain fields in compact 2-D devices | Grid, PML, port-position, and domain sweeps; power balance | Each run is single-frequency and inherits the dimensional reduction |
| EME | Tapers, MMIs, width steps, and long piecewise-invariant structures | Modal-basis, section, window, and absorber sweeps | Truncated radiation/evanescent bases can bias interfaces |
| Meep/MPB backend | Independent time-domain or eigenmode calculations in 2-D/3-D | Meep resolution, cell/PML, run-time decay, source, monitor, and reference runs | Availability does not imply a configured or qualified workflow |

See [`DESIGN_EM_SOLVERS.md`](DESIGN_EM_SOLVERS.md) for formulations and
[`PIC_COMPLETENESS.md`](PIC_COMPLETENESS.md) for explicit production-flow gaps.

## Minimal convergence study

Converge the quantity actually used by the design decision—not merely a field
plot. For a strip-waveguide effective index, for example:

```python
import photonix.em as em

study = em.adaptive_convergence(
    lambda resolution: em.n_eff_fullvector(
        wl=1.55,
        width=0.50,
        thickness=0.22,
        resolution=resolution,
        margin=1.5,
    ),
    initial_resolution=20,
    refinement=1.5,
    max_levels=5,
    rtol=2e-3,
)
print(study.resolutions, study.values)
print(study.extrapolated, study.grid_convergence_index, study.observed_order)
if not study.converged:
    raise RuntimeError("observable did not reach its requested grid accuracy")
```

`adaptive_convergence` uses at least three levels, Richardson extrapolation, and
a safety-factored Grid Convergence Index. Crucially, it also rejects a small
last-step change when successive corrections are not aligned with a single
asymptotic error term. Use `estimate_convergence` when results already exist or
when you need manual control over the sampled resolutions.

The returned `converged` flag can legitimately be false; that is evidence to
refine or revisit the model, not an exception to hide. Repeat the study with a
larger transverse margin. A small grid change with a
large margin change indicates domain truncation rather than mesh convergence.
For leaky/bent modes, converge the complex propagation constant or loss itself;
convergence of only its real part is insufficient. Set an acceptance tolerance
from the system error budget rather than copying a universal threshold.

For EME, replace the resolution sweep with independent sweeps of transverse
resolution, modal basis size, section count/placement, and window/absorber. For
FDFD or FDTD, independently vary resolution, PML thickness, domain padding,
source/monitor position, and termination/run time.

## S-parameter physicality

For power-normalized modal waves, passivity requires the largest singular value
of each S-matrix to be at most one. Checking each element is not sufficient: a
matrix can have `abs(S_ij) <= 1` everywhere and still create power for a coherent
multiport excitation.

```python
import photonix as px

report = px.analyze_sparameters(s_parameters, passivity_atol=1e-8)
print(report.passive, report.reciprocal, report.lossless)
print(report.worst_passivity_violation, report.worst_reciprocity_error)
```

These tests assume consistent power-wave normalization and modal phase
conventions. Pointwise passivity does not establish broadband causality, and
reciprocity is not expected for every material or biased device. Passive
projection can repair small fitting/interpolation violations, but it must not be
used to conceal a non-converged field solve or a normalization error.

## Reproducibility manifest

Record enough information to rerun the calculation. Photonix provides an
import-light snapshot of the software environment:

```python
import json
import photonix as px

manifest = px.runtime_info().as_dict()
print(json.dumps(manifest, indent=2, sort_keys=True))
```

The runtime snapshot is necessary but not sufficient. Also retain:

- geometry and material model names, values, units, and provenance;
- wavelength/frequency sampling and reference-plane definitions;
- solver method and every grid/domain/PML/modal-basis setting;
- mode ordering, polarization convention, and port normalization;
- convergence sweep values and the acceptance criterion;
- Photonix input data, random seeds, and relevant external-solver versions;
- the exact PDK and rule-deck versions for any process-specific claim.

## Repository evidence

- `tests/test_em_*.py` contains analytic, regression, gradient, and physical
  invariant tests for individual solvers.
- `tests/test_sparameter_quality.py` exercises coherent multiport physicality
  diagnostics and passive projection.
- `benchmarks/cases.py`, `benchmarks/references.json`, and
  `benchmarks/RESULTS.md` contain reproducible scalar benchmark cases and their
  stated tolerances.
- `benchmarks/external/` is the boundary for independent external runs. A stub or
  unavailable adapter is not external validation.

When adding a benchmark, identify whether its reference is analytic, internal,
published, externally computed, or measured. Include enough source and geometry
detail to reproduce it; a familiar-looking number without provenance is not a
traceable reference.

## Numerical-method references

- I. B. Celik et al., [“Procedure for Estimation and Reporting of Uncertainty
  Due to Discretization in CFD Applications”](https://doi.org/10.1115/1.2960953),
  *Journal of Fluids Engineering* 130 (2008). This is the basis for the
  three-grid Richardson/GCI workflow and its 1.25 safety factor; the same
  discretization argument applies to a convergent Maxwell observable.
- B. Fornberg, [“Generation of Finite Difference Formulas on Arbitrarily Spaced
  Grids”](https://doi.org/10.1090/S0025-5718-1988-0935077-0), *Mathematics of
  Computation* 51 (1988). Photonix's local polynomial spectral derivative uses
  the same arbitrary-grid finite-difference principle with local coordinate
  scaling for optical frequencies.
- K. Kurokawa, [“Power Waves and the Scattering
  Matrix”](https://doi.org/10.1109/TMTT.1965.1125964), *IEEE Transactions on
  Microwave Theory and Techniques* 13 (1965). The singular-value passivity test
  is meaningful only for a consistent power-wave normalization.
- D. Deschrijver and T. Dhaene, [“Accurate Passivity Enforcement Algorithm for
  Broadband S-Parameter Macromodels”](https://doi.org/10.1109/AFRCON.2009.5308273),
  IEEE AFRICON (2009). It illustrates why globally passive broadband rational
  models require more than Photonix's deliberately pointwise sampled-matrix
  projection.
