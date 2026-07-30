# PIC completeness and sign-off boundary

Photonix is an integrated research design stack, not a foundry sign-off system.
The package can construct layouts, extract/connect circuits, solve several levels
of electromagnetic models, optimize densities, write GDSII/OASIS, and hand exact native
polygons to Meep. It does not replace a foundry PDK, a commercial DRC/LVS deck, or
calibrated multiphysics.

## Implemented workflow

1. Build hierarchical `Cell` geometry with named, oriented ports.
2. Route and flatten geometry; read/write GDSII or OASIS when `gdstk` is installed.
3. Extract a netlist and solve interconnected component S-parameters.
4. Use analytic compact models, EME, FDFD, scalar/semivector/full-vector FDE,
   or the optional Meep/MPB backend at the appropriate fidelity.
5. Optimize continuous or filtered/projected density fields with NumPy/JAX.
6. Convert a native `Cell` plus a process-layer map to exact 2-D or 3-D Meep
   `Prism` geometry. Convert native ports to consistently oriented mode planes.
7. Extract a complete multimode/multiport Meep S matrix from native layout
   ports, optionally against a reference layout, at one or more wavelengths.
8. Store sampled S matrices in a validated, versioned `SParameterDataset`,
   interpolate within the sampled band, and round-trip pickle-free NPZ or
   standards-compatible Touchstone 1.0 single-ended `S RI` files.
9. Represent absolute process corners and reproducible correlated-Gaussian Monte
   Carlo samples without embedding undocumented foundry distributions.
10. Probe and run prepared Elmer, DEVSIM, Ansys Lumerical/DEVICE, or other
    site/licensed multiphysics jobs through validated, non-shell subprocess
    contracts. Capture logs and require declared output files without inventing
    physical observables.
11. Hand a GDSII/OASIS stream to an external KLayout executable and run a
    user-supplied DRC or LVS deck in headless batch mode.

## Interchange and external-solver handoff

`SParameterDataset.save_touchstone("device.s2p")` and `load_touchstone(...)`
provide dependency-free interchange. The internal implementation deliberately
targets the widely interoperable Touchstone 1.0 `S RI` subset: `.sNp` supplies
the port count, two-port files use `S11, S21, S12, S22`, larger files are
row-major, frequencies are increasing, and the default reference is 50 ohms.
Photonix port names and JSON metadata are stored in standards-safe comments.
See the [IBIS Touchstone 2.1 specification](https://www.ibis.org/touchstone_ver2.1/touchstone_ver2_1.pdf)
for the governing syntax and ordering rules.

`touchstone_capabilities()` reports the internal subset and whether optional
scikit-rf is installed. `SParameterDataset.to_skrf()` / `from_skrf()` bridge to
`skrf.Network`; install that optional path with `pip install 'photonix[rf]'`.
The core reader never silently delegates to scikit-rf, so minimal installations
remain deterministic and unsupported MA/DB, mixed-mode, noise, or Touchstone
2.x input fails with a local diagnostic.

Commercial adapters should exchange explicit artifacts rather than importing
proprietary SDKs into Photonix core: sampled models through Touchstone, masks
through `layout.write_oas`/`write_gds`, and circuit topology through a Photonix
`Netlist` converted by the adapter to its vendor-supported schema. The generic
`photonix.interop.ExternalSolverHandoff` records the solver, interface, artifact
format, required executable/module, license requirement, and JSON metadata. It
does not contain vendor commands, process decks, SDK code, credentials, or
license material; those remain with the licensed external installation.

## External verification contract

`klayout_available()` checks for a runnable KLayout executable without making
KLayout a Python dependency. `run_drc`, `run_lvs`, and the lower-level
`run_klayout_deck` invoke KLayout with its documented `-b`, `-rd`, and `-r`
arguments. The conventional runtime variables are `input` and `report`; their
names can be changed or disabled, and additional deck-specific string variables
can be supplied explicitly.

The adapter treats decks as opaque, user-owned files: it validates the path and
passes it to KLayout but never reads, copies, modifies, or redistributes its
contents. This is essential for licensed foundry rule decks. A zero exit code
only establishes successful execution. Photonix does not infer a clean result
from a proprietary report schema and does not turn an open example PDK into a
foundry-qualified sign-off flow.

## Meep integration contract

The pure objects (`LayerSpec`, `PreparedLayout`, `PreparedPort`, `DeviceGrid`)
are importable without Meep. Object realization and solvers fail locally with an
installation hint if Meep/MPB is unavailable.

Two grid meanings are kept separate:

- a Photonix epsilon raster is cell-centred and is realized with a
  piecewise-constant material callback;
- an inverse-design density grid is node-interpolated and is realized with
  Meep `MaterialGrid`.

`prepare_layout` flattens hierarchy, recentres polygons and ports, validates the
layer stack, and reserves 2-D/3-D cell margins. `build_layout_geometry` maps every
selected polygon to a Meep prism. `port_region(s)` supplies monitor planes plus
inward/outward wave vectors for `NO_DIRECTION` eigenmode sources and consistent
mode decomposition. Meep mode regions are axis-aligned, so diagonal native ports
are rejected explicitly rather than silently converted to the wrong plane.
`build_layout_simulation` creates the simulation shell.

The grid-based `waveguide_sparams` helper covers one spatial mode per port and
two collinear ports. It performs one run for each incident side; S12 and S22 are
measured, never inferred from reciprocity. `normalization_eps` enables a separate
reference run and subtracts its residual backward launch coefficient.

For native layouts, `PortModeSpec`, `plan_multiport`, and
`simulate_multiport_sparameters` expand every selected physical port into one or
more MPB-band terminals and measure every incident column of the full matrix.
The decomposition wave-vector follows each port's outward normal, so direction 0
is consistently outgoing. A reference layout supplies the incident denominator
and the backward launch-background vector at the incident physical port. Separate
narrow-band runs keep each requested mode locked to its wavelength.

## Numerical caveats

- In-house FDE/FDFD/EME discretizations remain approximations. Convergence in
  resolution, domain size, modal basis size, absorber/PML thickness, and monitor
  position is part of a valid result.
- The full-vector FDE uses scalar arithmetic subpixel epsilon rather than fully
  staggered tensor averaging. Its bend model uses an equivalent-index conformal
  map, now guarded against crossing its coordinate singularity, but is not a
  transformed-anisotropic 3-D bend solve.
- EME's optional transverse absorber is a graded lossy layer, not a true
  stretched-coordinate PML. Closed-window radiation modes are box modes.
- A Photonix wavelength-to-index `Material` can only be frozen to a
  non-dispersive Meep medium at one wavelength. Broadband causal material models
  require explicit Meep Lorentz/Drude susceptibilities.
- Meep eigenmode launch/decomposition requires MPB and is limited by MPB's modal
  material assumptions. Reference planes change S-parameter phase.
- Density filtering/projection encourages a length scale and binarity; it does
  not prove minimum solid and void dimensions.

## Process and multiphysics adapter boundary

`ProcessCorner`, `MonteCarloSpec`, and `ProcessStudy` carry parameter values,
mandatory units, covariance, names, and reproducible seeds. A study can map a
caller callback over nominal/corner/Monte Carlo cases while preserving names,
and can be registered on a `Pdk`. It does not supply a foundry distribution;
production values must come from the target PDK.

`photonix.multiphysics` is orchestration rather than a new PDE solver. Elmer
jobs point to an existing `.sif` deck. DEVSIM jobs point to an existing Python
script. Lumerical/DEVICE jobs point to a prepared Python script using the
user-installed `lumapi`; the adapter only tests module/file discoverability and
optional license-environment configuration, never imports `lumapi` or opens a
licensed session during probing. This matches Ansys's documented external
[Python API boundary](https://optics.ansys.com/hc/en-us/articles/360037824513-Python-API-overview)
and [installation model](https://optics.ansys.com/hc/en-us/articles/39744901602707-Installation-and-Getting-Started-Python-API).
`ExternalSolverAdapter` builds argv arrays for licensed/site tools and
can check that a license environment variable is configured without contacting
the license server. Capability detection launches no process. `run_job` uses
`shell=False`, captures stdout/stderr/return code/timing, enforces timeouts, and
checks declared output-file presence. Solver-specific mesh creation, boundary
conditions, material calibration, convergence, and output parsing remain the
responsibility of the prepared deck and its owning workflow.

`FieldDataset` is the neutral field handoff: finite mesh points, an explicit
coordinate unit, named scalar point fields with mandatory units, optional cell
connectivity, JSON metadata, and versioned pickle-free NPZ round-trip. Optional
`meshio` conversion is imported only at the call boundary and requires callers
to state coordinate/field units because VTK-family files do not provide one
portable unit convention. Interpolation onto optical-grid points uses SciPy
`griddata`, removes source-constant coordinate axes (such as planar VTU z), and
requires an explicit raise/fill/nearest out-of-domain policy. The scalar
`LinearIndexModel` contains no built-in silicon or carrier coefficients: every
term states source-field/coefficient units and the model requires calibration
provenance before producing `n` and relative permittivity `n**2`.

## Still required for a production PIC flow

The following are deliberately reported as open rather than hidden behind a
"complete" label:

- foundry-qualified layer stacks, cross-sections, component libraries, model
  corners, and design-rule decks;
- foundry-qualified DRC/LVS decks, deck-specific clean-report interpretation,
  density/fill sign-off, waivers, and tape-out release governance;
- versioned mask/process metadata;
- calibrated interpolation policy and automated compact-model extraction from
  field simulations or measurements beyond the implemented Touchstone exchange;
- calibrated thermo-optic, carrier, electrical, stress, and fabrication-yield
  decks, coupling laws, output parsers, and solver-backed validation;
- foundry-supplied corner distributions, Monte Carlo yield acceptance criteria,
  and regression baselines (the package now provides only neutral study records);
- a Meep-enabled CI runner. Normal CI validates every pure adapter, but real
  MPB/FDTD tests are skipped where the conda-only dependency is absent.

For tape-out, use Photonix as a design/simulation layer and pass the result through
the target foundry's qualified verification and sign-off flow.
