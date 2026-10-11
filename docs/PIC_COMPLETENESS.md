# PIC completeness and sign-off boundary

Photonix is an integrated research design stack, not a foundry sign-off system.
The package can construct layouts, extract/connect circuits, solve several levels
of electromagnetic models, optimize densities, write GDSII/OASIS, and hand exact native
polygons to Meep. It does not replace a foundry PDK, a commercial DRC/LVS deck, or
calibrated multiphysics.

## Implemented workflow

1. Build hierarchical `Cell` geometry with named, oriented ports.
2. Route and flatten geometry; read/write GDSII or OASIS when `gdstk` is installed
   (losslessly, hierarchy and ports included, with `read_layout`/`write_layout`).
3. Extract a netlist and solve interconnected component S-parameters.
4. Use analytic compact models, EME, FDFD, scalar/semivector/full-vector FDE,
   or the optional Meep/MPB backend at the appropriate fidelity.
5. Optimize continuous or filtered/projected density fields with NumPy/JAX.
6. Convert a native `Cell` plus a process-layer map to exact 2-D or 3-D Meep
   `Prism` geometry. Convert native ports to consistently oriented mode planes.
7. Extract a complete multimode/multiport Meep S matrix from native layout
   ports, optionally against a reference layout, at one or more wavelengths.
8. Store sampled S matrices in a validated, versioned `SParameterDataset`,
   interpolate within the sampled band (or fit a rational model with
   `to_rational()`, which stays exact between samples), and round-trip
   pickle-free NPZ or standards-compatible Touchstone 1.0 single-ended `S RI`
   files.
9. Represent absolute process corners and reproducible correlated-Gaussian Monte
   Carlo samples without embedding undocumented foundry distributions.
10. Probe and run prepared Elmer, DEVSIM, Ansys Lumerical/DEVICE, or other
    site/licensed multiphysics jobs through validated, non-shell subprocess
    contracts. Capture logs and require declared output files without inventing
    physical observables.
11. Hand a GDSII/OASIS stream to an external KLayout executable and run a
    user-supplied DRC or LVS deck in headless batch mode.
12. Place references by mating ports (`Reference.connect`), route with
    circular or Euler bends (`route_manhattan`, `route_auto`, S-bends,
    non-crossing bundles and fan-outs, obstacle-avoiding A*, length-matched
    meanders; see [`ROUTING.md`](ROUTING.md)), and extract hierarchical
    layouts whose cells carry their geometry-derived model settings, so the
    simulated circuit matches the drawn lengths, radii and gaps.
13. Pre-check minimum width, spacing and area with merged-polygon morphology
    before an external DRC run.
14. Describe the process with a `LayerStack` (mode-solver cross-sections and
    Meep layer specs) and `CrossSection`s whose waveguide parameters
    `Pdk.simulate` applies to extracted layouts.
15. Characterize responses (resonance Q/ER/FSR, pass bands, crosstalk), build
    link budgets, and estimate spec-based parametric yield over process
    corners and Monte Carlo samples with confidence intervals.
16. Program arbitrary unitaries onto MZI meshes and simulate them at component
    level with splitter and phase-shifter imperfections.
17. Rank parameter sensitivities (AD or finite differences), propagate process
    covariances to first and second order, generate factorial or
    Plackett-Burman corners, find the worst-case distance to a spec, and
    estimate rare-failure yield by importance sampling.
18. Synthesize Butterworth/Chebyshev coupled-resonator filters exactly and map
    the designed couplings to physical coupler gaps.
19. Extract compact models from sampled S-parameters (field solves,
    measurements, Touchstone): delay extraction plus complex-baseband vector
    fitting to a stable rational model with a Hamiltonian passivity
    certificate, minimum-energy passivity enforcement, a generalized
    Kramers-Kronig causality test of the raw data, state-space export and
    exact time-domain simulation (`photonix.macromodel`, see
    [`MACROMODEL.md`](MACROMODEL.md)).
20. Round-trip whole layouts through GDSII/OASIS without loss (`write_layout`
    / `read_layout`): units, hierarchy, references, arrays, instance names,
    labels, and ports (SiEPIC-style PinRec pins, labels on edges, or a JSON
    sidecar that also carries `model`/`model_settings` and detects edited
    structures), with byte-reproducible output. Foreign files (foundry
    black boxes, KLayout/gdsfactory/SiEPIC output) import with their
    hierarchy and pins. See [`LAYOUT_IO_AND_ASSEMBLY.md`](LAYOUT_IO_AND_ASSEMBLY.md).
21. Assemble dies: polygon text with a minimum-stroke guarantee, GDS text
    labels, grid / MaxRects packing / align / distribute, 127 µm fibre-array
    grating rows with an alignment loopback, pad arrays, die frames with
    dicing lane, keep-out and facet markers, SiEPIC-style `opt_in` test
    labels, and floorplan checks (outside-die, keep-out, overlaps,
    edge-coupler facet alignment).
    the designed couplings to physical coupler gaps; synthesize FIR lattice
    (cascaded-MZI) filters exactly from any power-complementary target, with
    maximally flat interleavers and equiripple band splitters built in.
22. Solve large circuits (thousands of terminals, e.g. 32x32 MZI meshes) with
    a sparse-LU engine whose analytic adjoint keeps them differentiable, or by
    subnetwork reduction; compile once and reuse the topology
    ([`CIRCUIT_ENGINE.md`](CIRCUIT_ENGINE.md)).
23. Draw material properties from a cited, range-checked database (n-k,
    temperature, anisotropy, thermal/electrical data) and import
    refractiveindex.info files ([`MATERIALS.md`](MATERIALS.md)).
24. Build polygonal multi-material cross-sections (sidewall angles, slabs,
    gaps, heaters) that rasterize exactly into tensor-smoothed permittivity and
    thermal/electrical coefficients, with differentiable full-vector `n_eff`
    and Richardson convergence studies ([`xsection.md`](xsection.md)).
25. Simulate links in the time domain: sample-mode circuits with feedback and
    modulators, PRBS, lasers with linewidth/RIN, MZM/EAM/ring modulators,
    photodiodes and TIAs with noise, eye diagrams, Q/BER, sensitivity and
    mask tests ([`TIMEDOMAIN.md`](TIMEDOMAIN.md)).

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

## Compact-model caveats

[`COMPONENT_LIBRARY.md`](COMPONENT_LIBRARY.md) lists each model's evidence and
limits. The ones most likely to matter in a design decision:

- `evanescent_coupler`'s default supermode splitting is a fit to Photonix
  full-vector solves of one cross-section (500 x 220 nm Si strips in SiO2,
  gaps 0.10-0.40 µm, 1.50-1.60 µm). It is a solver-derived default, not
  measurement; pass a calibrated `splitting=` callable for a real process.
- `awg` uses paraxial star couplers and matched output modes; its crosstalk
  floor is set by array truncation and the optional seeded phase errors, not by
  a full propagation model.
- `pn_phase_shifter` uses abrupt-junction depletion, the Soref-Bennett 1.55 µm
  fits and a uniform-lateral-field overlap; forward-bias injection, RC and
  velocity limits are absent.
- `grating_coupler` is phenomenological (Gaussian spectrum).
- The DRC pre-check covers width, spacing and area only; it is not a rule deck.

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
  corners, and design-rule decks (the die-frame, keep-out, trench and text
  layers of `layout.library.die_frame` are placeholders to map to a foundry
  layer map);
- foundry-qualified DRC/LVS decks, deck-specific clean-report interpretation,
  density/fill sign-off, waivers, and tape-out release governance;
- versioned mask/process metadata;
- measurement-calibrated compact models: `photonix.macromodel` fits, certifies
  and time-steps any sampled S-matrix, but the data's calibration (reference
  planes, de-embedding, noise floor, repeatability) and a parametric model
  across geometry or process remain the user's; the coupler splitting fit shows
  the intended pattern for one cross-section;
- calibrated thermo-optic, carrier, electrical, stress, and fabrication-yield
  decks, coupling laws, output parsers, and solver-backed validation;
- foundry-supplied corner distributions, Monte Carlo yield acceptance criteria,
  and regression baselines (the package now provides only neutral study records);
- calibrated transmitter and receiver parameter sets (laser rate-equation
  parameters, modulator `V_pi`/bandwidth, receiver noise) and standard-compliant
  TDECQ / eye-mask procedures: `photonix.timedomain` provides the physics with
  analytic and cross-solver validation only;
- a Meep-enabled CI runner. Normal CI validates every pure adapter, but real
  MPB/FDTD tests are skipped where the conda-only dependency is absent.

For tape-out, use Photonix as a design/simulation layer and pass the result through
the target foundry's qualified verification and sign-off flow.
