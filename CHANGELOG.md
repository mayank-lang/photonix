# Changelog

All notable changes to photonix are recorded here. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html). While the
major version is 0, a minor release may change public behaviour; each such change
is listed under **Changed**.

Each release section becomes the body of its GitHub Release (see
[`RELEASING.md`](RELEASING.md)). A section's heading carries its date only once
the release is tagged.

## [Unreleased]

### Added

- Hierarchical netlists, multimode circuits and netlist interop
  ([docs](docs/HIERARCHY_AND_INTEROP.md)): `circuit.Subcircuit`,
  `circuit.HierarchicalNetlist` (version-2 JSON, safe `"="` parameter
  expressions, cycle checks), `circuit.compile_hierarchical` (recursive block
  elimination or flattening, per-call caching of identical blocks) and
  `layout.extract_hierarchical_netlist`; modal terminals (`"o1:TE0"`) with
  `circuit.lift_model`, `circuit.ModalNetlist`, `circuit.eme_to_modal` and
  `circuit.meep_dataset_to_modal`; polarization and multimode models
  (`polarization_rotator`, `polarization_splitter_rotator`,
  `polarization_beam_splitter`, `grating_coupler_2d`, `jones_fiber`,
  `multimode_straight`, `asymmetric_mode_converter`, `mode_mux`) and
  `analysis.polarization` (PDL, Mueller/Stokes, DGD, principal states); SAX
  and gdsfactory netlist dictionaries (`circuit.netlist_io`), deterministic
  cached PCells (`layout.pcell`) and schematic-driven layout with pic-YAML I/O
  (`layout.SchematicSpec`, `layout.build_layout`,
  `layout.schematic_from_layout`); example `examples/polarization_diversity.py`.

### Fixed

- `photonix.xsection`: `photonix.materials` objects (physics sign `n + i k`) are
  converted to the engineering sign, so lossy materials are no longer treated
  as gain; JAX gradients of rasterized areas, tent moments and `n_eff` are exact
  for horizontal edges, including faces lying on grid lines; box-kernel tensor
  smoothing takes its normals over a continuous 3 x 3-cell kernel, so `n_eff`
  no longer jumps when a face crosses a grid line; `graded_grid` refine-box
  edges are now exactly grid lines.

## [0.2.0] - Unreleased

The first tagged release. It adds the layout-to-simulation fidelity work, a much
larger component library, circuit architecture builders, the `analysis` and
`synthesis` packages, PDK cross-sections and layer stacks, a task-oriented
command-line interface, and release, CI and documentation infrastructure.

### Added

- **Geometry-faithful layout extraction.** Cells record `Cell.model` and
  `Cell.model_settings`, and `extract_netlist` copies them into the netlist, so a
  drawn 25 um straight simulates as 25 um rather than as the model default.
  Extraction is hierarchical. Top-level ports are named, routed cells carry a
  model, and connections between ports that are not antiparallel or whose
  widths differ raise a warning.
- **Layout.** Port-mating placement (`Reference.connect`), optical and
  electrical port types, Euler and cosine S-bends, and bend-aware
  `route_manhattan` (straight, L, Z and U routes, or explicit waypoints). New
  generators: single and double rings, S-bend coupler, `mmi2x2`, crossing,
  Bragg grating, edge coupler, heater, p-n modulator, pad and a hierarchical
  MZI. A DRC pre-check (`run_drc_precheck`, `Pdk.drc_precheck`) checks minimum
  width, space and area through merged-polygon morphology.
- **Component library expansion (30+ models).** A gap-calibrated evanescent
  coupler (supermode splitting fitted to the package's own full-vector FDE,
  residual at most 1.3 %), an NxN MMI with Bachmann self-imaging phases, a
  crossing with an exact passivity bound, coupled-mode-theory transfer-matrix
  Bragg gratings (uniform, apodized and phase-shifted), an exactly passive
  phased-array AWG with seeded phase errors, heater, thermo-optic and
  carrier-depletion (Soref-Bennett) phase shifters, a photodetector, a Gaussian
  edge coupler, an Euler bend, a taper, a routed waveguide and geometry-driven
  rings. See `docs/COMPONENT_LIBRARY.md` for each model's evidence level.
- **Circuit builders.** `lattice_filter`, `crow`, `sagnac_mirror`, Clements
  decomposition derived for the library's physical MZI (`clements_decompose`,
  `mesh_matrix`, `mzi_mesh`) with splitter and phase-shifter imperfections, and
  netlist JSON I/O (`Netlist.to_json` / `from_json`).
- **`photonix.analysis`** (new). Resonance Q, extinction ratio, FSR and finesse
  with Lorentzian fits and the Qi/Qc split; pass-band and crosstalk metrics;
  link budgets; spec-based Monte Carlo yield with Wilson intervals; impulse and
  pulse response. Also sensitivities (JAX autodiff with a Richardson
  finite-difference fallback), first-order second-moment propagation, corner
  designs (full-factorial, Plackett-Burman, axial and gradient-sign),
  Antreich-Graeb worst-case distance, and importance-sampled yield.
- **`photonix.synthesis`** (new). Butterworth and Chebyshev prototypes, exact
  coupled-resonator (CROW) synthesis checked against the netlist simulation,
  exact FIR lattice (cascaded-MZI) synthesis by polynomial peeling with
  maximally flat half-band (interleaver) targets, and `coupling_to_gap`, which
  inverts the evanescent coupler.
- **Circuit engine.** `evaluate_circuit` and `circuit_from_netlist` compile the
  topology once and choose a dense, sparse-LU (SuperLU with an analytic
  adjoint, reverse-mode differentiable) or subnetwork-reduction backend;
  `compile_circuit` also deduplicates identical instances. 32x32 Clements
  meshes are practical. See `docs/CIRCUIT_ENGINE.md`.
- **`photonix.materials`** (new). Cited, range-checked complex-index models
  (temperature-dependent and uniaxial where published) with thermal,
  electrical and electro-optic data; refractiveindex.info YAML and CSV import;
  free-carrier silicon; Lorentz-Lorenz mixing; passive Lorentz-Drude fits with
  Meep export; Kramers-Kronig checks. See `docs/MATERIALS.md`.
- **`photonix.macromodel`** (new). Delay extraction and complex-baseband vector
  fitting of sampled S-parameters to stable rational models, Hamiltonian
  passivity certification and enforcement, generalized Kramers-Kronig causality
  tests, state-space export and exact time-domain simulation. See
  `docs/MACROMODEL.md`.
- **`photonix.xsection`** (new). Polygonal multi-material cross-sections rasterized
  exactly into tensor-smoothed permittivity and thermal/electrical
  coefficients, a full-vector mode solver with a JAX-differentiable `n_eff`, and
  Richardson convergence studies. See `docs/xsection.md`.
- **`photonix.timedomain`** (new). Sample-mode circuit simulation with feedback
  and time-varying elements; PRBS, lasers (linewidth, RIN, rate equations),
  MZM/EAM/ring modulators, photodiodes and TIAs with noise; eye diagrams,
  Q/BER, receiver sensitivity, power penalties and mask tests. See
  `docs/TIMEDOMAIN.md`.
- **Routing.** `route_sbend`, `route_auto`, `route_bundle`, `fanout`,
  obstacle-avoiding `route_astar` and `route_length_matched`, with clearance
  verified on the drawn geometry. See `docs/ROUTING.md`.
- **Layout I/O and die assembly.** Lossless GDSII/OASIS round trips
  (`write_layout` / `read_layout`: hierarchy, arrays, ports, JSON sidecar),
  polygon text and labels, packing and alignment, fibre-array grating rows,
  pad arrays, die frames and floorplan checks. See
  `docs/LAYOUT_IO_AND_ASSEMBLY.md`.
- **PDK.** `LayerStack` (mode-solver cross-sections and Meep layer specs), a
  `CrossSection` whose waveguide parameters `Pdk.models()` / `Pdk.simulate()`
  bind into every model that declares them, design rules, `Pdk.route` and
  `Pdk.drc_precheck`.
- **Command-line interface** (`photonix.cli`). Adds `models`, `sweep`,
  `sparams check|convert`, `mode`, `gds inspect|precheck`,
  `netlist simulate` and `validate` alongside `info`. Every command accepts
  `--json` (with a versioned schema) and returns documented exit codes. See
  `docs/CLI.md`.
- **Release engineering.** The version is single-sourced from
  `photonix.__version__`. This adds this changelog, `CITATION.cff`,
  `.zenodo.json`, a tag-driven release workflow (build, `twine check --strict`,
  wheel-content checks, a clean-venv smoke test, and PyPI Trusted Publishing),
  `RELEASING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue and pull-request
  templates, a pre-commit configuration and Dependabot.
- **CI.** Python 3.13 and macOS/Windows jobs, a NumPy-backend job, a
  minimal-install job (NumPy, SciPy and pytest only), an oldest-dependencies
  job, a coverage gate, a nightly Meep job, a docs build, a weekly scheduled
  run, and cancellation of superseded runs.
- **Documentation site.** A Sphinx configuration (MyST Markdown) whose API
  reference is generated from every public module's `__all__` when the site is
  built, documenting each object once at its defining module and linking
  re-exports there; a test checks that every public name is covered. Also an
  examples page with the committed figures, a command-line reference, and a
  conventions page covering time and propagation sign, complex index,
  S-parameter normalization, port naming and units. The build is warning-free
  under `sphinx -W`.

### Changed

- `circuit.crow` takes a `wl0` argument (default 1.55 um, backward compatible),
  so designs centred away from 1.55 um stay resonant.
- Project URLs point to <https://github.com/mayank-lang/photonix>. The package
  version lives in `photonix.__version__` only, and `pyproject.toml` reads it
  at build time.
- License metadata uses the PEP 639 SPDX form (`license = "MIT"`), which
  requires `hatchling >= 1.27`.
- `tests/conftest.py` no longer imports matplotlib unconditionally, so the suite
  collects on a minimal NumPy/SciPy installation.

### Fixed

Findings of the round-2 audit:

- **Components.** The raised-cosine apodization of `bragg_grating` was
  inverted, giving a notch, and its gradients were NaN at gamma = 0. The output
  index of `mmi_nxn` ran from the wrong side, so odd-N MMIs did not self-image
  under any port gauge. `awg_design` rounded the grating order up, letting the
  FSR drop below the channel plan, and accepted fewer arms than channels.
  `straight` and every model built on it, plus `edge_coupler`, are now
  differentiable and jittable in `neff`, `ng`, `wl0` and `n_chip`.
  `evanescent_coupler` warns outside its calibrated range. `crossing` is
  jittable with a traced `reflection_db`. The p-n models use the
  temperature-dependent intrinsic density.
- **Layout and PDK.** Features drawn exactly at a DRC rule value now pass.
  Corner-to-corner spacing violations are caught. Racetrack rings no longer
  self-intersect at the closing seam. Sharp-cornered routes keep their full
  width. `LayerStack.cross_section` integrates n^2 exactly and rejects windows
  that no background level covers. Flattened composites and GDS read-backs are
  marked geometry-only and refuse to simulate. `write_gds` keeps distinct cells
  that share a name apart (`name$1`, ...). The demo PDK's `mmi1x2` layout and
  its EME model describe the same device.
- **Circuit and analysis.** Time-domain responses resolve the bulk delay before
  unwrapping. Pulse delay and broadening use circular moments.
  `passband_metrics.ripple_db` measures in-band ripple. `find_resonances`
  handles exact zeros and keeps doublets apart. `yield_analysis` rejects
  duplicate case names. `mzi_mesh` applies phase loss to the output screen.
  `MeshProgram` is immutable and hashable. Netlist JSON is strict (no NaN or
  Infinity). `LinkBudget` averages power, not dB.

## [0.1.0] - 2026-08-03

Initial development version. It was never tagged or published; this section
reconstructs the state of the repository at commit `e218c5a`.

### Added

- **Core.** A JAX backend with a NumPy fallback (`PHOTONIX_BACKEND=numpy`), 64-bit
  precision by default, and the `SDict` / `SDense` / `SCoo` scattering types with
  conversions in the standard `S[out, in]` convention.
- **Circuit.** A differentiable named-port S-matrix solver that handles
  feedback networks, a `Netlist` with validation of reused, self-connected and
  multiply-exposed terminals, and `circuit_from_netlist` defaulting to the
  component registry.
- **Electromagnetics.** Scalar, semivectorial and full-vector (Yee) FDE mode
  solvers with differentiable `n_eff` adjoints; a polarization-resolved slab
  solver; stretched-coordinate PML and conformal-map bend loss; vectorial EME
  with a transverse absorber; TE and TM 2-D FDFD; an effective-index method;
  subpixel-averaged cross-section geometry; Sellmeier materials; and an
  optional Meep/MPB FDTD backend whose specifications import without Meep.
- **Accuracy tooling.** Adaptive Richardson / grid-convergence-index studies
  (`estimate_convergence`, `adaptive_convergence`); high-order group delay and
  GDD on nonuniform sweeps; quantitative passivity, reciprocity and losslessness
  diagnostics (`analyze_sparameters`); and pointwise passive projection.
- **Data exchange.** `SParameterDataset` with pickle-free NPZ, Touchstone 1.0
  `S RI` I/O and an optional scikit-rf bridge; GDSII/OASIS through gdstk; and
  external KLayout DRC/LVS execution with user-supplied decks.
- **Design and process.** Inverse-design objectives and fabrication filters,
  process corners and correlated Monte Carlo specifications, and import-safe
  Elmer, DEVSIM, Lumerical and generic external-solver adapters with unit-aware
  field exchange.
- **Diagnostics.** `photonix info [--json]` and `runtime_info()`, which report
  the backend, precision, devices and optional dependencies for a reproducibility
  manifest.

### Changed

- Optical ports are named `o1 ... oN` everywhere. The legacy names (`in0`/`out0`,
  `i1`/`t1`/`d2`) remain as read-only lookup aliases (`AliasedSDict`).
- `photonix.modes` is a compatibility facade over `photonix.em`, so there is a
  single scalar mode solver.

### Fixed

- EME no longer creates energy from evanescent modes, and it is deterministic
  (seeded ARPACK start vector). `em.components.mmi1x2` converges under grid
  refinement.
- `add_drop_ring` is a true four-port. Doctests are executed in CI.

[Unreleased]: https://github.com/mayank-lang/photonix/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mayank-lang/photonix/releases/tag/v0.2.0
[0.1.0]: https://github.com/mayank-lang/photonix/tree/e218c5a
