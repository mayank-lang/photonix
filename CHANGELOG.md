# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **MEEP/MPB FDTD backend** (`photonix.em.meep`): photonix now delegates all FDTD
  needs to [MIT MEEP](https://meep.readthedocs.io) instead of shipping its own
  time-domain solver.
  - `waveguide_sparams` — 2-D FDTD S-parameters as a native `SDict`, mirroring the
    `photonix.em.fdfd.waveguide_sparams` signature (PML added outside the supplied
    grid; edge cross-sections replicated through the absorber).
  - `solve_modes` / `n_eff` — MPB cross-section modes returned as a native
    `VectorModeData`, a drop-in cross-check for the in-house full-vector FDE.
  - `to_material_grid` / `material_grid_weights` / `DeviceGrid` — permittivity-grid
    → MEEP translation with a pure, tested core and a documented unit bridge.
  - Optional, fail-loud import contract: `import photonix.em` works without MEEP;
    touching `photonix.em.meep` raises `ImportError` with a conda install hint.
- Benchmark adapter `benchmarks/external/meep_adapter.py` now produces real MPB/MEEP
  reference numbers for `python benchmarks/run.py --external`.
- `tests/test_em_meep.py` (15 tests: 2 always-on import-contract + 13 MEEP-gated).
- `CONTRIBUTING.md`, `CHANGELOG.md`, and `MEEP_BACKEND_RUN.md`.

### Changed
- `photonix.em` exposes the `meep` backend lazily (via module `__getattr__`) so the
  core package never imports MEEP eagerly.
- `docs/DESIGN_EM_SOLVERS.md` extended with §17 documenting the MEEP backend.

## [0.1.0]

### Added
- Initial beta: differentiable core (`SDict` types, JAX/NumPy backend), component
  models, circuit S-parameter solver, mode/FDE/FDFD/EME solvers, layout/GDS,
  example PDK, visualization, inverse-design optimizers, and an internal benchmark
  suite.
