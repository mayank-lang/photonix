# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed (package audit)

- Corrected dense/COO scattering conversion to consistently use the standard
  `S[out, in]` convention, including non-reciprocal devices and duplicate COO
  entries.
- Circuit and layout validation now rejects reused, self-connected,
  multiply-exposed, ambiguous, and unknown terminals before they can create an
  invalid network.
- Geometry resolution is now an exact points-per-micrometre spacing; FDFD,
  fabrication filters, inverse-design schedules, spectrum sweeps, routing,
  plotting, and mode overlaps handle their zero/empty/invalid edge cases.
- Custom mode grids classify guided modes against the highest exterior index,
  avoiding false guidance below a substrate light line.
- Package namespaces import on the minimal NumPy/SciPy installation, the README
  version/install instructions match `pyproject.toml`, and wheel installation
  is covered by package smoke checks.

### Fixed (physics)

Findings from a full numerical audit of the EM stack; see `docs/PHYSICS_AUDIT.md`
for the evidence behind each.

- Corrected the asymmetric one-sided outer stencil in scalar EME/FDFD, removed
  post-solve parity projection, and staggered integer/half-cell PML samples.
- FDFD ports now use numerical longitudinal dispersion and conserved discrete
  flux, reject Nyquist-invalid modes, and independently extract reverse S12.
- Full-vector magnetic fields include the missing `1/n_eff`; bend domains cannot
  cross the conformal singularity; evanescent roots are preserved.
- Slab resolution is exact points per micrometre, custom mode grids are strictly
  validated, masked inverse-design filtering has deterministic exterior density,
  and touching MMI strips are rasterized as a union.
- **EME no longer manufactures energy.** `slab_modes` clipped negative `beta**2`
  to `beta = 0`, so evanescent modes propagated without decay *and* the
  `sqrt(beta)` power normalization divided them by `sqrt(1e-12)`, amplifying them
  by ~1e6. Asking for more modes than a section could propagate produced an
  S-matrix with singular values up to **73 215** (must be <= 1); through the
  public API, `taper(num_modes=20)` returned 2080 % power transmission. `beta` is
  now complex on the physical branch (`Re >= 0`, `Im <= 0`), so evanescent and
  leaky modes attenuate. The propagating sub-block is exactly unitary
  (1.00000000) at every mode count tested.
- **EME is deterministic.** ARPACK seeds itself randomly unless given a start
  vector, so with near-degenerate modes three identical `mmi1x2()` calls returned
  three different answers (spread 6.5e-4). A fixed seeded `v0` is now supplied.
- **`em.components.mmi1x2` converges.** Excess loss used to swing 0.66 -> 3.33 dB
  under a transverse grid refinement, with no plateau in any parameter. Three
  causes: the strip cross-sections were staircased (no subpixel averaging, unlike
  every other profile builder in `photonix.em`), which made the MMI beat length
  wander +-0.8 % and slide a fixed-length device off its self-imaging peak; the
  default modal basis was too small; and the solve was non-deterministic. Now
  1.147/1.147/1.152/1.149 dB across `points` 301..801.
- **MMI supermodes are identified by parity, not index.** For a weakly-coupled
  output pair the even and odd supermodes are degenerate to six digits, so the
  eigensolver's ordering of the two was arbitrary and silently swapped.

### Added

- Import-safe Meep/MPB specifications; exact native `Cell`/layer/port to Meep
  prism/monitor conversion for 2-D and 3-D; distinct cell-centred and density-grid
  realization; full bidirectional two-port and wavelength-array extraction with
  optional reference normalization.
- `SParameterDataset`: a versioned, validated sampled S-matrix contract with
  circuit-model evaluation, interpolation, provenance, pickle-free NPZ I/O,
  Touchstone 1.0 `S RI` interchange, and an optional scikit-rf bridge.
- General native-layout Meep orchestration for complete multimode/multiport
  matrices, with port-local direction conventions and optional reference-layout
  normalization/background subtraction.
- OASIS interchange through optional gdstk and safe external KLayout DRC/LVS
  execution for opaque user/foundry decks.
- Validated process-corner/covariance studies, PDK study registration, Elmer,
  DEVSIM, Lumerical DEVICE, and generic licensed-solver adapters, plus unit-aware
  mesh-field exchange and provenance-required linear index/permittivity response.
- An explicit production-PIC completeness/sign-off document.

- **Transverse absorber for EME** (`eme_smatrix(..., pml=(thickness, strength))`,
  on by default for the EME-backed components). Without it, non-guided basis
  modes are lossless box modes of the window that carry radiated power to the far
  end and re-couple. Implemented as a graded imaginary-permittivity layer rather
  than a stretched-coordinate PML: an SC-PML puts `k0^2 eps s` on the modal
  operator's diagonal, which makes the absorber act like a high-index medium and
  spawns a dense band of absorber modes right where shift-invert is aimed
  (measured: a spurious band at `n_eff ~ 3.49` ahead of the true guided mode at
  3.272). Validated -- guided modes untouched to 1e-16, radiation modes acquire
  monotonically increasing loss, uniform sections stay exactly transparent, and
  reciprocity holds to 1e-15.
- `PHYSICS_AUDIT.md` -- the full numerical audit and resolution status, plus the
  list of what was verified correct (unit conversions, constants, coupler
  unitarity to 2e-16, circuit-vs-analytic ring to 1.8e-15, FDFD energy
  conservation, and all four adjoint gradients against finite differences).

### Changed

- `em.components.mmi1x2` defaults: `num_modes` 12 -> 24 (loss only plateaus above
  20), `length_mmi` 29.5 -> 29.25 (re-optimized at converged settings).
  `em.components.taper`: `num_modes` 6 -> 16.
- `fdfd.waveguide_mode` raises instead of silently discarding an imaginary part
  when handed a non-guided port cross-section.

### Fixed
- **One mode solver, one answer.** `photonix.modes` and `photonix.em.fde` were two
  independent scalar Helmholtz eigensolvers exporting the same names
  (`solve_modes`, `n_eff`, `group_index`) with the same signatures and returning
  *different* numbers — 2.644 vs 2.612 for a 500×220 nm SOI strip at 1.55 µm,
  because only `em` used subpixel permittivity averaging and Richardson
  extrapolation. `photonix.modes` is now a thin facade re-exporting `photonix.em`
  (`photonix.modes.n_eff is photonix.em.n_eff`), so they cannot drift apart again.
- **`add_drop_ring` is now a real four-port.** It previously returned only three
  terminals (`i1`/`t1`/`d2`), leaving the add port undefined and the model
  unusable as a four-port in a circuit. Added the add branch (`o4`), which
  counter-circulates and reuses the same denominator with the couplers exchanged.
- **Layout → circuit round trip works out of the box.** `circuit_from_netlist`
  required a `models` registry positionally, so the flow advertised in
  `docs/ARCHITECTURE.md` raised `TypeError`. It now defaults to
  `photonix.components.MODELS`.
- **Broken subpackages no longer disappear silently.** `photonix/__init__.py`
  wrapped every subpackage import in a bare `except Exception: pass`, so a real
  bug anywhere in the tree turned into a missing attribute much later instead of
  a traceback. Only genuinely optional dependencies (`layout`, `viz`) are now
  tolerated, and the reason is recorded in `photonix.UNAVAILABLE`.
- Five doctests that had never been executed (CI did not run them) now pass:
  four compared NumPy scalars against `True`, and
  `layout.extract_netlist`'s example raised `TypeError` on a dead fallback path.
- `layout.extract_netlist` always returns a `circuit.Netlist`; the
  `except Exception -> plain dict` fallback was unreachable (`photonix.circuit`
  has no optional dependencies) and forced defensive calling code.
- Examples write figures to `examples/outputs/` via a shared `examples/_output.py`
  helper instead of the current working directory, which had been scattering PNGs
  into the repository root.
- Removed a committed editor artefact (`tests/.fuse_hidden…`) and broadened
  `.gitignore` to cover stray figures and editor debris.

### Changed
- **Port naming is now `o1 … oN` everywhere** (BREAKING, with aliases).
  `components.mzi` used `in0`/`out0`/`out1` and `add_drop_ring` used
  `i1`/`t1`/`d2` while every other model used `oN`. The legacy names keep working
  as **read-only lookup aliases** through the new `core.types.AliasedSDict`,
  which stores only canonical keys so `ports_of`, the circuit solver and the
  passivity/reciprocity checks still see one terminal per physical port.
  `circuit.mzi` now follows the 2×2 coupler convention (bar path `o1 → o4`).
- `BUILD_SPEC.md` no longer sanctions a second "semantic" port convention — the
  ambiguity that produced the inconsistency above.
- `photonix.em.materials` is the canonical home for the Sellmeier models (moved
  from `photonix.modes.materials`), keeping the dependency `modes → em`
  one-directional; `photonix.em` re-exports `CrossSection`,
  `rectangular_waveguide` and the materials so it is self-sufficient.
- `docs/ARCHITECTURE.md` documents `photonix.em` — roughly half the codebase, and
  previously absent from the layer diagram — plus a solver-selection table, the
  port convention and the layout → simulation loop. `README.md` no longer
  describes `photonix.modes` as "vectorial" (it is scalar).
- CI runs doctests and the example scripts, which is why the failures above went
  unnoticed.
- `tests/test_modes.py` now tests the facade contract (identity with
  `photonix.em`) rather than duplicating the numerical validation that
  `tests/test_em_fde.py` already performs against the canonical solver.

### Added
- `photonix.core.types.AliasedSDict` — an `SDict` that resolves legacy port names
  on lookup without storing them as extra terminals.
- `photonix.UNAVAILABLE` — maps any subpackage skipped for a missing optional
  dependency to the reason why.
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

- `photonix.em` exposes the `meep` backend lazily (via module `__getattr__`) so the
  core package never imports MEEP eagerly.
- `docs/DESIGN_EM_SOLVERS.md` extended with §17 documenting the MEEP backend.

## [0.1.0]

### Added
- Initial beta: differentiable core (`SDict` types, JAX/NumPy backend), component
  models, circuit S-parameter solver, mode/FDE/FDFD/EME solvers, layout/GDS,
  example PDK, visualization, inverse-design optimizers, and an internal benchmark
  suite.
