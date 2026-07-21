# photonix BUILD SPEC — contract for module builders

This document is the **binding interface contract**. Every builder reads it and
conforms to it so the independently-built modules compose into one coherent,
differentiable library. Do not change the core contract; build on top of it.

## Ground rules (all builders)

1. **Repo root (use these EXACT paths):**
   - Write files with Read/Write/Edit at: `C:\Users\My PC\Desktop\Photonics\photonix\...`
   - Run code with bash at: `/sessions/charming-gracious-knuth/mnt/Photonics/photonix/...`
   - Source lives under `src/photonix/<yourpackage>/`. Tests under `tests/`.
2. **Own only your directory.** Do NOT edit `src/photonix/__init__.py`,
   `pyproject.toml`, or another builder's package. The top-level package already
   imports your subpackage defensively once it exists. If you need a shared
   helper that doesn't exist, add it inside your own package.
3. **Backend discipline.** Never `import numpy`/`import jax` directly in library
   code. Use:
   ```python
   from photonix.core.backend import xp, jit, grad, value_and_grad, vmap
   ```
   Use `xp` for all array math so code runs on CPU/GPU and stays differentiable.
   For immutable updates use the JAX pattern `arr.at[idx].set(v)` via the helper
   `photonix.core.sparams._set` or write functionally.
4. **Differentiability is mandatory** for anything in the modeling path
   (`components`, `circuit`, `modes` forward solve, `optim`). No Python-side data
   dependent branching on traced values; no in-place mutation of traced arrays;
   prefer `xp.where` over `if`. Each such module must include at least one test
   that takes `grad` of an output w.r.t. a parameter and checks it is finite and
   matches finite differences to ~1e-4.
5. **Accuracy-first.** Default to 64-bit (already enabled). Validate against an
   analytic limit or published value where one exists; put that in a test.
6. **Style.** Type hints on public functions, NumPy-style docstrings with a short
   example, `from __future__ import annotations` at top, ruff-clean
   (line length 100). Public names listed in each module's `__all__` and the
   subpackage `__init__.py`.
7. **Tests.** Put tests in `tests/test_<yourpackage>_*.py`. They must pass with
   `cd .../photonix && PYTHONPATH=src python -m pytest tests/test_<yourpackage>_*`.
   If JAX isn't importable in your shell yet, you may also validate with
   `PHOTONIX_BACKEND=numpy` (autodiff tests will be skipped in that mode — guard
   them with `pytest.mark.skipif(not photonix.HAS_JAX, ...)`).
8. **No network at runtime.** Pure-Python + the dependencies in `pyproject.toml`
   (numpy, scipy, jax, gdstk, matplotlib, networkx). Don't add new heavy deps.

## The core contract (already built — import, don't reimplement)

```python
from photonix.core import (
    xp, jit, grad, value_and_grad, vmap,          # backend
    SDict, SDense, SCoo, SType, Model, PortName,  # types
    as_sdict, as_sdense, sdict_to_sdense, sdense_to_sdict,
    reciprocal, is_reciprocal, is_passive, power, insertion_loss_db,
    validate_sdict, ports_of,
)
from photonix.core import constants, units
```

**Scattering dictionary** — the universal interchange format:
```python
SDict = dict[tuple[in_port: str, out_port: str], complex_amplitude]
```
* Keys are `(input_port, output_port)` name pairs.
* Values are complex field-amplitude coefficients; may be arrays broadcast over a
  wavelength sweep. `|value|**2` is the power fraction.
* Only nonzero couplings need to be present.

**Model signature** — every component/circuit is a callable:
```python
def my_component(*, wl=1.55, **params) -> SDict: ...
```
* MUST accept keyword `wl` (wavelength in µm; scalar or 1-D array) and return an
  `SType` (usually an `SDict`).
* MUST have defaults for every parameter, so `my_component()` works.
* MUST be a pure function (no globals, no mutation) → differentiable & jit-able.

**Conventions**
* Lengths and wavelengths are in **micrometers (µm)** unless documented otherwise.
* Port naming: optical ports `o1, o2, ...` OR semantic `in0, out0, ...`. Be
  consistent within a component and document the ports in the docstring.
* Reciprocal passive components: include both `(a,b)` and `(b,a)`. Use
  `reciprocal()` to symmetrize.
* Use `units.db_per_cm_to_alpha_um`, `units.wl_to_freq`, etc. for conversions and
  `constants` for material indices / physical constants.

## Module assignments (non-overlapping)

### Builder 1 — `src/photonix/circuit/`  (the differentiable solver)
- `netlist.py`: a `Netlist`/`Circuit` data structure — instances (name -> model +
  settings), connections (`("inst1","o2") <-> ("inst2","o1")`), and exposed
  ports. Support hierarchical settings.
- `solver.py`: combine component `SDict`s into one composite `SDict` by
  eliminating internal ports. Implement the linear interconnection algorithm
  (Gaussian elimination / partial-trace on the dense form is acceptable and is
  fully differentiable). Must be `jit`/`grad` compatible and handle wavelength
  batch dims. Provide `circuit_from_netlist(netlist, models) -> Model`.
- Convenience builders that other code expects: `mzi(...)`, `ring(...)` returning
  ready circuit models (these compose `components` if available, else accept
  models as args — keep a soft dependency: import components lazily).
- Algorithm reference: interconnection of scattering matrices (e.g. the
  "SAX"/"Klu" partial-trace / Gauss elimination of internal ports).
- Tests: cascade of two known 2-ports equals analytic product; a loop (ring)
  matches the analytic all-pass ring transfer function; gradient of an output
  power w.r.t. a coupler parameter matches finite differences.

### Builder 2 — `src/photonix/components/`  (model library)
Physics-based, differentiable, parametric models returning `SDict`s. Each in its
own file with `__all__`:
- `waveguide.py`: `straight(length, wl, neff, ng, loss_db_cm, wl0)` with proper
  dispersive phase `exp(-j*beta*L)`, `beta = 2π/λ * n_eff(λ)`; `bend(...)`.
- `couplers.py`: `directional_coupler(coupling)` (lossless 2x2 with `t,κ` and the
  90° phase between bar/cross), `coupler(...)`, ideal `mmi1x2`, `mmi2x2`.
- `resonators.py`: building blocks for ring/racetrack if needed (the assembled
  ring lives in `circuit`, but a `ring_coupler` and analytic `ring_response`
  reference are useful and testable).
- `mzi.py`: a parametric MZI model (analytic, for validation/reference).
- `gratings.py`: `grating_coupler` (Gaussian reflection/transmission vs λ),
  `phase_shifter` (length/voltage -> phase), `terminator`, `attenuator`.
- Accept an `neff`/`ng` either as numbers or as callables of `wl` (so a real mode
  solve can be plugged in). Provide sensible 220 nm SOI strip defaults from
  `constants`.
- Tests: directional coupler is unitary & reciprocal and `|t|^2+|κ|^2=1`;
  straight waveguide phase matches `2π n L/λ`; gradient of MZI transmission
  w.r.t. `delta_length` matches finite differences.

### Builder 3 — `src/photonix/modes/`  (eigenmode solver)
- `materials.py`: dispersive index models — Sellmeier for SiO₂/Si₃N₄, a silicon
  model — as callables `n(wl_um)`. Differentiable.
- `geometry.py`: define a 2-D cross-section (rectangular waveguide core in
  cladding) on a grid → permittivity array.
- `solver.py`: a **vectorial (or robust semi-vectorial) FDFD** eigenmode solver
  returning effective index `n_eff`, group index `n_g` (via `dω`/finite diff over
  λ), and field profiles for the first N modes. Use `scipy.sparse.linalg.eigs`
  for the eigenproblem (NumPy/scipy is fine here — this is the one place a sparse
  eig is acceptable; keep the *interface* returning plain arrays so values feed
  components). Provide `n_eff(width, thickness, wl) -> float` convenience.
- Provide `overlap(mode_a, mode_b)` integral.
- Tests: a slab/symmetric case matches the analytic slab-waveguide effective
  index to ~1e-3; n_eff decreases with λ (normal dispersion) for a Si strip;
  grid convergence (finer grid → smaller error).

### Builder 4 — `src/photonix/layout/` and `src/photonix/pdk/`
- `layout/cell.py`: a `Cell` of polygons/paths/references with ports (name,
  position, orientation, width, layer).
- `layout/components.py`: parametric layout generators (straight, bend, taper,
  ring, mmi, grating coupler footprint) producing `Cell`s with ports.
- `layout/routing.py`: simple manhattan/`route` between two ports.
- `layout/gds.py`: export a `Cell` tree to GDSII via `gdstk`; import too if easy.
- `layout/extract.py`: walk a cell's references + port connectivity to produce a
  `circuit.Netlist` (soft import of `circuit`).
- `pdk/base.py`: a `Pdk` registry mapping component names -> (layout generator,
  circuit model, default settings), plus layer definitions.
- `pdk/example_pdk.py`: one open example PDK ("photonix_demo") wiring the
  `components` models + `layout` generators with 220 nm SOI defaults.
- Tests: build a cell, export GDS to a temp file and re-read it (cell/poly counts
  match); extract a netlist from a 2-component layout and confirm the connection.

### Builder 5 — `src/photonix/viz/`, `src/photonix/optim/`, infra
- `viz/spectrum.py`: plot transmission/phase vs λ from an `SDict` (matplotlib,
  return `Axes`, never call `plt.show()` in library code).
- `viz/modes.py`: plot a mode field profile; `viz/layout.py`: render a `Cell`;
  `viz/circuit.py`: draw a netlist as a graph (networkx optional).
- `optim/objectives.py`: common figures of merit (target transmission, flat-top,
  extinction ratio) as differentiable functions of an `SDict`.
- `optim/adjoint.py`: thin wrappers around `value_and_grad` for circuit params;
  `optim/optimizers.py`: a minimal Adam loop (pure JAX) + scipy L-BFGS bridge.
- Infra: `tests/conftest.py` (shared fixtures, `pytest.importorskip` helpers),
  `README` sections you own, `docs/` tutorials, `.github/workflows/ci.yml`
  (lint + tests on 3.10–3.12), `examples/` scripts (MZI sweep, ring resonator,
  inverse-design a coupler). Make `examples/` runnable end-to-end.

## Definition of done (per builder)
- Subpackage imports cleanly: `python -c "import photonix; import photonix.<pkg>"`.
- `__init__.py` exports the public API with `__all__`.
- Tests in `tests/` pass; at least one differentiability test and one
  accuracy/analytic test.
- NumPy-style docstrings with a runnable example on every public function.
- ruff-clean. No edits outside your assigned directories.
