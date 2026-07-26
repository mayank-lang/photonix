# photonix architecture

photonix is a differentiable, GPU-accelerated library for **photonic integrated
circuit (PIC)** design and simulation. It is built on JAX and unifies the parts
of the photonic design flow that today live in separate tools.

## Why photonix

| Concern | State of the art today | photonix |
|---|---|---|
| Circuit S-parameter sim | SAX (JAX, circuit-only) | Built-in, validated, same JAX core |
| Component models | scattered / per-PDK | Unified physics-based, differentiable library |
| Mode solving | femwell / Tidy3D / Lumerical (separate) | Built-in scalar, semivectorial and full-vector FDE solvers |
| Layout / GDS | gdsfactory (layout-only) | Built-in layout + netlist extraction |
| Inverse design | ceviche / SPINS-B (separate) | End-to-end gradients across the whole stack |

The thesis: **one differentiable graph from cross-section geometry → component
models → circuit response → figure of merit**, so a single `grad` call gives
sensitivities of a system-level metric with respect to any physical parameter.

## Layered design

```
                 ┌─────────────────────────────────────────────┐
   optim/  ◄────  inverse design, adjoint, optimizers, objectives
                 └─────────────────────────────────────────────┘
                 ┌──────────────┐   ┌──────────────┐  ┌─────────┐
   high level     circuit/        components/        layout/+pdk/
   (modules)      netlist+solver   model library      geometry+GDS
                 └──────┬───────┘   └──────┬───────┘  └────┬────┘
                        │                  │  ▲            │
                 ┌──────┴──────────────────┴──┴────────────┴────┐
   physics        em/   FDE (scalar · semivector · full-vector) ·
                        slab · EIM · EME · FDFD · bend loss ·
                        geometry · materials · [meep] FDTD backend
                        modes/  → compatibility facade over em/
                 └───────────────────────┬──────────────────────┘
                 ┌───────────────────────┴──────────────────────┐
   foundation     core/  backend(JAX) · types(SDict) · units · sparams
                 └──────────────────────────────────────────────┘
   viz/  reads scattering objects, mode fields, layouts → plots
```

Data flows **up**: `em` computes effective indices and rigorous S-parameters
that feed `components`, which emit scattering dictionaries that `circuit`
combines, which `optim` differentiates. `layout` produces GDS and extracts
netlists that `circuit` can simulate. `viz` reads any of these. Everything below
a module is a hard dependency; nothing imports from a layer above it.

### `em/` vs `modes/`

`photonix.em` is the physics layer and holds every solver. `photonix.modes` is a
**thin compatibility facade** that re-exports `em`'s scalar solver, cross-section
geometry, and material models under their older names — the objects are
identical (`photonix.modes.n_eff is photonix.em.n_eff`), not copies.

The two were once independent implementations exporting the same names and
returning different numbers (2.644 vs 2.612 for a 500×220 nm SOI strip, because
only `em` did subpixel averaging and Richardson extrapolation). New code should
import from `photonix.em`.

Pick a solver by index contrast:

| Solver | Call | Use when |
|---|---|---|
| Scalar FDE | `em.n_eff` | Low/medium contrast; fast first estimate |
| Semivectorial | `em.n_eff_vector` | Quasi-TE/TM splitting matters |
| Full-vector | `em.n_eff_fullvector` | High contrast (SOI strips) — the physical answer |
| FDTD | `em.meep.*` | Time domain; requires the optional MEEP backend |

For a 500×220 nm SOI strip at 1.55 µm the scalar solver gives ≈2.61 and the
full-vector solver ≈2.45; the latter is the one to quote.

### Layout → simulation

`layout.extract_netlist` returns a `circuit.Netlist` whose model names are the
layout cell names, and `circuit.circuit_from_netlist` defaults its registry to
`components.MODELS`, so the loop closes with no manual wiring:

```python
nl = photonix.layout.extract_netlist(top_cell)
S  = photonix.circuit.circuit_from_netlist(nl)(wl=1.55)
```

## The central abstraction: `SDict`

A component's optical behaviour is an `SDict`:

```python
SDict = dict[tuple[str, str], complex_array]   # (in_port, out_port) -> amplitude
```

A **model** is a pure function `f(*, wl=1.55, **params) -> SType`. Because values
are JAX arrays, `jax.grad` flows through models and through the circuit solver
unchanged. See `core/types.py` and `BUILD_SPEC.md` for the exact contract.

### Port naming

Optical ports are named `o1, o2, ... oN` in **every** component, circuit builder
and layout cell — one convention, no exceptions. That is what makes a layout
cell's ports line up with its model's ports during netlist extraction.

A few early models used semantic names (`in0`/`out0` on the MZI,
`i1`/`t1`/`d2` on the add-drop ring). Those still resolve, as *read-only
aliases*, through `core.types.AliasedSDict`: the mapping stores only canonical
keys, so `ports_of`, the solver and the passivity checks see one terminal per
physical port, while `s[("in0", "out0")]` keeps working. Aliases are never
stored as extra entries — doing so would double-count terminals and corrupt any
circuit built from the model.

## Backend

All numerical code imports `from photonix.core.backend import xp, jit, grad`
and uses `xp` (JAX `numpy`, or NumPy fallback). 64-bit precision is enabled by
default. GPU/TPU execution is automatic when JAX sees an accelerator.
