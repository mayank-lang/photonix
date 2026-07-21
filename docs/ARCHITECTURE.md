# photonix architecture

photonix is a differentiable, GPU-accelerated library for **photonic integrated
circuit (PIC)** design and simulation. It is built on JAX and unifies the parts
of the photonic design flow that today live in separate tools.

## Why photonix

| Concern | State of the art today | photonix |
|---|---|---|
| Circuit S-parameter sim | SAX (JAX, circuit-only) | Built-in, validated, same JAX core |
| Component models | scattered / per-PDK | Unified physics-based, differentiable library |
| Mode solving | femwell / Tidy3D / Lumerical (separate) | Built-in vectorial FDFD solver |
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
   physics        modes/  (cross-section eigenmode solver)
                 └───────────────────────┬──────────────────────┘
                 ┌───────────────────────┴──────────────────────┐
   foundation     core/  backend(JAX) · types(SDict) · units · sparams
                 └──────────────────────────────────────────────┘
   viz/  reads scattering objects, mode fields, layouts → plots
```

Data flows **up**: `modes` computes effective indices that feed `components`,
which emit scattering dictionaries that `circuit` combines, which `optim`
differentiates. `layout` produces GDS and extracts netlists that `circuit` can
simulate. `viz` reads any of these. Everything below a module is a hard
dependency; nothing imports from a layer above it.

## The central abstraction: `SDict`

A component's optical behaviour is an `SDict`:

```python
SDict = dict[tuple[str, str], complex_array]   # (in_port, out_port) -> amplitude
```

A **model** is a pure function `f(*, wl=1.55, **params) -> SType`. Because values
are JAX arrays, `jax.grad` flows through models and through the circuit solver
unchanged. See `core/types.py` and `BUILD_SPEC.md` for the exact contract.

## Backend

All numerical code imports `from photonix.core.backend import xp, jit, grad`
and uses `xp` (JAX `numpy`, or NumPy fallback). 64-bit precision is enabled by
default. GPU/TPU execution is automatic when JAX sees an accelerator.
