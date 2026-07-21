# photonix

**Differentiable, GPU-accelerated, accuracy-first design and simulation of
photonic integrated circuits.**

photonix unifies the photonic design stack — cross-section mode solving,
component modeling, circuit-level S-parameter simulation, layout/GDS, and
inverse design — into a single library built on [JAX](https://github.com/google/jax).
Because the entire pipeline is differentiable, a single `grad` call gives the
sensitivity of a *system-level* figure of merit with respect to *any* physical
parameter, from a waveguide width to a coupler gap.

> Status: **v0.1 (beta).** MIT-licensed. Built for researchers.

## Why another photonics package?

Today the photonic design flow is split across separate tools: layout in one
package, mode solving in another, circuit simulation in a third, inverse design
in a fourth. photonix brings them together behind one consistent, functional API
and one differentiable numerical core:

- **Differentiable everywhere** — gradients flow through component models *and*
  the circuit solver, so adjoint/inverse design works out of the box.
- **GPU/TPU-ready** — the JAX backend runs unchanged on accelerators; the heavy
  loops are `jit`-compiled and vectorized over wavelength with `vmap`.
- **Accuracy-first** — 64-bit by default; models validated against analytic
  limits and the literature in the test suite.
- **Composable** — components are pure functions returning scattering
  dictionaries; circuits are built by connecting named ports.

## Install

```bash
pip install -e ".[all]"      # full feature set (jax, gdstk, matplotlib, networkx)
# or a minimal core:
pip install -e .
```

photonix runs on a NumPy fallback backend when JAX is absent (autodiff disabled);
install the `jax` extra for the full differentiable experience.

### Optional: FDTD via MEEP

photonix does not ship its own time-domain Maxwell solver. Workflows that need
FDTD are delegated to [MIT MEEP](https://meep.readthedocs.io) through the optional
`photonix.em.meep` backend. MEEP is conda-only (Linux/macOS; on Windows use WSL2),
so it is **not** a pip dependency:

```bash
conda install -c conda-forge pymeep        # or "pymeep=*=mpi_mpich_*" for the parallel build
```

Without MEEP, `import photonix` and `import photonix.em` work normally; only
touching `photonix.em.meep` raises an `ImportError` with this install hint.

## Quick start

```python
import photonix as px

wl = px.linspace(1.50, 1.60, 501)          # wavelength sweep [µm]

# Build a Mach–Zehnder interferometer and simulate it differentiably
mzi = px.circuit.mzi(delta_length=20.0)     # µm path imbalance
S = mzi(wl=wl)
T = px.power(S[("in0", "out0")])            # transmission spectrum

# Gradient of mean transmission w.r.t. the path imbalance — for free
import photonix as px
def fom(dl):
    return px.power(px.circuit.mzi(delta_length=dl)(wl=wl)[("in0", "out0")]).mean()
g = px.grad(fom)(20.0)
```

## Package layout

| Subpackage | Purpose |
|---|---|
| `photonix.core` | Backend, units, constants, the `SDict` type system, S-param utils |
| `photonix.em` | Rigorous EM solvers: scalar/semivector/full-vector FDE, FDFD, EME, bend loss, plus an optional MEEP/MPB **FDTD backend** (`photonix.em.meep`) |
| `photonix.components` | Differentiable, parametric component models (waveguides, couplers, MZIs, rings, gratings) |
| `photonix.circuit` | Netlist + differentiable S-parameter circuit solver |
| `photonix.modes` | Vectorial cross-section eigenmode solver (n_eff, n_g, profiles) |
| `photonix.layout` | Cells, routing, GDSII export, netlist extraction |
| `photonix.pdk` | PDK-agnostic interface + an open example PDK |
| `photonix.viz` | Plot spectra, mode profiles, layouts, circuit graphs |
| `photonix.optim` | Objectives, adjoint helpers, optimizers for inverse design |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design, and
[`BUILD_SPEC.md`](BUILD_SPEC.md) for the developer contract.

## License

MIT — see [`LICENSE`](LICENSE).
