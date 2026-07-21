# photonix

PIC based computational solver Python package compatible with MEEP (another OSS)

photonix tries to bring the photonic design stack into a single library built on
[JAX](https://github.com/google/jax).


> Status: **v1.** MIT-licensed. Built for researchers.


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

## Example

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
