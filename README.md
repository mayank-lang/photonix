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
T = px.power(S[("o1", "o4")])               # bar transmission spectrum

# Gradient of mean transmission w.r.t. the path imbalance — for free
def fom(dl):
    return px.power(px.circuit.mzi(delta_length=dl)(wl=wl)[("o1", "o4")]).mean()
g = px.grad(fom)(20.0)
```

Cross-section physics, and the layout → simulation loop:

```python
import photonix.em as em

em.n_eff_fullvector(wl=1.55, width=0.5, thickness=0.22)   # ≈2.45, SOI TE0

nl = px.layout.extract_netlist(top_cell)   # coincident ports → connections
S  = px.circuit.circuit_from_netlist(nl)(wl=1.55)   # models default to px.components.MODELS
```

## Package layout

| Subpackage | Purpose |
|---|---|
| `photonix.core` | Backend, units, constants, the `SDict` type system, S-param utils |
| `photonix.em` | All EM physics: scalar/semivector/full-vector FDE, slab, EIM, FDFD, EME, bend loss, cross-section geometry, dispersive materials, plus an optional MEEP/MPB **FDTD backend** (`photonix.em.meep`) |
| `photonix.components` | Differentiable, parametric component models (waveguides, couplers, MZIs, rings, gratings) |
| `photonix.circuit` | Netlist + differentiable S-parameter circuit solver |
| `photonix.modes` | Compatibility facade re-exporting `photonix.em`'s scalar solver, geometry and materials — *deprecated*, prefer `photonix.em` |
| `photonix.layout` | Cells, routing, GDSII export, netlist extraction |
| `photonix.pdk` | PDK-agnostic interface + an open example PDK |
| `photonix.viz` | Plot spectra, mode profiles, layouts, circuit graphs |
| `photonix.optim` | Objectives, adjoint helpers, optimizers for inverse design |

**Port naming.** Optical ports are `o1, o2, ... oN` everywhere. The legacy
semantic names (`in0`/`out0`, `i1`/`t1`/`d2`) still work as lookup aliases; see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#port-naming).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design, and
[`BUILD_SPEC.md`](BUILD_SPEC.md) for the developer contract.

## License

MIT — see [`LICENSE`](LICENSE).
