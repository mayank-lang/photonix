# Getting started with photonix

This tutorial walks through the core workflow: simulate a circuit, read its
spectrum, run a mode solve, and inverse-design a component — all differentiable.

## 1. Install

```bash
pip install -e ".[all]"     # from the repo root
```

photonix enables 64-bit precision automatically and uses JAX if present
(`photonix.HAS_JAX`), falling back to NumPy otherwise.

## 2. Simulate a circuit

Every component is a function returning a scattering dictionary
(`SDict`): `{(in_port, out_port): complex_amplitude}`. Ports are named
`o1, o2, ... oN` throughout photonix. Circuits are built by connecting ports and
solved differentiably.

```python
import photonix as px

wl = px.linspace(1.50, 1.60, 1001)        # µm
mzi = px.circuit.mzi(delta_length=40.0)    # two couplers + two arms
s = mzi(wl=wl)
T_bar = px.power(s[("o1", "o4")])          # |S|^2 bar transmission
T_cross = px.power(s[("o1", "o3")])        # cross output
```

The MZI follows the 2×2 coupler convention: `o1`/`o2` in, `o3`/`o4` out, bar
path `o1 → o4`. The older names (`in0`, `out0`, ...) still resolve as aliases.

The solver eliminates internal ports with a single linear solve, so it handles
feedback loops (ring resonators) exactly and stays differentiable and
`jit`-able.

## 3. Plot a spectrum

```python
import matplotlib; matplotlib.use("Agg")
ax = px.viz.plot_spectrum(s, wl, [("o1", "o4"), ("o1", "o3")])
ax.figure.savefig("mzi.png")
```

## 4. Solve a waveguide mode

All solvers live in `photonix.em`. (`photonix.modes` still works, but it is now
just a compatibility facade over the same functions.)

```python
import photonix.em as em
r = em.solve_modes(wl=1.55, width=0.5, thickness=0.22, resolution=50)
print(r.neff0)                              # scalar effective index
ng = em.group_index(wl=1.55, width=0.5, thickness=0.22, resolution=25)
```

`solve_modes` is the **scalar** solver: fast, and exact in the low-contrast
limit, but it overestimates the index of a high-contrast SOI strip. For the
physical quasi-TE value use the full-vector solver:

```python
em.n_eff_fullvector(wl=1.55, width=0.5, thickness=0.22)   # ≈2.45 vs scalar ≈2.61
```

You can feed a real mode solve straight into a waveguide model:

```python
neff_fn = lambda wl: em.n_eff(wl=float(wl), width=0.5, thickness=0.22)
wg = px.components.straight(wl=1.55, length=100.0, neff=neff_fn)
```

## 5. Inverse design (the differentiable payoff)

Define a model, a differentiable objective, and optimize with gradients that flow
through the whole simulation:

```python
import photonix.optim as opt
loss = opt.make_loss(px.components.directional_coupler,
                     opt.target_transmission,
                     wl=1.55, port=("o1", "o3"), target=0.25)
res = opt.adam(loss, {"coupling": 0.5}, steps=300, lr=0.02)
print(res.params["coupling"])               # tuned to hit the target split
```

Because `grad` flows end to end, the same pattern optimizes any exposed parameter
of any circuit — couplers, arm lengths, ring gaps — toward any differentiable
figure of merit.

## 6. Layout & GDS

```python
import photonix.layout as lay
from photonix.layout import components as lc
top = lay.Cell("top")
top.add_ref(lc.straight(10.0), origin=(0, 0), name="a")
top.add_ref(lc.straight(10.0), origin=(10, 0), name="b")
lay.write_gds(top, "demo.gds")
nl = lay.extract_netlist(top)               # -> a circuit.Netlist you can simulate
s  = px.circuit.circuit_from_netlist(nl)(wl=1.55)   # models default to components.MODELS
```

Extraction connects reference ports whose centres coincide, and names the
remaining (unconnected) ports `"{instance}_{port}"` — so above, the composite
runs from `"a_o1"` to `"b_o2"`.

See the `examples/` directory for runnable scripts.
