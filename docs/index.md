# photonix

photonix is an accuracy-first Python toolkit for photonic integrated circuit
(PIC) design. It connects differentiable compact models, S-parameter circuit
simulation, electromagnetic mode solvers, layout and GDSII exchange, process
studies and external verification hand-offs through one port and data model,
on a JAX backend with a NumPy fallback.

```bash
pip install "photonix[all]"     # or: pip install photonix   (NumPy/SciPy core only)
photonix info                    # backend, precision, optional dependencies
photonix validate --quick        # analytic self-checks of this installation
```

```python
import photonix as px

wl = px.linspace(1.50, 1.60, 501)                    # wavelengths in um
s = px.circuit.mzi(delta_length=20.0)(wl=wl)          # two-coupler MZI, solved as a circuit
bar = px.power(s[("o1", "o4")])                       # |S|^2 of the bar path
g = px.grad(lambda dl: px.power(px.circuit.mzi(delta_length=dl)(wl=1.55)[("o1", "o4")]))(20.0)
```

Every model and solver states its evidence level (see
[Validation](VALIDATION.md)); none of them is a substitute for a foundry's
qualified PDK, DRC/LVS decks or tape-out sign-off
([PIC completeness](PIC_COMPLETENESS.md)).

```{toctree}
:maxdepth: 1
:caption: Getting started

tutorial_getting_started
tutorial_design_flow
CLI
examples
```

```{toctree}
:maxdepth: 1
:caption: User guide

theory/conventions
COMPONENT_LIBRARY
VALIDATION
ARCHITECTURE
DESIGN_EM_SOLVERS
INTEGRATIONS
PIC_COMPLETENESS
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
changelog
```

The source, issue tracker and release history are at
<https://github.com/mayank-lang/photonix>. To cite photonix, use the
repository's `CITATION.cff` (GitHub's "Cite this repository").
