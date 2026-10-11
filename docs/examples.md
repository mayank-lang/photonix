# Examples

Each script in [`examples/`](../examples/) runs headless (`MPLBACKEND=Agg`) and
writes its figure to `examples/outputs/`; CI runs all of them on every push.
The figures below are the committed outputs of those scripts.

```bash
python examples/ring_resonator.py
```

## Ring resonator as a feedback circuit

The all-pass ring solved by the circuit solver agrees with its closed form to
machine precision.

```{image} ../examples/outputs/ring_resonator.png
:alt: ring resonator through spectrum
```

```{literalinclude} ../examples/ring_resonator.py
:language: python
```

## Mach-Zehnder interferometer

```{image} ../examples/outputs/mzi_sweep.png
:alt: MZI bar and cross spectra
```

```{literalinclude} ../examples/mzi_sweep.py
:language: python
```

## Mode solver

```{image} ../examples/outputs/mode_profile.png
:alt: fundamental mode of a silicon strip
```

```{literalinclude} ../examples/mode_solver.py
:language: python
```

## Full-vector EM showcase

```{image} ../examples/outputs/vector_em_showcase.png
:alt: full-vector mode, accuracy ladder, bend loss and EME/FDFD agreement
```

```{literalinclude} ../examples/vector_em_showcase.py
:language: python
```

## Layout to circuit

```{image} ../examples/outputs/layout_to_circuit.png
:alt: routed ring filter layout and its simulated spectrum
```

```{literalinclude} ../examples/layout_to_circuit.py
:language: python
```

## Bragg gratings and a ring modulator

```{image} ../examples/outputs/bragg_and_ring_modulator.png
:alt: Bragg grating variants and ring modulator spectra
```

```{literalinclude} ../examples/bragg_and_ring_modulator.py
:language: python
```

## AWG demultiplexer

```{image} ../examples/outputs/awg_demux.png
:alt: AWG channel spectra and yield
```

```{literalinclude} ../examples/awg_demux.py
:language: python
```

## Programmable MZI mesh

```{image} ../examples/outputs/programmable_mesh.png
:alt: realized DFT on an MZI mesh with fabrication errors
```

```{literalinclude} ../examples/programmable_mesh.py
:language: python
```

## Inverse-designed coupler

```{image} ../examples/outputs/inverse_design_coupler.png
:alt: coupler optimization history
```

```{literalinclude} ../examples/inverse_design_coupler.py
:language: python
```

## Command-line hand-off

Sweep a model to Touchstone, gate it with `photonix sparams check`, and read it
back (see [Command-line interface](CLI.md)).

```{image} ../examples/outputs/cli_handoff.png
:alt: ring through spectrum read back from a Touchstone file
```

```{literalinclude} ../examples/cli_handoff.py
:language: python
```

## Accuracy workflow

An analytic anchor, a convergence check, an S-parameter physicality report and a
runtime manifest, without claiming foundry qualification.

```{literalinclude} ../examples/accuracy_workflow.py
:language: python
```
