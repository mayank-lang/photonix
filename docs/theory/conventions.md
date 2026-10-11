# Conventions

These conventions hold across every photonix model, solver, file format and the
command line. Getting one of them wrong when exchanging data with another tool
is the most common source of silently wrong results, so they are collected here.

## Units

| Quantity | Unit | Notes |
|---|---|---|
| Wavelength `wl`, lengths, widths, gaps, radii | µm | vacuum wavelength |
| Propagation loss `loss_db_cm` | dB/cm | power loss; the field decays as `exp(-alpha L)` with `alpha = loss_db_cm * ln(10) / 20 / 1e4` per µm |
| Group delay | s | `-d arg(S) / d omega` on the exact sampled frequency grid |
| Group-delay dispersion | s² | `-d² arg(S) / d omega²` |
| Touchstone frequency | Hz, kHz, MHz or GHz | converted with `c = 299 792 458 m/s` |
| Layout coordinates | µm | GDSII is written with a 1 µm user unit and a 1 nm database unit |

## Time and propagation sign

photonix uses the engineering phasor convention: fields carry an implied
`exp(+j omega t)` time dependence, so a forward wave is `exp(j (omega t - beta z))`
and propagation over a length `L` multiplies the amplitude by

```text
exp(-j beta L) = exp(-j 2 pi n_eff L / wl)
```

Every compact model (for example `components.straight`, whose transmission is
`exp(-alpha L) exp(-j beta L)`), the EME solver (`exp(-i beta z)`, see
`photonix.em.eme`) and the group-delay helpers follow this sign. A positive
group delay therefore corresponds to a phase that *decreases* with frequency.

## Complex refractive index

With `exp(+j omega t)`, an absorbing medium or a decaying mode has a **negative**
imaginary part: `n = n' - j n''` with `n'' > 0`, and `Im(beta) < 0` for a mode
that decays along `+z`. The EME solver picks that branch for evanescent modes.

Tools that use the physics convention `exp(-i omega t)` (Meep, MPB and many
optics texts) write the same absorbing medium as `n' + i n''`. **Conjugate complex indices and permittivities when moving data
between the two conventions.** The FDE mode solvers accept real permittivity
only and reject complex input instead of truncating it
(`photonix.em.geometry.as_real_eps`); bend and PML losses are reported through
`|Im(n_eff)|` (see `photonix.em.bend_loss_fullvector`).

## S-parameters

* **Keys.** An `SDict` maps `(in_port, out_port)` to the complex amplitude
  transmitted from `in_port` to `out_port`. Dense matrices use the standard
  `S[out, in]` order (`SDense = (S, port_map)` with `S` of shape
  `(..., n_ports, n_ports)`, wavelength axes first).
* **Normalization.** Amplitudes are power-normalized modal waves: `|S|²` is a
  power fraction, a passive network satisfies `S^H S <= I` (largest singular
  value at most 1), a reciprocal one `S = S^T`, and a lossless one is unitary.
  `photonix.analyze_sparameters` and `photonix sparams check` test exactly these
  statements.
* **Reference planes.** A model's phases are referred to its ports as drawn: a
  `straight` of length `L` contributes `beta L` from port to port, and layout
  cells place their ports at the same planes, so an extracted circuit adds no
  hidden phase.
* **Touchstone.** Files are Touchstone 1.0, single-ended, `S RI`, with
  increasing frequency. Two-port data uses the legacy order
  `S11 S21 S12 S22`; larger matrices are written row by row (`S[out, in]`).
  The reference resistance (50 Ω unless the data says otherwise) is recorded
  but has no physical meaning for optical modal waves. Port names and metadata
  survive in `! Photonix ...` comment lines.

## Port naming

Optical ports are named `o1, o2, ... oN` in every model, circuit builder and
layout cell; that is what lets layout extraction connect a cell's ports to its
model's ports. Two-by-two couplers and MMIs number their ports
counter-clockwise from the top-left one: `o1` top-left, `o2` bottom-left, `o3`
bottom-right, `o4` top-right, so the through (bar) paths are `o1 -> o4` and
`o2 -> o3` and the cross paths `o1 -> o3` and `o2 -> o4`. An NxN MMI has inputs
`o1..oN` on the left (top to bottom) and outputs `o(N+1)..o(2N)` on the right
(bottom to top). A crossing is `o1` west, `o2` north, `o3` east, `o4` south.
Each model's docstring states its own port map; the legacy names `in0/out0` and
`i1/t1/d2` resolve as read-only aliases (see
[ARCHITECTURE.md](../ARCHITECTURE.md#port-naming)).

## Models

A model is a pure function `f(*, wl=1.55, **params) -> SDict` with a default for
every parameter. Under the JAX backend (the default when JAX is installed) it is
differentiable and jit-able in its numeric parameters; with
`PHOTONIX_BACKEND=numpy` the same code runs on NumPy without autodiff. 64-bit
floating point is enabled at import.
