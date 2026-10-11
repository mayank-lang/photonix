# Filter synthesis

`photonix.synthesis` goes from a target response to component parameters,
which `photonix.circuit` then simulates. It covers the two workhorse
integrated filter families:

* **IIR (all-pole): coupled-resonator (CROW) filters.** `synthesize_crow`
  designs Butterworth or Chebyshev band-pass ring chains.
* **FIR: lattice (cascaded-MZI) filters.** `synthesize_lattice` peels a
  power-complementary polynomial pair into coupler ratios and heater phases.
  Typical uses are interleavers, WDM band splitters, flat-top MZI filters and
  gain-flattening or dispersion-compensating filters.

Both are exact for the lossless compact models they target. Every design is
checked against an independent netlist simulation. `coupling_to_gap` then maps
the designed couplings onto a calibrated coupler model.

## Coupled-resonator filters

```python
from photonix import synthesis as sy

d = sy.synthesize_crow(4, 100e9, kind="chebyshev", ripple_db=0.2, radius=10.0)
d.couplings          # (0.3453, 0.0446, 0.0299, 0.0446, 0.3453)
d.max_error_db       # ~4e-11 dB worst deviation from the target (netlist simulation)
```

How it works:

1. Little et al.'s closed form (JLT 15, 998, 1997) gives the starting
   couplings.
2. These are refined until the lossless drop response equals the periodic
   all-pole target `1 / (1 + eps^2 F_N(x)^2)` exactly, where
   `x = sin(phi/2) / sin(phi_c/2)`.
3. The couplings are kept symmetric. Wide bands are reached by continuation
   from a narrow one.

See section 7 of [`tutorial_design_flow.md`](tutorial_design_flow.md) for
mapping couplings to gaps.

## FIR lattice filters

A lattice of `N + 1` couplers and `N` stages behaves as follows:

* Each stage's upper arm is longer by `dL`, plus a heater phase `phi_k`.
* The bar output (`o1 -> o4`) is a degree-`N` polynomial `A(z)`, and the
  cross output (`o1 -> o3`) is `B(z)`.
* `z^-1 = exp(-j omega)`, with `omega = 2 pi df / FSR`.
* The pair is lossless: `|A|^2 + |B|^2 = 1`.

`dL` is rounded so that `neff dL / wl0` is an integer. This puts `omega = 0`
at `wl0`. With the linear-dispersion waveguide model the phase is exactly
linear in frequency, so the response is exactly periodic with
`FSR = c / (ng dL)`.

```python
import numpy as np
from photonix import synthesis as sy

d = sy.interleaver(3, fsr_hz=100e9)      # 50 GHz odd/even interleaver
d.couplings      # (0.9889, 0.2297, 0.8547, 0.5)
d.delays         # (2, 2, 1): unit delays per stage, dL = 713.65 um
d.phases         # (0, pi, pi) heater settings in radians
d.max_error      # 6e-16: bound on the complex-response error over the FSR
S = d.model()(wl=d.wavelength(np.linspace(-50e9, 50e9, 1001)))   # circuit.lattice_filter

# any power-complementary target; omitting the bar uses the minimum-phase partner
cross, bar = sy.equiripple_lowpass(5, 0.25 * np.pi, 0.75 * np.pi)
d2 = sy.synthesize_lattice(cross, bar, fsr_hz=100e9)
```

### API

| Name | Purpose |
|---|---|
| `synthesize_lattice(cross, bar=None, *, fsr_hz \| delta_length, neff, ng, wl0, length)` | Exact peeling; returns a `LatticeDesign` |
| `LatticeDesign` | `couplings`, `phases`, `delays`, `output_phase`, `common_phase`, `delta_length`, `fsr_hz`, `max_error`; `.model()`, `.wavelength(df)`, `.omega(wl)`, `.response(omega)`, `.target(omega)` |
| `interleaver(order, ...)` | `synthesize_lattice` of `maxflat_halfband(order)` |
| `maxflat_halfband(order)` | Maximally flat half-band pair (Daubechies polynomial), degree `2 order - 1` |
| `equiripple_lowpass(degree, wp, ws, weight=1)` | Parks-McClellan power response, factored into a lattice pair |
| `spectral_factor(r or R(omega), degree=None)` | Minimum-phase factor of a non-negative trigonometric polynomial, with its error bound |
| `complementary_bar(cross)` | Minimum-phase `A` with `|A|^2 = 1 - |B|^2` |
| `lattice_response(couplings, phases, omega, delays=None)` | Fast differentiable (JAX) transfer-matrix evaluation |
| `lattice_coefficients(couplings, phases, delays=None)` | The same response as polynomial coefficients |

The pairs come back as a `CrossBar(cross, bar)` named tuple, with cross first
everywhere.

### Algorithm

The conventions are those of `circuit.lattice_filter`:

* A coupler is `C = [[c, -j s], [-j s, c]]`, with `K = s^2`.
* Stage `k` is `diag(z^-1 e^{-j phi_k}, 1)`.
* The response is the first column of `C_N D_N ... D_1 C_0`.

Stage `N` comes off with `C(theta)^-1`, where
`tan theta = |a_0 / b_0| = |b_N / a_N|`. This clears the bar's constant term
and the cross's top term, so the delay divides out exactly.

The relative phase between the two arms is a free choice at each stage. At the
top it is the output phase; below that it is the stage phase. It is set so that
`a_0 / b_0` lies on the negative imaginary axis, which keeps
`theta in [0, pi/2]`. As a result, the decomposition is unique whenever
`0 < K < 1`: a random lattice synthesized from its own response returns
exactly its own couplings and phases.

Interior couplers that come out as `K = 0` (to 1e-12 rad) are identity
matrices, so their two stages merge exactly into one longer delay. This is how
the half-band designs get their `2 dL` stages.

The target's two output phases are not realizable by the lattice. They are
reported as a gauge:

* target cross `= exp(-j common_phase) * cross`
* target bar `= exp(-j (common_phase + output_phase)) * bar`

A phase shifter of `output_phase` on `o4` reproduces the complex target. Power
responses need nothing.

### Spectral factorization

The `2n` roots of `z^n R(z)` pair up as `rho` and `1 / rho*`. The minimum-phase
factor keeps the ones inside the unit circle.

Zeros *on* the circle need special care:

* They have even multiplicity `2m`.
* Numerically, they scatter into a ring of radius about `eps^(1/2m)`, which is
  0.05 for the 12-fold zero of a sixth-order max-flat complement.
* The ring's centroid stays accurate.

So the roots are clustered. Each even cluster centred on the circle
contributes `m` copies of its centroid, projected onto the circle. The
clustering radius is chosen by the smallest reconstruction error. A
Levenberg-Marquardt polish of the autocorrelation equations then finishes the
job.

The returned error `sum_k |r_k - r_k(h)|` is a rigorous bound on
`max |R - |H|^2|`. A factor whose error exceeds `tol * max R` raises an error
rather than being returned.

`maxflat_halfband` never root-finds a unit-circle zero. It takes out
`((1 + z^-1)/2)^N` analytically and factors only the Daubechies polynomial
`P(x)`, whose roots lie off `[0, 1]`.

`equiripple_lowpass` works in four steps:

1. Design a zero-phase minimax FIR `G` with `scipy.signal.remez`.
2. Find its exact extremes from the roots of `G'` in `cos omega`.
3. Map it to `|B|^2 = (G - G_min) / (G_max - G_min)`, following Herrmann &
   Schuessler, Electron. Lett. 6, 329, 1970.
4. Factor both `|B|^2` and `1 - |B|^2`.

### Validation evidence

All of these checks are in `tests/test_synthesis_lattice.py`.

| Check | Reference | Result |
|---|---|---|
| Round trip: random `K in (0.05, 0.95)` and phases, `N = 1..8` | The lattice itself | Same couplings and phases to 1e-12; output gauge recovered |
| Netlist simulation of `LatticeDesign.model()` vs `response(omega)` over a full FSR, including merged `2 dL` stages and `wl0 != 1.55` | Independent S-matrix solver (level 3) | 1e-10 complex |
| `lattice_coefficients` vs FFT of `lattice_response` | Independent evaluation | 1e-14 |
| `maxflat_halfband(N)`, `N = 1..6`: Daubechies polynomial, power complementarity, half-band mirror, derivatives of orders 1 to `2N - 1` vanishing at `omega = 0` | Analytic (level 2) | 1e-12 |
| `sqrt(2) B` for `N = 2, 3, 4` | Daubechies, *Ten Lectures on Wavelets* (SIAM 1992), Table 6.1; closed-form D4 and D6 (level 4) | 1e-14 |
| Order-2 interleaver: `K = ((2 + sqrt 3)/4, 3/4, 1/2)`, delays `(2, 1)`, phases `(0, pi)` | Derived by hand in the test (level 2) | 1e-14 |
| Interleaver through the netlist: crosstalk equals `x^N P(1 - x)`; the MZI's equals `sin^2(pi df / FSR)` | Analytic | 1e-13 |
| `spectral_factor` on random minimum-phase filters (real and complex, `n` up to 15) | Known factor | 1e-10 coefficients |
| `spectral_factor` with 2- and 4-fold unit-circle zeros, and the max-flat complements (up to 16-fold) | Known factor | 1e-12 reconstruction |
| `equiripple_lowpass`: complementarity; affine image of an independent `remez` run; Chebyshev alternation at `>= degree + 2` points | Analytic and independent run | 1e-12; equal ripple to remez's ~1 % |
| JAX gradient of `lattice_response` in couplings, phases and `omega` | Finite differences | rel 1e-6 |

### Limits

* **Ideal building blocks.** The synthesis is exact for lossless,
  wavelength-independent couplers and linear-dispersion waveguides, which is
  what `circuit.lattice_filter` simulates. Real directional couplers disperse,
  and their coupling drifts across a wide band. This degrades flat-top and
  stop-band performance in a way the synthesis does not model.
  `d.model(loss_db_cm=...)` shows the effect of propagation loss only. There
  is no coupler-dispersion compensation.
* **Process sensitivity.** Phase errors (heater calibration, `dL` variation)
  and coupler-ratio errors are not part of the synthesis. Study them with
  `photonix.analysis` sensitivity and yield tools on `d.model()`.
* **Large orders.** `maxflat_halfband` keeps power complementarity to about
  1e-13 through order 12, and about 2e-11 at order 20. It refuses orders that
  lose 1e-9. `spectral_factor` is root-based and is intended for degrees up
  to a few tens.
* **Equiripple accuracy.** `equiripple_lowpass` is equiripple only to the
  convergence of `scipy.signal.remez`, typically 0.1 to 1 % of the ripple.
  The half-band (interleaver) case is not forced to be exactly symmetric.
* **Couplers to geometry.** There is no lattice-specific coupler-to-geometry
  mapping. `coupling_to_gap` targets ring couplers, and near-unity couplings
  need long directional couplers or MMI-based tunable couplers.
