# Rational macromodels (`photonix.macromodel`)

`photonix.macromodel` turns sampled S-parameters (from a field solver, a
measurement or a Touchstone file) into a **delayed rational model** that is
stable, can be certified passive and causal, evaluates exactly between samples,
drops into circuits as an ordinary component model, and has a state-space
realization for time-domain simulation.

Before this package the only way to use sampled data in a circuit was
`SParameterDataset.__call__`, which interpolates real and imaginary parts
linearly. For a delayed response that loses amplitude between samples: a
lossless 200 µm straight sampled every 0.5 nm and re-queried on a 10× finer grid
reads `min |S21| = 0.83`, a fictitious 1.6 dB loss. The rational model gives
`|S21| = 1` to 1e-9 (`tests/test_macromodel.py::test_rational_model_fixes_the_interpolation_defect`).

## The model

Optical S-parameters are complex envelopes about a carrier
`omega_c = 2 pi carrier_hz`. With `Omega = omega - omega_c` every entry is

```
S[out, in](Omega) = exp(-1j Omega tau[out, in]) * ( sum_k R_k[out, in] / (1j Omega - p_k) + D[out, in] )
```

with one set of poles `p_k` (rad/s, `Re p_k < 0`) shared by all entries. Signs
follow Photonix (`exp(+1j omega t)`, propagation `exp(-1j beta L)`): a causal,
stable impulse response has left-half-plane poles and a pure delay is
`exp(-1j Omega tau)`. The constant carrier phase of a delay is absorbed into
`R_k` and `D`. Because the baseband impulse response is complex, poles and
residues are **not** paired with their conjugates (Ye et al. 2019); a real RF
system (`carrier_hz = 0`, conjugate pairs, real `D`) is supported as a special
case with `vector_fit_frequency(..., real=True)`.

```python
import numpy as np, photonix as px
from photonix import macromodel as mm

ds = px.SParameterDataset.load_touchstone("device.s4p")
print(mm.causality_report(ds.wavelengths, ds.s, ports=ds.ports).causal)   # check raw data first
model = ds.to_rational()                         # = mm.fit_dataset(ds); delays + poles chosen automatically
print(model.n_poles, model.metadata["relative_rms_error"])
if not mm.passivity_report(model).passive:       # Hamiltonian test over all frequencies
    model = mm.enforce_passivity(model)
s = model(wl=np.linspace(1.50, 1.60, 5001))      # SDict, jit/grad-compatible
ss = model.to_statespace()                       # A, B, C, D (+ delays) for time-domain tools
y = model.simulate(u, dt)                        # exact delayed response to sampled input envelopes
```

## API

| Name | Purpose |
|---|---|
| `RationalModel` | Frozen container (`ports`, `carrier_hz`, `poles`, `residues[k, out, in]`, `d`, `delays`, `metadata`). `__call__(*, wl, poles=, residues=, d=, delays=)` is a Photonix model built from `xp` arithmetic, so the overrides are differentiable parameters. `response(freq_hz)`, `rms_error`, `to_dataset`, `impulse_response`, `port_delays`, `is_real_system`, `save_npz` / `load_npz` (pickle-free, versioned). |
| `RationalModel.to_statespace(minimal=True, real=False)` | Gilbert realization `A = blockdiag(p_k I_{r_k})` from rank-revealing SVDs of the residues; `real=True` gives real `A, B, C, D` for a real system. |
| `RationalModel.discretize(dt, "zoh" \| "foh")` | Exact zero/first-order-hold discrete equivalent (`DiscreteStateSpace` with `simulate`, `response`). |
| `RationalModel.simulate(u, dt)` | Exact output for a piecewise-linear input envelope, including fractional per-path delays (each pole is propagated in closed form). |
| `vector_fit(wavelengths, S, ...)` | Complex-baseband fit of `(N, n, n)` data (or an SDict) sampled over wavelength (µm). |
| `vector_fit_frequency(freq_hz, S, ..., real=False)` | The same over frequency; `real=True` for RF data. |
| `fit_dataset(ds, **kw)`, `SParameterDataset.to_rational(**kw)` | Fit a dataset, keeping its ports and metadata. |
| `estimate_delays(freq_hz, S)` | Conservative per-entry arrival delays. |
| `passivity_report(model)` | Hamiltonian crossings, violation bands, peaks, `max_sigma`, `certified`. |
| `enforce_passivity(model, margin=1e-6, band_hz=None)` | Minimum in-band-energy residue perturbation until the Hamiltonian is clean. |
| `hamiltonian_matrix(model)` | The scattering Hamiltonian (rad/s). |
| `causality_report(wavelengths, S)` | Generalized Kramers-Kronig test with truncation and discretization bounds. |

Main fitting options: `n_poles` (`"auto"` grows the order until the RMS error
over all `n**2` entries is below `target_rms * max|S|`, default 1e-4, or stops
improving), `delays` (`"auto"`, `"entry"`, `"none"` or explicit seconds),
`weighting` (`"uniform"` or `"inverse_magnitude"`), `symmetric` (exactly
reciprocal model; auto-detected), `relaxed`, `n_iter`, `tol`, `max_poles`.

## Algorithms

**Delay extraction.** A 1 mm guide winds its phase about 175 times across
1.50-1.60 µm; without delay removal it needs hundreds of poles. The removable
delay is the *arrival* time of each path. A delay-dominated entry (a
least-squares linear phase leaves less than 0.01 rad RMS of residual phase:
rounding error, measurement noise, weak dispersion) gets that least-squares
delay, exact for a dispersion-free guide; otherwise the arrival is the first
time the Hann-windowed, band-limited impulse response reaches 1 % of its peak,
minus one resolution cell `1/B`. Entries whose band-limited impulse response is
not concentrated in time (envelope peak below 10× its median, as for the noise
floor of a structurally zero entry in measured data; white noise gives about 4,
the library models 50 to 1e8) have no arrival and do not constrain the delays.
The minimum group delay is *not* a safe estimate: an all-pass ring's is positive
although its impulse response starts at `t = 0`, and removing it leaves a
non-causal core (with that estimate the ring needed 12 poles instead of 8). With
`delays="auto"` the delays are restricted to per-port reference-plane delays
`tau_ij = a_i + b_j` (`a = b` for reciprocal data), chosen by a two-stage linear
program to remove as much delay as possible without exceeding any significant
entry's estimate. Those delays multiply `S` by diagonal unitary matrices, so
they leave every singular value unchanged and the Hamiltonian passivity test of
the delay-free core is exact for the whole model. `delays="entry"` removes each
path's delay independently, which can save poles for multi-path devices, but
then passivity can only be checked on a frequency sweep (`certified=False`).

**Vector fitting.** Starting poles `-alpha_k + 1j beta_k` span the band
(`alpha = spacing / 100`). Relaxed pole relocation (Gustavsen 2006) solves for
the scaling function `sigma(s) = sum c~_k/(s - a_k) + d~` with
`Re sum_n sigma(s_n) = N` imposed exactly, using the fast per-entry QR
reduction of Deschrijver et al. (2008) (cost `O(N P^2 n^2)` per iteration). New
poles are `eig(diag(a) - 1 c~^T/d~)` (a real block form keeps exact conjugate
pairs for RF), unstable poles are reflected, and the residues and `D` follow from
linear least squares. Frequencies are normalized to the band half-width
internally. Iteration stops on a pole-change tolerance or when the error
stagnates; the best iterate is kept.

**Passivity.** For power-normalized waves a model is passive exactly when
`sigma_max(S(1j Omega)) <= 1` at every real frequency, including outside the
fitted band. The imaginary eigenvalues of the scattering Hamiltonian
`M = [[A - B R^-1 D^H C, -B R^-1 B^H], [C^H Q^-1 C, -A^H + C^H D R^-1 B^H]]`
(`R = D^H D - I`, `Q = D D^H - I`, valid for complex realizations) are the
frequencies where a singular value equals one (Grivet-Talocia 2004). Candidate
eigenvalues are accepted generously and each interval between them is
classified by evaluation, so near-tangencies cannot hide a band. Enforcement
first clips `sigma_max(D)` to `1 - margin` (re-fitting the residues in band to
compensate), then repeatedly linearizes `delta sigma = Re(u^H delta S v)` at every
local maximum above one and applies the perturbation of the residues with
minimum in-band energy `integral_band ||delta S||_F^2` (an exact closed-form Gram
matrix, no sampling) that brings each peak to `1 - margin`. Reciprocity and
real-system symmetry are preserved when present; residues that are structurally
zero stay zero.

**Causality.** Generalized dispersion relations with `n` subtractions
(Triverio & Grivet-Talocia 2008) reconstruct each entry from its own in-band
samples. With `|S| <= 1` out of band the truncation error has a rigorous bound;
the quadrature error (trapezoidal-linear principal value, exact per segment) is
estimated by Richardson comparison with every other sample, and additive
measurement noise (not Kramers-Kronig consistent: noise alone fails the test)
is propagated through the same linear reconstruction with 16 Monte-Carlo
white-noise realizations, scaled by a per-entry noise level that is estimated
from fourth differences (a robust median) unless `noise_std` is given. `n_support`
subtraction sets spread over the band localize the test, and a sample is flagged
if any set exceeds its bound. The test can prove non-causality (a time
advance, a wrong reference-plane sign, corrupted data), not causality; it is
blind at the band edges, where the bound diverges.

## Validation evidence

All rows are executable tests in `tests/test_macromodel.py`.

| Check | Reference | Result | Level |
|---|---|---|---|
| Random stable complex poles (6 poles, 3 ports, 400 samples) | exact rational data | poles to 2e-15 relative (test 1e-9), RMS 7e-15; `auto` order stops at 6 | 2 |
| Real RF system with 50 ps delay | exact data | conjugate poles to 1e-9, RMS < 1e-12; real realization equals the model | 2 |
| `straight(length=1000)` over 1.50-1.60 µm | `tau = n_g L / c = 14.0097 ps` | `tau` to 2e-14 relative, 0 poles, RMS 7e-12 | 2 |
| 1 cm straight, 1.54-1.56 µm | 140.10 ps | `tau` to 6e-14, 0 poles, RMS 3e-11 | 2 |
| 200 µm line sampled at 0.5 nm, re-queried 10× finer | lossless | `|S21|` = 1 to 1e-9 (linear interpolation: 0.83) | 2 |
| 1 mm straight with 1e-3 complex noise on all entries (noise-only reflections) | `n_g L / c`; noiseless response | delay to 1e-7 (test 1e-5); reflections ignored for the delays; `auto` order stops at the noise floor with 0 poles; error vs the noiseless truth 3e-5 | 2 |
| `add_drop_ring` with 1e-3 noise | noiseless response | 8 poles chosen at the noise floor; error vs truth 1.9e-4 (test 5e-4) | 3 |
| One resonance of `all_pass_ring(coupling=0.01)` | `analysis.find_resonances` | one pole, `Q_L = omega_0 / (2 |Re p|)` within 0.07 % (test 1 %), centre within FWHM/100 | 3 |
| `all_pass_ring` over 5 FSRs | resonances | RMS 2.5e-2, 3.9e-3, 4.0e-4, 2.7e-5, 5.5e-8 for 5, 6, 7, 8, 10 poles; at 8 poles every pole within 3e-4 FWHM of its resonance | 3 |
| `all_pass_ring`, `add_drop_ring`, uniform and apodized `bragg_grating` (40 nm, 2001 samples, all defaults) | the analytic models on a 4× denser offset grid | 8, 8, 20, 20 poles; RMS 2.5e-6, 1.3e-5, 7.8e-7, 2.5e-7 between samples | 3 |
| Add-drop fit with residues × 1.02 | dense σ_max sweep (1e5 points) | reported violation band equals the sweep sample-for-sample; enforced model passive on a 20× grid, no imaginary Hamiltonian eigenvalues, in-band change 2.0× the pointwise lower bound `RMS(σ_max - 1)+`, and closer to the original passive fit (3.06e-3 → 2.12e-3), as a projection onto the convex passive set must be | 2 |
| Lossless Bragg fit with 10 nm guard bands, enforced over the band of interest | analytic model | passive, in-band RMS 1.4e-6 | 3 |
| Causality of 8 library models (straight, rings, Bragg, coupler, MMI, MZI) | causal by construction | all pass; the same data × `exp(+1j Omega 5 ps)` all fail (a 200 µm line's 2.8 ps < 5 ps) | 2 |
| Causal vs anticausal analytic pole | analytic | pass / fail | 2 |
| Add-drop ring with 1e-4, 1e-3, 1e-2 complex noise | causal by construction | passes with estimated or known noise level, fails with `noise_std=0` (noise alone is acausal); a 5 ps advance is still detected at every level | 2 |
| `simulate` of a fitted ring behind a 2 mm bus, 5 ps pulse | `analysis.pulse_response` (FFT of the sampled data) | output power within 1.9e-4 of peak (test 1e-3) | 3 |
| FOH-discretized add-drop fit driven by the band-limited (Hann) impulse | FFT of the model's own response on a uniform grid; `analysis.impulse_response` of the data | 7e-7 / 8e-6 of peak, through / drop (test 1e-4); 1.7e-4 / 5.3e-4 (test 1e-3), which is the analysis routine's own wavelength-to-frequency resampling error | 2 / 3 |
| ZOH step response; FOH recursion vs `simulate`; integer delay shift | analytic | 1e-12; 4e-14; exact | 2 |
| `jax.grad` of drop power w.r.t. residues | central differences | 1e-9 relative (test 1e-6) | 2 |
| Touchstone `.s4p` round trip, then fit | written samples | RMS < 1e-4 max\|S\| | 2 |

## Limits

- A fit is qualified only inside its band. Outside it the model is a rational
  extrapolation. Lossless devices (Bragg gratings, couplers) fitted to 1e-6
  typically violate passivity by a few per cent *just outside* the band, where
  the truncated series of sidelobe poles overshoots; enforcing that directly
  costs in-band accuracy near the edges (3.9e-3 RMS for the 40 nm Bragg fit).
  **Fit guard bands** a little wider than the band of interest and pass
  `band_hz` (the band of interest) to `enforce_passivity`: the same grating then
  ends at 1.4e-6.
- With `delays="entry"` passivity is checked and enforced on a dense sweep of the
  fitted band (extended by 50 %) and every pole's neighbourhood; this is not a
  certificate (`certified=False`, with a warning).
- `estimate_delays` is deliberately conservative: it removes less than the true
  arrival when the band is narrow (time resolution `1/B`), leaving the rest to
  poles. Long delays require a sweep that resolves the phase (less than `pi` per
  sample); otherwise a warning is issued and delays should be passed explicitly.
- `n_poles="auto"` grows the order on a fixed schedule (0, 1, 2, 3, 4, 6, 8, ...)
  and fits each order from scratch; very large models (many ports and hundreds
  of poles) are slow (`O(N P^2 n^2)` per iteration) and no pole-adding or
  model-order reduction is attempted. There is no per-entry pole option: common
  poles are always used, which a compact state-space realization requires.
- Passivity enforcement perturbs residues only (poles and delays are fixed) and
  uses equality constraints at the violation peaks; it is first-order and
  iterative, not a convex optimum, but in the tests it lands within a factor of
  two of the pointwise lower bound. Clipping `sigma_max(D)` can fill structural
  zeros of `D`.
- The causality test assumes `|S| <= 1` out of band (passive S-parameters; set
  `out_of_band_max` otherwise), cannot detect violations at the band edges, and
  loses sensitivity (never gaining false alarms) on under-resolved sweeps. Its
  noise model is additive white noise; correlated or systematic errors (drift,
  calibration ripple) are not modelled and can be flagged as non-causal.
- Noisy measured data: the fitter has no noise model. When `target_rms` is
  below the noise floor the order search stops once the error stagnates
  (at least order 16 and five orders without a 10 % gain) and keeps the
  smallest order within 5 % of the best error, with a warning; setting
  `target_rms` near the noise floor is faster. Stagnation at low orders on
  strongly resonant data is not mistaken for a noise floor below order 16.
- Performance: the kernels are many small dense LAPACK calls. On an
  oversubscribed machine a multi-threaded BLAS can make them two orders of
  magnitude slower (measured here); set `OPENBLAS_NUM_THREADS=1` (or the MKL
  equivalent) for such runs.

## References

- B. Gustavsen and A. Semlyen, "Rational approximation of frequency domain
  responses by vector fitting," IEEE Trans. Power Delivery 14(3), 1052 (1999).
- B. Gustavsen, "Improving the pole relocating properties of vector fitting,"
  IEEE Trans. Power Delivery 21(3), 1587 (2006).
- D. Deschrijver, M. Mrozowski, T. Dhaene, D. De Zutter, "Macromodeling of
  multiport systems using a fast implementation of the vector fitting method,"
  IEEE Microw. Wireless Compon. Lett. 18(6), 383 (2008).
- Y. Ye, D. Spina, D. Deschrijver, W. Bogaerts, T. Dhaene, "Time-domain compact
  macromodeling of linear photonic circuits via complex vector fitting,"
  Photonics Research 7(7), 771 (2019).
- S. Grivet-Talocia, "Passivity enforcement via perturbation of Hamiltonian
  matrices," IEEE Trans. Circuits Syst. I 51(9), 1755 (2004).
- B. Gustavsen, "Fast passivity enforcement for S-parameter models by
  perturbation of residue matrix eigenvalues," IEEE Trans. Adv. Packag. 33(1),
  257 (2010).
- P. Triverio and S. Grivet-Talocia, "Robust causality characterization via
  generalized dispersion relations," IEEE Trans. Adv. Packag. 31(3), 579 (2008).
- S. Grivet-Talocia and B. Gustavsen, *Passive Macromodeling* (Wiley, 2016).
