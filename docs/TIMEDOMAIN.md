# Time-domain simulation and signal integrity (`photonix.timedomain`)

`photonix.timedomain` simulates photonic circuits **sample by sample** in the
complex-envelope picture, and supplies the transmitter, receiver and
signal-integrity models needed to go from a bit pattern to an eye diagram and a
bit-error ratio. It complements the frequency-domain solver: the frequency
domain is exact and fast for linear, time-invariant circuits; the time domain is
needed for modulators and other time-varying elements, long bit sequences
(without the circular wrap-around of FFT methods), lasers and detectors inside a
link, and every eye/BER figure of merit.

Conventions (shared with the rest of photonix):

- an optical signal is the complex envelope `A(t)` about a carrier of wavelength
  `carrier_wl` (µm), with `|A|^2` the instantaneous power in **W**; the field is
  `Re[A(t) exp(+j w_c t)]`, so a device with transfer function `H(f)` acts on
  the envelope spectrum at offsets `nu = f - f_c`;
- times are in **seconds**, the step `dt` is fixed, and the envelope represents
  optical frequencies `f_c +/- 1/(2 dt)`;
- ports `o1..oN`, scattering matrices with rows = outgoing port, propagation
  `exp(-j beta L)`; electrical waveforms are real, in V or A.

## Quick start

```python
import photonix as px
from photonix.timedomain import (TimeDomainCircuit, AnalogFilter, cw_laser, drive_waveform, prbs,
                                 mzm, push_pull, photodiode, eye_diagram, mask_test)

baud, sps = 25e9, 32
dt = 1 / (baud * sps)
bits = prbs(9, 1022)
laser = cw_laser(1e-3, bits.size * sps, dt, linewidth_hz=1e6, rin_db_hz=-150, seed=1)
drive = drive_waveform(bits, baud=baud, samples_per_symbol=sps, amplitude_v=2.4, offset_v=-1.5,
                       rise_time_20_80_s=12e-12)
tx = mzm(laser, *push_pull(drive), v_pi=3.0, insertion_loss_db=4.0)

nl = px.circuit.Netlist()                       # any netlist of photonix models
nl.add("wg", "straight", length=2000.0, loss_db_cm=2.0)
nl.expose("in", ("wg", "o1")).expose("out", ("wg", "o2"))
rx = TimeDomainCircuit.from_netlist(nl, carrier_wl=1.55, dt=dt).simulate({"in": tx})["out"]

i = photodiode(rx, responsivity_a_w=0.9, bandwidth=AnalogFilter.bessel(4, 0.75 * baud), seed=2)
eye = eye_diagram(i, baud=baud, symbols=bits, skip_symbols=8)
print(eye.q_factor, eye.ber_gaussian, eye.extinction_ratio_db, mask_test(eye).margin)
eye.plot()                                      # matplotlib; never calls plt.show()
```

`examples/transceiver_link.py` runs a complete 25 Gb/s link (MZM with a lossy
traveling-wave electrode, waveguide and add-drop ring filter, noisy receiver,
eye and mask test) and checks the time-domain ring against the frequency-domain
solver.

## The circuit simulator

### Algorithm

Every element is linear in its incident waves and splits, at each step, into an
**instantaneous** part `D` and **strictly causal channels** `y` (`b = D a + y`,
`y_k` depending only on incident waves before step `k`). The connections give
`a = Gamma b + P u` with `u` the waves injected at external ports. The
instantaneous loop `(I - D Gamma) b = ...` is eliminated **once** at assembly, so
each step costs a gather from circular history buffers and a few small
matrix-vector products. Any element with at least one sample of delay breaks a
feedback loop, so rings, CROWs and loop mirrors are explicit and causal (the
approach of Photontorch, Laporte et al., *Sci. Rep.* 9, 5918, 2019). A loop
with no delay is solved algebraically; a *singular* one (a lossless zero-delay
loop) is rejected with an explanation. Time-varying elements re-solve only their
own small block per step (precomputed for the whole record). Channels that light
entering at the driven ports can never reach are pruned (exact).

Under JAX the stepping runs in `jax.lax.scan` inside `jax.jit` and is
differentiable with respect to element coefficients, inputs and control
waveforms; under NumPy it is a plain loop with identical results.

### Elements (`photonix.timedomain.elements`)

| Element | Implementation |
|---|---|
| `MemorylessTD(ports, S)` | Frequency-flat `b = S a` (couplers, MMIs, crossings, terminations at the carrier) |
| `DelayLineTD(tau_s, carrier_phase, loss_amp, beta2_l_s2=None, interp="lagrange3")` | Reciprocal delay `loss * exp(-j phase) * exp(-j 2 pi nu tau)`: Lagrange FIR (odd order, phase error `O((omega dt)^(N+1))`), Thiran all-pass (lossless at every frequency) or nearest sample; with `beta2_l_s2` a GVD FIR exact to `gvd_tol/2` in band |
| `FIRTD(ports, dt, taps)` | Causal FIR per port pair (generic sampled models) |
| `PoleResidueTD(ports, poles, residues, direct, delays)` | Rational model `d + sum r/(s - p)`, integrated exactly for piecewise-linear (`"foh"`, recursive convolution of Gustavsen & Semlyen) or held (`"zoh"`) input; the bridge for vector-fitted macromodels |
| `TimeVaryingTD(ports, fn, controls)` | Quasi-static `S(t)` from electrical control waveforms (any vectorized photonix model via `from_model`) |
| `CompositeTD` | A sub-network, flattened into the enclosing circuit |

### Converting photonix models (`to_td`)

`to_td(model, settings, carrier_wl=..., dt=...)` (used by
`TimeDomainCircuit.from_netlist` for every instance) applies, in order:

1. a model's own `td_model` attribute (element or factory);
2. waveguides (`straight`, `bend`, `bend_euler`, `taper`, `waveguide_route`,
   waveguide-type phase shifters) become a `DelayLineTD` with the exact carrier
   transmission and group delay `-d arg S21 / d omega` (`n_g L / c`);
3. frequency-flat devices become a `MemorylessTD`, after a flatness check;
4. `all_pass_ring` and `add_drop_ring` become their physical structure (couplers
   plus ring delays), exactly the circuit their closed forms describe;
5. anything else is sampled over `f_c +/- 1/(2 dt)` and fitted with a causal
   FIR: pure-delay pairs get a fractional-delay interpolator, the rest a
   Planck-tapered inverse DFT (exact to `fir_tol/2` for `|nu| <= band/(2 dt)`,
   smooth roll-off to zero at Nyquist). A passive model is scaled to a
   **strictly** passive FIR (largest singular value ≤ 1 over the whole band), so
   lossless feedback loops stay stable over arbitrarily long runs. Non-causal
   responses (e.g. a zero-phase Gaussian passband) get reported latency.

`to_td_time_varying(model, settings, control_params={"voltage": "v"})` turns
a modulator model into a quasi-static element (passive delay plus a lumped
modulation at its output). Circuits from `from_netlist` accept
`overrides={inst: {param: value}}` in `simulate`, which may be JAX tracers: the
static layout (integer delays, tap counts) is kept, so time-domain figures of
merit can be differentiated with respect to circuit parameters.

Diagnostics: a warning when a delay is shorter than one step (its
interpolation is off-centre), when an input has more than 0.1 % of its energy
outside the exact band of a sampled-FIR element, when an FIR fit exceeds its
tolerance or `max_taps`; an error for singular zero-delay loops.
`TimeDomainCircuit.discrete_response(nu)` returns the response the discretized
network actually implements, so the discretization error is directly visible.

## Transceiver and signal-integrity library

| Module | Contents |
|---|---|
| `sources` | `prbs(7/9/11/15/23/31)` (ITU-T O.150 Fibonacci LFSR `x^n + x^m + 1`, block-doubled so PRBS31 runs at NumPy speed), `symbols_to_levels` (NRZ, Gray PAM4), `drive_waveform` (closed-form Gaussian or raised-cosine edges with a 20-80 % rise time and Gaussian jitter), `cw_laser` (Wiener phase noise = Lorentzian linewidth, white RIN), `LaserParams` + `rate_equation_laser` (single-mode rate equations with gain compression, spontaneous emission, Henry phase coupling and Langevin noise; RK4 in `lax.scan`), `LaserParams.noise_spectra` (linearized Langevin RIN and FM-noise spectra) |
| `modulators` | `mzm` (dual drive, push-pull via `push_pull`, finite extinction, chirp, optional `TravelingWaveElectrode` and drive filter), `TravelingWaveElectrode` (velocity mismatch, causal skin-effect loss, source/load reflections), `eam` (absorption vs voltage, Henry chirp), `RingModulator` / `ring_modulator` (temporal CMT with photon-lifetime dynamics, pn-junction physics of `components.pn_phase_shifter`, RC-limited drive, Sacher-Poon small-signal response) |
| `filters` | `AnalogFilter` (pole-residue; RC, Butterworth, Bessel-Thomson prototypes; exact FOH/ZOH discretization; noise bandwidth, rise time), `lowpass`, `linear_recurrence` |
| `receivers` | `photodiode` (responsivity, dark current, shot / thermal / RIN noise, receiver filter), `tia`, `recover_clock` |
| `metrics` | `eye_diagram` → `EyeDiagram` (levels, sigmas, Q, Gaussian BER for NRZ and Gray PAM4, height, width, crossing %, ER, OMA, decisions, `plot`), `ber_from_q`, `q_from_ber`, `ber_count` (Wilson interval), `receiver_sensitivity` (feeds `analysis.LinkBudget`), `power_penalty_db`, `extinction_ratio_penalty_db`, `EyeMask` / `mask_test` |
| `viz.plot_eye` | Eye-diagram plot (Agg-safe) |

Receiver noise is generated before the filter as band-limited white noise and
filtered **exactly within the simulation band** (DFT multiplication by `H(f)`),
while the deterministic signal goes through the exact FOH discretization. This
gives output variances `2 q I B_n`, `4 k T B_n / R_L`, `RIN (R P)^2 B_n` and
`i_n^2 B_n` with the filter's true noise bandwidth (less the part beyond
`f_s/2`, `(2/pi) f_3dB/(f_s/2)` for a single pole).

## Validation evidence

Evidence levels follow [`VALIDATION.md`](VALIDATION.md). All checks run in CI
(`tests/test_timedomain.py`, `tests/test_timedomain_si.py`).

| Check | Reference | Result / tolerance | Level |
|---|---|---|---|
| CW at a 23 GHz offset through all-pass ring, add-drop ring, 3-ring CROW, 3-stage lattice | frequency-domain `circuit_from_netlist` | `\|S_td - S_fd\| < 1e-6` after 2 ns | 3 |
| dt sweep with Lagrange-3 interpolation (ring) | — | error slope 4.0 ± 0.3 | 1 |
| Gaussian pulses through CROW, lattice, ring | FFT propagation with `S(f)` | < 2e-3 of peak | 3 |
| Simulation vs `discrete_response`; JAX vs NumPy runner | implementation identities | 1e-10; 1e-14 | — |
| Integer delays (Lagrange, Thiran, nearest) | exact shift | 1e-13 | 2 |
| 2 mm straight, 1 ps pulse | `n_g L / c` = 28.02 ps (`analysis.pulse_response` doctest) | 1e-4 ps | 2 |
| GVD broadening | `T1 = T0 sqrt(1 + (beta2 z / T0^2)^2)` (Agrawal) | 1e-4 | 2 |
| Lossless CROW energy; reciprocity; 50:50 Sagnac mirror | exact invariants | 1e-9; 1e-13; reflect 1 - 1e-9 | 2 |
| `jax.grad` of drop-port energy w.r.t. a coupling through `lax.scan` | finite differences | 1e-6 relative | — |
| 2^15-bit PRBS, 16 samples/bit, 4-ring CROW | throughput | < 5 s compiled (CPU) | — |
| Generic FIR fits (grating coupler, Bragg grating) | sampled model | in-band error < 1e-4; strictly passive | 2 |
| PRBS 7/9/11/15 | period `2^n - 1`, `2^(n-1)` ones, run-length counts `2^(n-k-1)`, two-valued autocorrelation; PRBS23/31 recurrence and naive LFSR | exact | 2 |
| Laser L-I curve, threshold; f_R; small-signal response | `I_th = qV N_th/(eta_i tau_n)`, slope `eta_d h nu / q`; `f_R` formula; linearized rate equations | 1e-4, 1e-3; peak within 2 %; TD modulation within 0.2 % | 2 / 3 |
| Laser FM noise, RIN, Henry factor | linearized Langevin theory (DC limit = `(1 + alpha_H^2) dnu_ST / pi` to 1e-3) | ST level 3 %, FM noise 8 %, Henry ratio ≈ 17 within 10 %, RIN 5-12 %, RO peak 2 % (≈4 sigma) | 3 |
| CW-laser linewidth and RIN | Lorentzian FWHM from the field autocorrelation; RIN PSD | 5 %; 0.2 dB | 2 |
| Shot, thermal, TIA noise | `2 q I B_n`, `4 k T B_n / R`, `i_n^2 B_n`; PSD `N0 \|H\|^2` | within 4 sigma of the exact (Bartlett) standard error; PSD 3 % | 2 |
| Thermal-noise-limited sensitivity at Q = 7; counted BER at Q ≈ 3; ER penalty | `P = Q sigma_T / R` (Agrawal Ch. 4); `erfc(Q/sqrt 2)/2` in the Wilson CI; `10 log10((1+r)/(1-r))` | 0.1 dB; inside CI; 0.05 dB | 2 |
| RC rise time; Butterworth/Bessel f3dB and noise bandwidth; FOH step response | `ln 9 / (2 pi f3dB)` = 0.35/f3dB; closed forms | 1e-9 / 2e-3 simulated; 1e-6 | 2 |
| MZM transfer, ER, chirp | `cos^2(pi V / 2 V_pi)`; push-pull imbalance | 1e-12; 2 % | 2 |
| Traveling-wave electrode | `\|H\| = 1/sqrt 2` at `u = 1.3916`; direct quadrature of the standing wave; causal skin-effect tail `B / sqrt(pi t)` | 1 %; 1e-6; 3 % | 2 |
| Ring modulator | `all_pass_ring` (high Q); photon-lifetime relaxation; Sacher-Poon small-signal at 2/15/40 GHz | 0.01; 1e-3; 1 % | 2 / 3 |
| Eyes | ideal eye height = OMA, ER = drive ER, width = 1 UI; exact RC-channel ISI closure; PAM4 levels and Gray BER; analytic mask margin; mask hits vs `matplotlib.path` | 1e-12 / 7 % width; 1e-12; 1e-9 | 2 |
| `jax.grad` through MZM, Bessel drive filter and ring modulator | finite differences | 1e-6 relative | — |

No model in this package is calibrated against measurements (level 5); the
laser defaults are illustrative textbook values.

## Limits

- **Quasi-static modulation.** `TimeVaryingTD` responds instantly to its
  controls; `to_td_time_varying` lumps the modulation at the output of the
  device's delay (velocity-matched, lumped picture). Use `RingModulator` (CMT)
  or filter the drive (`AnalogFilter`, `TravelingWaveElectrode`) for dynamics.
- **Sampling.** The envelope band is `+/- 1/(2 dt)`: ring FSRs and signal
  bandwidths must fit inside it, and delays should be at least one step.
  Lagrange interpolation has phase error `O((omega dt)^4)`; sampled FIR fits are
  exact only for `|nu| <= band/(2 dt)` (default half the Nyquist band).
- **Generic FIR conversion** of short dispersive devices needs a two-sided
  kernel and therefore latency: a 1 mm `pn_phase_shifter` at `dt = 0.5 ps` gets
  2.5 ps of reported latency (its plasma dispersion makes it not quite a pure
  delay). High-Q devices exceed `max_taps`: model them structurally or as
  `PoleResidueTD`. No vector fitting is built in yet.
- **No nonlinear optics** (Kerr, TPA, free-carrier dynamics) and no thermal
  transients other than through user-supplied control waveforms.
- **Rate-equation laser**: single mode, no spatial hole burning or thermal
  rollover; the Langevin forces are integrated by Euler-Maruyama (weak order 1),
  which biases the delicate low-frequency RIN by a few percent at the default
  substeps; with large RO fluctuations the (correct) nonlinear simulation departs
  from the linear-noise theory near `f_R` and `2 f_R`.
- **Traveling-wave electrode**: frequency-independent `Z0`; with skin-effect
  loss the `t^-3/2` tail beyond the FFT grid is folded back (error
  `skin_tail(T)`, ~7e-4 for 0.8 dB/cm/sqrt(GHz), 4 mm, 64 ns grid).
- **Ring modulator**: CMT is accurate in the high-Q limit; the FOH integration
  is exact for piecewise-linear envelopes, so put the envelope carrier at the
  laser frequency; the small-signal closed form neglects loss modulation.
- **Receivers**: Gaussian white noise only (no APD excess noise, no 1/f); noise
  beyond `f_s/2` is not simulated.
- **Eye metrics**: Gaussian BER from level statistics (with known symbols the
  ISI is part of sigma); choosing the best of `sps` sampling phases biases Q
  upward when the per-phase noise is independent (fix `sampling_index` for an
  unbiased estimate). TDECQ is not implemented, and the mask margin uses
  photonix's definition (mask scaled about the eye centre at the Q-optimal
  sampling instant), not the exact IEEE 802.3 procedure.
- **Scale**: the per-step maps are dense over terminals and channels, which
  suits circuits of up to a few hundred elements.

## References

- F. Laporte, J. Dambre, P. Bienstman, *Sci. Rep.* 9, 5918 (2019) — Photontorch.
- T. I. Laakso et al., *IEEE Signal Process. Mag.* 13(1), 30 (1996) — fractional delay.
- B. Gustavsen, A. Semlyen, *IEEE Trans. Power Del.* 13(2), 605 (1998) — recursive convolution.
- ITU-T O.150 (1996) — PRBS polynomials.
- G. P. Agrawal, *Fiber-Optic Communication Systems*, 4th ed. (Wiley, 2010) — noise, sensitivity, penalties, laser noise.
- L. A. Coldren, S. W. Corzine, M. L. Mašanović, *Diode Lasers and Photonic Integrated Circuits*, 2nd ed. (Wiley, 2012).
- C. H. Henry, *IEEE J. Quantum Electron.* 18(2), 259 (1982) — linewidth enhancement.
- W. D. Sacher, J. K. S. Poon, *Opt. Express* 16(20), 15741 (2008) — ring-modulator dynamics.
- J. Witzens, *Proc. IEEE* 106(12), 2158 (2018); G. Ghione, *Semiconductor Devices for High-Speed Optoelectronics* (CUP, 2009) — traveling-wave electrodes.
- W. E. Thomson, *Proc. IEE* 96, 487 (1949) — Bessel filters.
