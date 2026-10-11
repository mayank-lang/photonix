# Component library reference

Every model is a pure function `f(*, wl=1.55, **params) -> SDict`, differentiable
and `jit`-able, with defaults for every parameter. Lengths and wavelengths are in
µm. This page lists what each model computes, the evidence behind it (levels from
[`VALIDATION.md`](VALIDATION.md)), and where it stops being trustworthy.

All models are registered in `photonix.components.MODELS`, which is also the
registry netlist extraction uses; `tests/test_components_extended.py` checks that
every entry evaluates with its defaults, is passive (largest singular value ≤ 1),
and also runs under the NumPy fallback backend.

## Waveguides

| Model | Ports | Physics | Evidence | Limits |
|---|---|---|---|---|
| `straight` | o1, o2 | `exp(-αL) exp(-jβL)` with linear dispersion (exact `ng` at `wl0`) or a callable `neff(wl)` | 2 | Single mode, no reflection |
| `bend` | o1, o2 | Arc length `R·θ` plus lumped `excess_loss_db` (default 0) | 2 | Loss is not computed; use `bend_from_solver` |
| `bend_from_solver` | o1, o2 | Full-vector conformal-map bend solve for loss and index | 1 | Equivalent-index conformal map; no junction loss |
| `bend_euler` | o1, o2 | Clothoid path length `R·θ·(1+p)` (`R` = minimum radius) | 2 | Loss lumped, as for `bend` |
| `taper` | o1, o2 | Adiabatic phase `∫ n_eff(w(z)) dz` by 5-point Gauss–Legendre for callable `neff(wl, w)` | 2 | No mode conversion/reflection; see `em.components.taper` (EME) |
| `waveguide_route` (`route`) | o1, o2 | Routed path: straight of the drawn length + `n_bends · bend_loss_db` | 2 | Bend loss is an input |

## Couplers and splitters

| Model | Ports | Physics | Evidence | Limits |
|---|---|---|---|---|
| `directional_coupler` / `coupler` | o1,o2 → o4,o3 | Ideal lossless 2×2, bar `√(1−κ²)`, cross `−j κ` | 2 | Wavelength-independent unless `coupling` is an array |
| `evanescent_coupler` | o1,o2 → o4,o3 | Coupled-mode theory, `sin²(π Δn(g,λ) L_eff / λ)`; curved gaps add `√(2πR/(m γ))` | 1 (calibration), 2 (unitarity, cross-over length) | Default `Δn(g,λ)` is a fit to Photonix full-vector solves of 500×220 nm Si strips in SiO₂ (gaps 0.10–0.40 µm, 1.50–1.60 µm; fit ≤1.3 %, grid spread ±1.3 %, curved-gap closed form ≤2 %). Pass `splitting=` for any other cross-section. Warns outside the calibrated range |
| `mmi1x2`, `y_branch` | o1 → o2, o3 | Ideal symmetric 1×2 | 2 | Sub-unitary by construction (odd mode dissipated); see `em.components.mmi1x2` |
| `mmi2x2` | o1,o2 → o4,o3 | Ideal 50/50 | 2 | — |
| `mmi_nxn` | o1..oN → o2N..oN+1 | General-interference self-imaging phases (Bachmann et al., 1994) | 2 (unitarity for N = 2–8; N = 2 equals `mmi2x2`) | Ideal imaging; `imbalance_db` is a phenomenological taper |
| `crossing` | o1 W, o2 N, o3 E, o4 S | Through loss, quadrature crosstalk, optional reflection | 2 (exact passivity bound (\|t\| + \|r\|)² + 4\|x\|² ≤ 1, enforced) | Wavelength-independent |

## Resonators and interferometers

| Model | Ports | Physics | Evidence | Limits |
|---|---|---|---|---|
| `all_pass_ring` | o1 → o2 | Closed-form feedback response, stable at weak coupling | 2 | Coupling is an input |
| `add_drop_ring` | o1 in, o2 thru, o3 drop, o4 add | Closed form, both circulation senses | 2 | Coupling is an input |
| `ring_single` | o1 → o2 | `H = t + k²A/(1 − tA)` with the `evanescent_coupler` of the drawn gap/radius/racetrack length | 2 (equals the circuit-assembled network to 1e-12) | Inherits the coupler calibration |
| `ring_double` | o1..o4 as `add_drop_ring` | Two `evanescent_coupler`s and two half rings, solved by the circuit engine | 3 (matches `add_drop_ring` with the same coupling) | As above |
| `mzi` | o1 → o2, o3 | Closed-form two-coupler interferometer | 2 | Common arm length not modelled; use `circuit.mzi` |

## Filters

| Model | Ports | Physics | Evidence | Limits |
|---|---|---|---|---|
| `bragg_grating` | o1, o2 (with reflections) | Exact CMT section transfer matrices: uniform, apodized (`gaussian`, `raised_cosine`) and phase-shifted gratings | 2 (peak `tanh²(κL)`; null-to-null width `λ²/(π n_g) √(κ² + (π/L)²)`; quarter-wave shift transmits fully at Bragg; lossless unitarity) | First-order Bragg, single mode; `dn` → `κ` uses the 50 %-duty Fourier coefficient — take `dn` from a mode solve of the corrugation |
| `awg` (+ `awg_design`) | o1 → o2..oN+1 | Phased array with Gaussian illumination; output modes Löwdin-orthonormalized so Σ\|S\|² ≤ 1 exactly; design from channel plan (order, ΔL, FSR, envelope); `delta_neff`, insertion loss, non-uniformity, arm loss, seeded arm phase errors | 2 (peaks on the grid, 3-dB width, FSR periodicity, index shift `δn·λ/n_g`) | Paraxial star couplers; matched output modes; no polarization or higher diffraction-order cross-talk |

## Active and opto-electronic

| Model | Ports | Physics | Evidence | Limits |
|---|---|---|---|---|
| `phase_shifter` | o1, o2 | `phase0 + 2π/λ (neff + dn/dV·V) L` | 2 | Linear electro-optic response |
| `heater_phase_shifter` | o1, o2 | `π P / P_π · (λ₀/λ)` plus propagation | 2 | No thermal time constant or crosstalk |
| `thermo_optic_phase_shifter` | o1, o2 | `dn_eff/dT · ΔT` (default bulk Si 1.86e-4 /K, Komma 2012) | 2 | Modal coefficient should come from a mode solve or measurement |
| `pn_phase_shifter` | o1, o2 | Abrupt-junction depletion widths and Soref–Bennett plasma dispersion (Drude λ² scaling), first-order uniform-lateral-field overlap | 2 (formula level); `V_π L` ≈ 2 V·cm at 5e17 cm⁻³, in the range reported for lateral junctions | Depletion mode only (forward bias clamped); no RC/velocity limits; overlap is an input |
| `photodetector` (+ `responsivity`, `photocurrent`) | o1 | Optical termination; `η q λ / (h c)` | 2 | Current is computed outside the S-matrix |

## Input/output

| Model | Ports | Physics | Evidence | Limits |
|---|---|---|---|---|
| `grating_coupler` | o1 wg, o2 fibre | Gaussian spectral response (FWHM, peak loss) | 0 (phenomenological) | No angle/polarization dependence |
| `edge_coupler` (+ `gaussian_overlap`) | o1 chip, o2 fibre | Closed-form elliptical Gaussian overlap with offsets and tilt; optional Fresnel facet | 2 (matches the numerical overlap integral to 1e-8) | Gaussian modes; converter loss is an input |

## Lumped

`attenuator` (o1, o2), `terminator` (o1).

## Circuit builders (`photonix.circuit`)

| Builder | What it assembles | Evidence |
|---|---|---|
| `mzi`, `ring` | Two-coupler MZI; all-pass ring feedback circuit | 2 |
| `lattice_filter` | Cascaded-MZI FIR lattice with per-stage delays and phases | 2 (equals the 2×2 transfer-matrix product) |
| `crow` | Series-coupled rings between two buses, drop port wired by ring-count parity | 2 (one ring = `add_drop_ring`; lossless energy conservation) |
| `sagnac_mirror` | Loop mirror, \|S11\|² = 4κ(1−κ) | 2 |
| `clements_decompose` + `mzi_mesh` | Rectangular MZI mesh programmed for any unitary; component-level simulation with splitter/phase-shifter imperfections | 2 (reconstructs Haar-random unitaries to 1e-10; exact for identity/permutations) |

## Time-domain transceiver models (`photonix.timedomain`)

These act on sampled waveforms (`Waveform`: optical envelopes in sqrt(W),
electrical signals in V or A) rather than returning an `SDict`; any `SDict`
model above can be placed in a time-domain circuit with
`TimeDomainCircuit.from_netlist`. Full description, validation table and limits:
[`TIMEDOMAIN.md`](TIMEDOMAIN.md).

| Model | Physics | Evidence | Limits |
|---|---|---|---|
| `prbs`, `symbols_to_levels`, `drive_waveform` | ITU-T O.150 LFSRs (PRBS7–31); NRZ / Gray PAM4; closed-form Gaussian or raised-cosine edges with 20–80 % rise time and Gaussian jitter | 2 (period, balance, run lengths, autocorrelation; rise time to 0.2 %) | Random jitter only (no DCD/PJ) |
| `cw_laser` | Wiener phase noise (Lorentzian linewidth) and white RIN | 2 (FWHM within 5 %, RIN PSD within 0.2 dB) | No RIN spectrum shape; use the rate equations |
| `LaserParams`, `rate_equation_laser` | Single-mode rate equations with gain compression, spontaneous emission, Henry phase coupling and Langevin forces (Coldren; Agrawal; Henry 1982); `noise_spectra` gives the linearized RIN and FM noise | 2–3 (threshold, slope, f_R, small-signal response; FM noise/RIN against linear-noise theory, Henry factor) | Illustrative parameters, not a calibrated device; single mode |
| `mzm` + `push_pull` | Dual-drive `cos^2(pi V / 2 V_pi)` with arm imbalance (finite ER) and chirp | 2 (1e-12) | Quasi-static unless an electrode or drive filter is given |
| `TravelingWaveElectrode` | Velocity mismatch, causal skin-effect loss (`(1 + j) alpha(f)`), load/source reflections | 2 (`sin u / u` limit; direct quadrature to 1e-6) | Frequency-independent `Z0` |
| `eam` | `exp(-(1 + j alpha_H) kappa(V))` absorption with Henry chirp | 2 | Absorption curve is an input |
| `RingModulator`, `ring_modulator` | Temporal CMT with photon-lifetime dynamics, `pn_phase_shifter` junction physics, RC drive | 2–3 (steady state vs `all_pass_ring`; Sacher–Poon small-signal within 1 %) | High-Q limit; carrier should be at the laser |
| `photodiode`, `tia`, `AnalogFilter` | `I = R \|E\|^2 + I_d` with shot / thermal / RIN noise filtered exactly in band; RC, Butterworth, Bessel-Thomson filters, exact FOH/ZOH | 2 (`2qIB`, `4kTB/R`, `i_n^2 B` within 4 sigma; rise time `0.35/f3dB`) | Gaussian white noise; no APD gain |
| `eye_diagram`, `mask_test`, `receiver_sensitivity` | Q-optimal sampling, `Q = (mu1 - mu0)/(sigma1 + sigma0)`, `BER = erfc(Q/sqrt 2)/2` (Gray PAM4 per eye), hexagonal mask, sensitivity by root finding | 2 (sensitivity `Q sigma_T / R` within 0.1 dB; counted BER within Wilson CI) | Gaussian BER; no TDECQ
