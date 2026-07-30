# photonix physics audit

Independent numerical audit of the physics in `photonix.em`, `photonix.components`,
`photonix.circuit`, `photonix.optim` and `photonix.core`. Every claim below was
checked against a closed form, an independent solver, a convergence study, or
central finite differences — the evidence is quoted inline.

Findings are graded:

| Grade | Meaning |
|---|---|
| **A** | Bug — produces silently wrong numbers |
| **B** | Documented accuracy claim that does not hold as stated |
| **C** | Physical modelling caveat or convention that is undocumented |
| **D** | Coherence / consistency issue, not a numerical error |

Section 5 lists what was checked and found **correct**. The numerical core — unit
conversions, constants, the circuit interconnection algebra, the compact
component models, FDFD, and all four adjoint gradients — is sound.

---

## 0. 2026-07 deep-audit addendum

The latest independent multi-agent pass found and fixed issues that were not in
the original audit:

- Scalar EME/FDFD gradients now include both exterior Dirichlet faces. The old
  square forward difference imposed different left/right outer conditions.
- EME no longer projects already-solved modes onto parity subspaces (which made
  leaky modes cease to satisfy their eigenproblem). The lossy transverse layer is
  documented as an absorber, not SC-PML.
- FDFD port launch/decomposition now uses the central-difference longitudinal
  phase and discrete conserved flux, rejects longitudinal Nyquist violations, and
  measures reverse transmission instead of copying S21.
- Full-vector magnetic reconstruction now uses `H_t = Q E_t / n_eff`; its PML
  samples integer and half-grid stretches separately.
- Bend grids reject domains crossing the conformal-map singularity `x=-R`.
- Scalar and full-vector solvers preserve decaying evanescent roots rather than
  clipping all negative propagation-constant squares to cutoff.
- Slab `resolution` now means exactly points per micrometre, with subpixel TM
  segment coefficients and consistent two-sided Dirichlet truncation.
- Custom cross-section grids are validated for shape, finiteness, monotonicity,
  and uniform spacing. Inverse-design densities outside a mask are fixed before
  filtering. Touching MMI output strips are rasterized as a geometric union.
- The optional Meep boundary is import-safe and separates cell-centred epsilon
  rasters from node-interpolated density grids. MPB custom grids use a documented
  position-dependent material rather than an unsupported array default.

Remaining approximations are not hidden: full-vector scalar subpixel epsilon,
the equivalent-index bend map, finite EME bases/absorber calibration, MPB material
limits, and the absence of foundry DRC/LVS and calibrated multiphysics. See
`docs/PIC_COMPLETENESS.md` for the production sign-off boundary.

---

## 1. Grade A — bugs producing silently wrong numbers

### A1. EME manufactured energy when `num_modes` exceeded the propagating count — **FIXED**

`em.eme.slab_modes` clipped negative eigenvalues to zero:

```python
betas = np.sqrt(np.clip(vals, 0.0, None))     # removed
```

`beta**2 < 0` means an *evanescent* mode. Clipping gave `beta = 0` exactly, so

1. `_prop` returned `exp(-1j*0*L) = 1` — the mode propagated through any length
   with **no decay**, instead of `exp(-|Im beta| L)`; and
2. `_interface` rescaled to power amplitudes with `sqrt(max(beta, 1e-12))` and
   divided by it, amplifying those rows and columns by `~1e6`.

Measured on a 0.5 → 1.0 µm taper. All singular values of S must be ≤ 1:

| `num_modes` | modes with `beta == 0` | max singular value (before) | after fix |
|---|---|---|---|
| 4  | 0  | 1.000000 | 1.000000 |
| 12 | 0  | 1.000000 | 1.000000 |
| 16 | 4  | **19.1** | 1.0014 |
| 20 | 8  | **235.7** | 1.0014 |
| 24 | 12 | **73 215.6** | 1.0014 |

Reachable through the public API: `taper(num_modes=20)` returned `|T|**2 = 20.8`,
i.e. 2080 % power transmission.

**Fix.** `beta` is now complex on the physical branch (`Re >= 0`, `Im <= 0`), so
evanescent modes attenuate instead of propagating, and the `1e-12` floor is gone.
The residual 1.0014 sits entirely in the **evanescent** rows and columns, whose
amplitudes are not power amplitudes; the **propagating sub-block is exactly
unitary (1.00000000)** at every mode count up to 32, and the power balance on the
fundamental is `sum|Tf|^2 + sum|Rf|^2 = 1.00000000`.

### A2. `em.components.mmi1x2` did not converge numerically — **FIXED**

Sweeping each numerical parameter independently, at the old defaults:

| `num_modes` | 6 | 8 | 10 | **12** | 14 | 16 |
|---|---|---|---|---|---|---|
| excess loss (dB) | 0.340 | 0.345 | 0.473 | **0.664** | 0.912 | 1.124 |

| `points` | 301 | **401** | 601 | 801 |
|---|---|---|---|---|
| excess loss (dB) | 2.530 | **0.664** | 2.744 | 3.333 |

No plateau anywhere; transmission swung by a factor of five under a grid
refinement that should change nothing.

**The original hypothesis was wrong.** C3 (lossless box modes retaining radiated
power) looked like the obvious culprit, but adding a transverse absorber barely
moved the numbers — 0.779 → 0.862 dB at the defaults, with the `points` swing
essentially unchanged. Three separate causes were actually at work:

1. **Staircased cross-sections (dominant).** `_strip` built the index profile
   with a hard `np.where(|x - c| < w/2, ...)` — no subpixel averaging, unlike
   every other profile builder in `photonix.em`. The sidewalls snap to the
   nearest cell, so the modal propagation constants jump as the grid changes:
   the MMI body's beat length `L_pi = pi/(beta0 - beta1)` wandered
   non-monotonically over 15.51–15.75 µm (±0.8 %) across `points` 301..1201.
   Over a ~30 µm device that shifts the self-imaging point by ±0.4 µm — enough to
   slide a fixed-length MMI off its low-loss peak. The *device* had converged;
   the *length* it was evaluated at had not.
2. **Too small a modal basis.** Excess loss climbs from 0.42 dB at 8 modes and
   only plateaus (~1.21 dB) above 20. The old default of 12 was under-counting
   scattered power. This could not be fixed before A1, because `num_modes >= 16`
   blew the S-matrix up.
3. **Non-determinism** (see A5).

**Fix.** Subpixel averaging in `_strip`; defaults raised to `num_modes=24`
(MMI) and `16` (taper); `length_mmi` re-optimized at converged settings to
29.25 µm. Result:

| `points` | 301 | 401 | 601 | 801 |
|---|---|---|---|---|
| before (dB) | 2.404 | 0.779 | 2.785 | 3.008 |
| after (dB)  | 1.147 | **1.147** | 1.152 | 1.149 |

and the optimal length now locks to 29.25–29.50 µm independent of the grid.
`num_modes` plateaus at ~1.21 dB above 20. The taper is unaffected in value
(0.9894) and now flat to 1e-4 across `num_modes` 4..24.

### A3. `em.slab.slab_neff` returned unphysical values when no guided mode existed — **FIXED**

A guided mode requires `n_clad < n_eff < n_core`. The solver never checks:

| `n_core` | `n_clad` | `slab_neff` | |
|---|---|---|---|
| 2.000 | 1.444 | 1.798560 | physical |
| 1.500 | 1.444 | 1.449097 | physical |
| 1.444 | 1.444 | 1.433770 | **below `n_clad`** |
| 1.400 | 1.444 | 1.402258 | **no guided mode exists** |
| 1.200 | 1.444 | 1.263527 | **no guided mode exists** |

`eigsh` is asked for the eigenvalue nearest `(n_core*k0)**2` and returns whatever
box/cladding mode is closest. `slab_neff_analytic` at least raises — though with
an opaque `ValueError: The function value at x=1e-12 is NaN` from `brentq`.

This propagates into **`em.eim.neff` through `lateral_clad`** — the parameter
that exists specifically for rib waveguides, where the slab index beside the core
can approach the vertical effective index `n_v = 2.8475`:

| `lateral_clad` | `eim.neff` | |
|---|---|---|
| 1.444 | 2.4911 | fine |
| 2.800 | 2.8099 | fine |
| 2.850 | 2.8497 | **above `n_v`** |
| 3.000 | 2.8662 | **above `n_v`, no guided mode** |

**Fix.** Both slab entry points now reject `n_core <= n_clad` and validate the
computed root against `n_clad < n_eff < n_core`. `em.eim.neff` inherits the same
validation for its vertical and lateral slab solves.

### A4. `Netlist.validate()` permitted netlists that create energy — **FIXED**

`validate()` only checks that instance *names* exist. It does not check that a
terminal appears in at most one connection, nor that a terminal is not both
connected and exposed. Both malformed cases solve silently:

```python
nl.connect(("a", "o2"), ("b", "o1"))
nl.connect(("d", "o1"), ("a", "o2"))     # a.o2 now in two connections
```

`validate()` passes. The composite has max singular value **1.414 = sqrt(2)**, and
one unit of input power emerges as one unit at *two different* output ports:

```
('p1','p2'): |S|^2 = 1.000000
('p1','p3'): |S|^2 = 1.000000
```

Exposing an already-connected terminal fails the same way. Connecting a port the
model does not have *is* caught, but only at solve time, not by `validate()`.

The interconnection algebra itself is correct — see §5. `Netlist.validate()` and
the direct `evaluate_circuit()` API now reject repeated, self-connected,
connected-and-exposed, unknown, and multiply-exposed terminals before solving.

### A5. EME was non-deterministic — **FIXED**

Three identical `mmi1x2()` calls returned three different transmissions
(0.3836567053, 0.3836567053, 0.3843035359 — spread 6.5e-4). ARPACK seeds itself
from a **random** start vector unless given one, and with near-degenerate modes
the returned basis, and therefore every S-matrix built on it, varied run to run.

**Fix.** A fixed seeded start vector (`v0`) is now passed to both eigensolvers.
Repeated calls are bit-identical, and a regression test pins it.

### A6. MMI even/odd supermode identification by index — **FIXED**

`mmi1x2` assumed output mode 0 was the even supermode and mode 1 the odd. For a
weakly-coupled pair (`gap = 1.0` gives `n_eff` degenerate to six digits) the
eigensolver's ordering of the two is arbitrary, so the assignment silently
swapped. They are now identified by measured **parity** (`<psi|psi(-x)>`), which
comes out at exactly ±1.000000.

### A7. MMI output self-reflections were unequal — **FIXED**

`Rb[even, odd] ~ 0.016` — a mirror-symmetric junction forbids cross-parity
reflection, yet it is non-zero, so the two output ports get unequal
self-reflections (`|r22 - r33| ~ 0.031`). Everything around it checks out: the
supermodes have parity exactly ±1.000000, the geometry and absorber are
bit-symmetric, `Rb` is symmetric to 3e-16 (reciprocal), and the *transmission*
split is balanced to ~1e-3. Transmission is unaffected; only the port
self-reflections are.

**Fix.** For mirror-symmetric PML cross-sections, the computed modes are projected
onto their even/odd parity subspaces and re-normalized before interface matching.
The former strict `xfail` is now a passing regression test.

---

## 2. Grade B — accuracy claims that do not hold as stated

### B1. "<0.1 % vs analytic" holds only for well-confined modes — **DOCUMENTED**

`em.slab` claims "<0.1 % versus the closed-form transcendental for *both*
polarizations". At the default `margin = 2.0`, `resolution = 40`:

| t (µm) | n_core | pol | analytic | numeric | rel. err |
|---|---|---|---|---|---|
| 0.22 | 3.476 | te | 2.847486 | 2.847477 | 3.2e-06 |
| 0.22 | 3.476 | tm | 2.053098 | 2.053081 | 7.9e-06 |
| 0.22 | 1.600 | te | 1.457750 | 1.455608 | **1.5e-03** |
| 0.10 | 1.600 | te | 1.447123 | 1.441006 | **4.2e-03** |
| 0.10 | 2.000 | tm | 1.458226 | 1.459963 | **1.2e-03** |

The error is **domain truncation, not discretization** — the fixed `margin` with
Dirichlet walls cuts off the evanescent tail. For t = 0.10 µm, n_core = 1.6 the
1/e decay length is 2.60 µm, so the default 2.0 µm margin is 0.8 decay lengths:

| margin | decay lengths | rel. err |
|---|---|---|
| 2.0 µm | 0.8 | 4.2e-03 |
| 5.0 µm | 1.9 | 2.0e-04 |
| 10.0 µm | 3.9 | 3.8e-06 |
| 20.0 µm | 7.7 | 1.7e-09 |

Richardson extrapolation cannot help: the truncation error is nearly identical at
both resolutions and passes straight through. The same applies to the 2-D solvers
(`margin = 1.5`).

### B2. Richardson extrapolation was applied in 2-D where its premise fails — **FIXED**

`(4*f - c)/3` assumes error `~ C*h**2` and an exact factor-of-2 refinement.

In **1-D the premise holds** — the slab error ratio converges to exactly 4.00:

| res | 10 | 20 | 40 | 80 | 160 |
|---|---|---|---|---|---|
| err | 8.30e-03 | 8.30e-03 | 4.70e-03 | 9.33e-04 | 2.34e-04 |
| ratio | — | 1.00 | 1.77 | 5.03 | **4.00** |

In **2-D it does not.** Against a res = 320 reference:

| res | 10 | 20 | 40 | 80 | 160 |
|---|---|---|---|---|---|
| err | 8.32e-02 | 9.83e-03 | 2.18e-03 | 7.54e-05 | 2.83e-04 |
| ratio | — | 8.47 | 4.50 | 28.96 | **0.27** |

The error at res = 160 is *larger* than at res = 80. Yet `em.n_eff` has
`richardson=True` by default. Comparing fairly (Richardson at `r` costs an `r`
and a `2r` solve, so compare against plain at `2r`):

| res | Richardson err | plain-at-2r err | verdict |
|---|---|---|---|
| 20 | 5.29e-04 | 2.02e-03 | helps |
| 30 | 3.15e-03 | 2.06e-03 | **worse** |
| 40 | 9.90e-04 | 2.38e-04 | **worse** |
| 60 | 6.54e-04 | 2.59e-05 | **worse** |
| 80 | 2.40e-04 | 1.21e-04 | **worse** |

Building the grid so core edges land on cell faces does not repair it either
(ratios 1.06, −240), so the limitation is not only grid alignment — the
rectangular core's corner field singularity caps the achievable order.

**Fix.** All three 2-D convenience solvers (`n_eff`, `n_eff_vector`, and
`n_eff_fullvector`) now default to `richardson=False`; callers can still opt in
after checking convergence for their geometry. The 1-D slab default remains on,
where the second-order premise is validated.

### B3. `resolution` spacing was inexact in 2-D — **FIXED**

`rectangular_waveguide` uses `linspace(-wx/2, wx/2, nx)`, so `h = wx/(nx-1)`:

| requested `resolution` | actual dx | actual pts/µm |
|---|---|---|
| 20 | 0.05072464 | 19.71 |
| 40 | 0.02517986 | 39.71 |
| 80 | 0.01254480 | 79.71 |

and `h(r)/h(2r) = 2.00719`, not 2 — an independent violation of B2's assumption.

Worse, `em.slab.slab_neff` computes `m = max(int(round((thickness/2)*resolution)), 3)`.
For a 220 nm slab, **every `resolution` from 5 to 30 gives bit-identical results**
(all clamp to `m = 3`, i.e. 27.3 pts/µm):

```
resolution=  5 -> m=3 -> 27.3 pts/um  neff=2.83918412
resolution= 20 -> m=3 -> 27.3 pts/um  neff=2.83918412
resolution= 30 -> m=3 -> 27.3 pts/um  neff=2.83918412
resolution= 40 -> m=4 -> 36.4 pts/um  neff=2.84278889
```

**Fix.** The 2-D geometry builders now use cell-centred grids with spacing
exactly `1 / resolution` and round the cell count up so the requested margin is
never shortened. The slab clamp has also been reduced to the one-cell minimum
and emits a warning when a requested resolution leaves fewer than three cells
in the half-core, so low requested resolutions are no longer silently identical.

---

## 3. Grade C — undocumented physical caveats

### C1. The full-vector solver uses one scalar permittivity for all tensor components — **DOCUMENTED**

`_assemble_fullvector` sets `erxx = eryy = diag(er)` and `erzz_inv = diag(1/er)`
from the same arithmetic subpixel-averaged array. On a Yee grid `Ex`, `Ey` and
`Ez` sit at three different staggered locations, and the field component *normal*
to a dielectric interface requires harmonic (inverse) averaging. Proper
anisotropic subpixel smoothing is what recovers clean second-order convergence at
high-contrast interfaces. Observed order between the two finest grids: 1.73.

### C2. Mode solvers returned non-guided modes unlabelled — **FIXED**

`solve_modes_fullvector(num_modes=8)` on the standard strip returns:

```
 i   Re n_eff  te_frac
 0    2.44657    0.983   guided
 1    1.80895    0.053   guided
 2    1.50284    0.694   guided
 3    1.42960    0.756   <- below n_clad: box mode of the Dirichlet domain
 4..7  < 1.43            <- box modes
```

Five of eight are artifacts of the truncated domain, with `te_fraction` values
that look meaningful but are not. `bend_loss_fullvector` filters internally with
`n_clad < n.real < n_core`, so the codebase knows the filter is required — it is
just not applied in the public solvers.

### C3. EME's closed-window "radiation" basis is a box basis — **DOCUMENTED**

With `absorber=None`, non-guided EME modes are box modes of the finite window:
real `beta`, lossless propagation, re-coupling downstream. Radiated power is
retained rather than lost.

A transverse absorber is now available and is the default for the EME-backed
components (see A2). It is a **graded imaginary-permittivity layer**, not a
stretched-coordinate PML: a true SC-PML puts `k0^2 eps s` on the diagonal of the
modal eigenproblem, and since `|s|` reaches several units inside the absorber it
makes the layer behave like a high-index medium, spawning a dense band of
*absorber* modes at `Re(n_eff) ~ 3.49` — ahead of the true guided mode at 3.272,
with ~58 % of their energy inside the layer, exactly where shift-invert is aimed.
The imaginary-permittivity layer leaves `Re(eps)` untouched and produces no such
band. For TM the absorber is applied to the potential term rather than to `eps`,
because letting complex `eps` into the `B = diag(1/eps)` weight reintroduces the
band (33 of 60 returned modes).

Validated: the guided mode is untouched (`rel. shift 1.3e-16`, `Im(beta) ~ 1e-17`),
radiation modes acquire monotonically increasing `|Im(beta)|`, a uniform section
stays exactly transparent (`|Tf00|^2 = 1.00000000`) at every mode count in both
polarizations, and reciprocity holds to 1e-15.

### C4. `phase_shifter` applied propagation loss but no propagation phase — **FIXED**

```
phase_shifter(length=100, dn_dv=0, voltage=0, loss_db_cm=3): |t|=0.996552, arg=+0.0000
straight     (length=100,              loss_db_cm=3): |t|=0.996552, arg=+1.0134
```

Identical loss — the full `exp(-alpha*L)` for a 100 µm guide — but zero
propagation phase; only the tuning delta is applied. Placing a phase shifter in
one MZI arm against a `straight` of equal length leaves the arms mismatched by
the entire `beta*L`.

### C5. Dangling instance ports are perfect absorbers — **DOCUMENTED**

A terminal that is neither connected nor exposed gets `a_p = 0`, an ideal matched
load. A 2×2 coupler with two ports dangling returns `|S21|**2 = 0.5` with the
other half silently absorbed. The `evaluate_circuit` API now documents this
matched-load convention and directs callers to model reflective terminations
explicitly when required.

### C6. Full-vector `VectorModeData.fields` keeps only the dominant component — **DOCUMENTED**

`_solve_fullvector` stores `dom = ex if fx >= fy else ey`. The minor component is
discarded; `fullvector_transverse_fields` returns both. The dataclass now states
this explicitly and points users to the full-component API.

### C7. `components.bend` had no bridge to the rigorous bend-loss solver — **FIXED**

`bend` models a bend as a straight arc plus a lumped `excess_loss_db`, which
**defaults to 0** — a lossless bend of any radius. `em.bend_loss_fullvector`
computes the real number, but nothing connects them. `bend` also ignores the
bend-induced change in effective index.

---

## 4. Grade D — coherence

- **D1 — FIXED.** All 2-D `n_eff*` helpers default to `richardson=False`; the
  validated 1-D slab solver retains `True`.
- **D2 — FIXED.** `EMEResult.sdict()` uses canonical `o1 … oN` names and exports
  all four blocks, including output-side reflection `Rb`.
- **D3 — FIXED.** `em/operators.py` now points to the shipped vectorial operators
  in `em/fde_vector.py` and accurately scopes itself to scalar assembly.
- **D4 — FIXED.** `benchmarks/RESULTS.md` distinguishes 2/2 external references,
  2/2 internal FDFD cross-checks, and the one case without a reference.

---

## 5. Verified correct

**Units and constants**

- `db_per_cm_to_alpha_um`: exact round trip at 0.5, 3, 20 dB/cm (< 1e-9 dB).
- `neff_linear`: reproduces the requested `ng` at `wl0` analytically.
- `1/sqrt(MU0*EPS0)` vs `C0`: rel. 2.2e-14. `sqrt(MU0/EPS0)` vs `ETA0`: 3.0e-12.
- `N_SIO2`, `N_SIN` match the Sellmeier models to 4 decimals. (`N_SI = 3.4757`
  vs `silicon(1.55) = 3.4777` — different literature fits, not an error.)

**Compact component models**

- `directional_coupler` unitary to 2.2e-16 across `coupling` in [0, 1].
- `components.mmi1x2` singular values exactly (1, 1, 0) — matching the documented
  deliberate sub-unitarity of a matched reciprocal 3-port.
- `grating_coupler`: `bandwidth` is the power FWHM to 1e-4 relative.
- All 14 models in `components.MODELS` are reciprocal and passive.

**Circuit solver**

- The interconnection algebra `M = (I - S*Gamma)^-1 * S` is correct, including
  the exposed-port extraction.
- Circuit-assembled ring vs closed-form all-pass ring: max difference **1.8e-15**
  over a 4001-point resonance sweep (independent implementations).
- Analytic vs circuit-assembled MZI: 2.9e-14.
- Add-drop ring: passive, reciprocal, measured FSR 9.081 nm vs `wl^2/(ng*L)`
  9.104 nm (0.25 %).

**EME**

- Straight-guide transparency exact (`|Tf00|^2 = 1.0000000000`, `max|Rf| = 0`),
  with and without the absorber, both polarizations, every mode count tested.
- Step junction: propagating sub-block max singular value exactly 1.00000000;
  `|S - S^T| = 3e-16`.

**FDFD**

- Straight-guide energy conservation `T + R = 1.000041` (TE), `1.000032` (TM),
  stable across `npml` = 8, 12, 20.
- `|S21|^2` invariant to 1e-6 across four source/monitor placements.

**Adjoint gradients** — all match central finite differences:

| quantity | rel. error |
|---|---|
| `fdfd.focus_objective` (4 pixels) | 3.5e-10 … 2.2e-08 |
| `fabrication.density_to_eps_vjp` (corners, edges, interior) | 1.7e-10 … 1.7e-09 |
| `n_eff_eps_vector` (non-symmetric adjoint) | 3.7e-08 |
| `n_eff_eps_fullvector` | 4.3e-08 |

`conic_filter_adjoint`'s boundary handling is correct — corner and edge pixels
match to 1e-9, which the naive self-adjoint shortcut would not.

**Other**

- `fullvector_transverse_fields` bi-orthonormality: `|M - I| = 1.6e-15`.
- `bend_loss_fullvector` dB conversion `4.343 * 2*k0*Im(n_eff) * arc` with
  `arc = pi*R/2` is dimensionally and numerically correct.
- The TM finite-volume slab operator is correctly assembled and symmetric; the
  semivectorial five-point stencil matches the analytic derivation of
  `d/dx[(1/eps) d(eps Ex)/dx]` term by term.
- EIM's polarization rotation (vertical TE → lateral TM) is correct, and EIM
  reproduces the vertical slab in the wide-width limit to <0.1 %.
- "TE" consistently means out-of-plane **E** in every module.

---

## 6. Remaining accuracy work

The correctness and coherence items above are fixed or explicitly documented.
The main numerical improvement still available is **C1**: anisotropic subpixel
smoothing for cleaner second-order full-vector convergence at high-contrast
interfaces. This is an accuracy enhancement to the documented solver
approximation rather than a silent API or packaging defect.
