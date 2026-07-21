# photonix EM solver suite — technical design

**Status:** partially implemented & validated — see Section 16 for as-built
state (semivector + full-vector FDE, PML/bend loss, vectorial TE/TM EME, TM
FDFD, and the EME↔FDFD cross-solver benchmark are built and tested; the
reciprocal 2-D hybrid EME cascade is the main open item). **Scope:** the rigorous, differentiable
electromagnetic core that elevates photonix from compact-model circuit
simulation (v0.1) toward parity with commercial PIC tools (Lumerical MODE/FDTD,
Ansys/RSoft, Tidy3D). **Author:** photonix core. **Target version:** 0.2–0.4.

---

## 1. Goals and competitive thesis

photonix v0.1 computes circuit responses from *analytic compact models*. That is
fast and fully differentiable, but the S-parameters are only as good as the
hand-written models. Commercial tools are trusted because their S-parameters come
from **solving Maxwell's equations on the actual geometry**. This document
specifies that rigorous core.

We will not try to clone 20 years of Lumerical features. We win on one wedge:
**rigorous physics that is also end-to-end differentiable and GPU-ready, in one
open stack.** Concretely, every solver here exposes gradients of its outputs
(n_eff, S-parameters, fields) with respect to geometry and material parameters,
so the entire chain — cross-section → modes → propagation → circuit → figure of
merit — is a single autodiff graph. That is the capability the incumbents lack.

Four solvers, mirroring the standard PIC toolbox:

| Solver | Class | What it computes | Best for |
|---|---|---|---|
| **FDE** | Finite-Difference Eigenmode | Full-vectorial modes, complex n_eff, n_g, dispersion, fields | Waveguide cross-sections; feeds everything |
| **FDFD** | Finite-Difference Frequency-Domain | Rigorous fields & S-params at one frequency (2D / 2.5D) | Compact devices, adjoint topology optimization |
| **EME** | Eigenmode Expansion | S-matrix of z-varying structures by modal propagation | Tapers, MMIs, long adiabatic devices, periodic structures |
| **varFDTD** | Effective-index 2.5D FDTD | Broadband S-params of planar circuits in time domain | Rings, MZIs, MMIs, planar layouts (fast 3D-like accuracy) |

Honest non-goals for this phase: full vectorial **3D** FDTD/FEM, multiphysics
(thermal/charge), and foundry sign-off DRC/LVS. Those are later tiers; the EM
core is the prerequisite for all of them.

---

## 2. Architecture

New subpackage `photonix.em`, layered on `photonix.core`. The existing scalar
solver in `photonix.modes` is kept as a fast approximation and re-homed as a
special case; FDE becomes the rigorous default.

```
src/photonix/em/
  grid.py        # Yee grid, coordinate arrays, 1-D derivative operators
  operators.py   # curl / divergence operators on the Yee mesh (sparse)
  pml.py         # stretched-coordinate PML (SC-PML) factors s_w(w)
  materials.py   # dispersive (Lorentz/Drude/Sellmeier) + anisotropic eps tensors
  eig.py         # differentiable sparse generalized eigensolver (custom VJP)
  fde.py         # full-vectorial finite-difference eigenmode solver
  fdfd.py        # frequency-domain Maxwell solver + adjoint
  eme.py         # eigenmode-expansion propagator (reuses core S-matrix algebra)
  fdtd.py        # 2-D FDTD engine (Yee, CPML, mode source, DFT monitors)
  varfdtd.py     # effective-index collapse + 2-D FDTD driver
  sources.py     # mode sources, Gaussian pulses, plane waves
  monitors.py    # field / flux monitors, S-parameter extraction (overlap + DFT)
  results.py     # ModeData, FieldData, SweepResult dataclasses
```

**Data contracts.** All solvers return plain backend arrays/`ModeData`/`FieldData`
and ultimately an `SDict` (the v0.1 type), so solver outputs drop straight into
`photonix.circuit` and `photonix.optim`. A component model gains a `method=`
switch: `"analytic" | "fde" | "fdfd" | "eme" | "varfdtd"`, defaulting to analytic
for speed and escalating to a rigorous solver on demand.

**Backend.** All array math goes through `photonix.core.backend.xp` (JAX). Sparse
assembly uses `scipy.sparse` for matrix *construction* and ARPACK/`splu`/GMRES
for the heavy factorizations; differentiability is provided by **custom VJPs**
that wrap these non-differentiable kernels with analytic adjoints (Sections 4 and
6.2). This keeps us differentiable today on CPU and lets us swap in
`jax.experimental.sparse` / GPU iterative solvers later without changing the API.

---

## 3. Shared infrastructure

### 3.1 Yee grid and derivative operators
A staggered Yee grid stores E and H components at offset locations, giving
second-order-accurate, charge-conserving curl operators. We build 1-D forward/
backward difference matrices `Dxf, Dxb, Dyf, Dyb` (sparse, with the chosen
boundary condition) and assemble 2-D/3-D curl operators by Kronecker products.
The same operators serve FDE, FDFD, and FDTD, so correctness is tested once.

### 3.2 PML — stretched-coordinate formulation
Open boundaries use a stretched-coordinate PML (SC-PML). Each spatial derivative
∂/∂w is replaced by (1/s_w) ∂/∂w with the complex stretch factor

```
s_w(w) = κ_w(w) + σ_w(w) / (j ω ε0)              (frequency domain, FDFD/FDE)
```

with a polynomial-graded conductivity σ_w ramping into the absorber and κ_w ≥ 1.
For FDTD we use the equivalent **CPML** (convolutional PML) recursion. PML enters
FDE as complex-valued stretch factors multiplying the derivative operators, which
is exactly what makes **leaky and bent** modes (complex n_eff with radiation
loss) computable.

### 3.3 Materials
- Non-dispersive: scalar or diagonal/anisotropic ε tensor.
- Dispersive (FDFD per-frequency): ε(ω) from Sellmeier (reuse
  `photonix.modes.materials`) or tabulated n,k.
- Dispersive (FDTD): pole models — Drude, (multi-)Lorentz, Debye — fitted to n,k
  and integrated with the auxiliary-differential-equation (ADE) update.
- All material parameters are differentiable inputs (enables material/spatial
  inverse design).

---

## 4. Differentiable sparse eigensolver (`em/eig.py`)

The crux of the whole suite. FDE and EME need eigenpairs of a large sparse
**generalized** problem

```
A(p) x = λ B(p) x ,        x normalized so that  xᵀ B x = 1
```

where `p` are the differentiable parameters (ε grid, geometry, wavelength).
ARPACK (`scipy.sparse.linalg.eigs`) computes `(λ, x)` but is not differentiable,
so we wrap it in a `jax.custom_vjp`.

**Forward:** shift-invert ARPACK around a target near the core index returns the
`k` most-confined eigenpairs.

**Backward (analytic adjoint).** For a simple eigenvalue with left eigenvector
`y` (`yᵀ A = λ yᵀ B`; `y = x` when A,B symmetric):

```
∂λ = (yᵀ (∂A − λ ∂B) x) / (yᵀ B x)          (generalized Hellmann–Feynman)
```

For the eigenvector cotangent `x̄`, solve one bordered/projected linear system per
returned mode,

```
[ A − λB    −Bx ] [ v ]   [ (I − B x xᵀ) x̄ ]
[ (Bx)ᵀ      0  ] [ μ ] = [        0        ]
```

(`A − λB` is singular on its own; the border removes the null direction). The
parameter gradient then assembles from `v`, `x`, and the operator derivatives
`∂A/∂p`, `∂B/∂p`, which are themselves sparse and cheap. This is standard
implicit differentiation of eigenproblems and gives exact gradients at the cost
of one extra sparse solve per mode.

**Why this matters:** it makes `d n_eff / d(width, thickness, ε, λ)` available
analytically, so waveguide geometry can be optimized by gradient descent and
group index / dispersion fall out of the same machinery without finite
differences.

---

## 5. FDE — full-vectorial finite-difference eigenmode solver (`em/fde.py`)

### 5.1 Formulation
We solve the source-free vectorial wave equation on the 2-D cross-section for the
transverse magnetic field `Hₜ = (Hx, Hy)`, following the canonical full-vector
finite-difference scheme (Fallahkhair–Li–Murphy–Webb; the method used by EMpy and
Lumerical FDE). Eliminating the longitudinal components yields a generalized
eigenproblem

```
P [Hx; Hy] = (n_eff² k0²) [Hx; Hy] ,     P = [[Pxx, Pxy], [Pyx, Pyy]]
```

where the 2×2 operator blocks `P··` contain the Yee-grid derivative operators and
the (possibly anisotropic, PML-stretched) permittivity. Solving for the largest
`n_eff²` gives the most-confined modes; PML stretch factors make `n_eff` complex,
encoding radiation/leakage loss. We assemble `A = P`, `B = I`, hand them to
`em/eig.py`, and recover `(Ex,Ey,Ez,Hx,Hy,Hz)` from `Hₜ` by back-substitution.

### 5.2 Capabilities
- TE/TM/hybrid full-vector modes; mode ordering and polarization fraction (TE/TM
  ratio) for labeling.
- **Complex n_eff** → propagation loss (dB/cm) from `Im(n_eff)`.
- **Group index & dispersion**: `n_g` and `D` from analytic `dn_eff/dλ` (via the
  eigenvalue VJP, exact) rather than finite differencing.
- **Bent modes**: conformal map of the radial coordinate (index transform
  `n(x) → n(x)·(1 + x/R)`, equivalently solving in cylindrical coordinates) gives
  bend mode profiles and bend/radiation loss vs radius.
- Anisotropic/diagonal ε (e.g. stress, LiNbO₃) supported by the tensor assembly.
- Symmetry boundary conditions (PEC/PMC) to halve the grid.

### 5.3 API
```python
from photonix.em import fde
res = fde.solve(
    eps,                  # 2-D permittivity (from em.materials / geometry), or a
                          # cross-section spec (width, thickness, stack)
    wl=1.55, num_modes=4,
    boundary="pml",       # "pml" | "pec" | "pmc" | "periodic"
    radius=None,          # bend radius (µm); None = straight
)
res.n_eff        # complex array, descending Re
res.n_group(wl)  # group index (analytic dλ)
res.loss_db_cm   # from Im(n_eff)
res.fields       # FieldData: Ex,Ey,Ez,Hx,Hy,Hz on the grid
res.te_fraction  # polarization label
# differentiable convenience:
neff = fde.n_eff(width=0.5, thickness=0.22, wl=1.55)   # grad-able scalar
```

### 5.4 Validation targets
- Symmetric slab TE0/TM0 vs analytic transcendental solution (< 1e-4 at fine grid;
  the scalar solver already hits 5e-3 — full-vector must do better on TM).
- 500×220 nm SOI strip @1550: TE0 `n_eff ≈ 2.44`, TM0 `≈ 1.78` (literature);
  `n_g(TE0) ≈ 4.2`. Target < 1% vs published/Lumerical.
- Bend loss vs radius monotonic and matching known SOI curves (e.g. negligible
  above ~5 µm, rising sharply below ~3 µm).
- Grid convergence O(h²); Richardson-extrapolated n_eff stable.
- Gradient check: `d n_eff/d width` vs central finite difference < 1e-4.

---

## 6. FDFD — frequency-domain Maxwell solver (`em/fdfd.py`)

### 6.1 Formulation
At a single frequency we solve the curl-curl equation on the Yee grid with SC-PML:

```
( ∇× μ⁻¹ ∇×  −  ω² ε ) E  =  −j ω J            →   A(ε) e = b
```

In 2-D this separates into TE/TM scalar Helmholtz problems; for planar PICs we use
a **2.5-D** variant (effective vertical index, Section 8.1) to capture slab
confinement cheaply. `A` is large, sparse, complex, non-Hermitian; we factor with
sparse LU (`scipy.sparse.linalg.splu`) for 2-D, GMRES + ILU for larger problems.

### 6.2 Differentiability (adjoint)
For any objective `L(e)` with `A e = b`:

```
forward:   e  = A⁻¹ b
adjoint:   solve  Aᵀ a = (∂L/∂e)ᵀ
gradient:  ∂L/∂p = −aᵀ (∂A/∂p) e  +  aᵀ (∂b/∂p)
```

one extra solve with the *same factorization* (transpose) gives the full gradient
w.r.t. every permittivity pixel — i.e. **adjoint topology optimization** out of
the box (the ceviche approach), exposed through `jax.custom_vjp`.

### 6.3 S-parameter extraction
Inject a mode at an input port (FDE-computed profile as a current source),
measure complex overlap at each output port → `SDict`. Reciprocity and (lossless)
unitarity are asserted in tests.

### 6.4 API & validation
```python
from photonix.em import fdfd
sim = fdfd.Simulation(eps, wl=1.55, boundary="pml")
e = sim.solve(source=fdfd.ModeSource(port="o1"))
s = sim.sparameters(ports=["o1","o2","o3","o4"])   # -> SDict
g = px.grad(lambda eps: objective(fdfd.Simulation(eps, wl=1.55).sparameters(...)))(eps)
```
Validation: point-source Green's function vs analytic; a 50/50 directional
coupler S-matrix vs EME and vs the analytic component; energy balance
(Σ|S|²≤1); adjoint gradient vs brute-force finite difference on a few pixels.

---

## 7. EME — eigenmode expansion (`em/eme.py`)

### 7.1 Formulation
Slice the device along propagation `z` into piecewise-z-invariant **cells**. In
each cell, FDE gives a modal basis `{ψ_k}` with propagation constants `β_k`. The
field anywhere in a cell is `Σ_k (a_k e^{−jβ_k z} + b_k e^{+jβ_k z}) ψ_k`
(forward + backward). Two operations build the full response:

- **Propagation** within a cell of length `L`: a diagonal S-matrix of phases
  `e^{−jβ_k L}` (with `Im(β_k)` giving loss).
- **Interface** between cell A and cell B: continuity of transverse E and H gives
  an overlap matrix `O_{kl} = ¼∫ (E_k^A × H_l^B + E_l^B × H_k^A)·ẑ dA`. From the
  overlaps we form the interface transmission/reflection S-matrix (mode matching).

The device S-matrix is the **cascade** of all interface and propagation
S-matrices — and we reuse `photonix.core.sparams` / the v0.1 circuit S-matrix
combination for the cascade, so EME and circuit simulation share one tested
engine. Repeated identical cells (periodic structures, long straights) collapse
via fast matrix powers, the key EME speed advantage over FDTD for long adiabatic
devices.

### 7.2 Capabilities, API, validation
- Tapers, MMIs, adiabatic couplers, grating-like periodic sections; bidirectional
  (handles reflections, unlike BPM); length sweeps are nearly free (re-cascade,
  no re-solve).
- Differentiable through mode profiles (FDE VJP), overlaps, and phases → optimize
  taper length, MMI width, etc. by gradient.

```python
from photonix.em import eme
dev = eme.Device(cross_sections=[...], lengths=[...], wl=1.55, num_modes=20)
s = dev.sparameters()                 # -> SDict
T = dev.length_sweep("L_taper", values)   # cheap re-cascade
```
Validation: adiabatic taper transmission → 1 as length grows; 1×2 MMI 50/50 split
and imbalance vs published Lumerical EME values; reciprocity & lossless
unitarity; convergence vs number of modes.

---

## 8. varFDTD — effective-index 2.5-D FDTD (`em/varfdtd.py`, `em/fdtd.py`)

### 8.1 Effective-index collapse
Most silicon-photonics geometry is planar (a patterned slab). varFDTD reduces the
3-D problem to 2-D by collapsing the vertical (thickness) dimension: solve the
local 1-D vertical slab mode to get an in-plane **effective index map**
`n_eff(x, z)` (and an effective dispersion), then propagate in 2-D. This gives
near-3-D accuracy for planar devices at ~2-D cost — the Lumerical varFDTD/“2.5-D”
propagator. The collapse uses a reference slab mode plus a perturbative
correction for lateral index steps.

### 8.2 2-D FDTD core
`em/fdtd.py` is a Yee-grid leapfrog time-stepper with CPML, implemented as a
`jax.lax.scan` over time steps so it is `jit`-compiled and runs on GPU. Features:
- Mode source (FDE profile) injected via total-field/scattered-field or a soft
  source; broadband Gaussian pulse for one-shot spectra.
- DFT monitors accumulate frequency-domain fields during the run → broadband
  S-parameters from a single simulation via modal overlap at each port.
- Dispersive media via ADE (Drude/Lorentz) when needed.
- Stability via the Courant condition; auto-set `dt`.

### 8.3 Differentiability
FDTD is a time recurrence, so gradients come from the **adjoint (reverse-time)
simulation**. Two routes: (a) unrolled autodiff through `lax.scan` with
**gradient checkpointing** (`jax.checkpoint`) to bound memory; (b) a hand-written
adjoint FDTD for large runs. This yields gradients of broadband S-params w.r.t.
geometry/material — broadband adjoint optimization.

### 8.4 API & validation
```python
from photonix.em import varfdtd
sim = varfdtd.Simulation(layout_or_eps, wl=(1.5, 1.6), thickness=0.22, resolution=30)
s = sim.sparameters(ports=[...])      # broadband SDict from one pulsed run
```
Validation: ring/MZI/MMI broadband S-params vs the analytic models *and* vs a
full-3-D reference (Meep/Tidy3D, used only as an offline cross-check, never a
runtime dependency); energy conservation; grid + PML reflection convergence.

---

## 9. Differentiability summary

| Solver | Forward kernel | Gradient mechanism | Status of method |
|---|---|---|---|
| FDE | ARPACK generalized eig | custom VJP: Hellmann–Feynman + bordered solve | standard, exact |
| FDFD | sparse LU / GMRES | adjoint (one transpose solve), `custom_vjp` | standard (ceviche), exact |
| EME | FDE modes + overlaps + cascade | autodiff through overlaps/phases (FDE VJP underneath) | exact |
| varFDTD | `lax.scan` time-stepping | reverse-time adjoint / checkpointed autodiff | standard, exact (memory-bounded) |

Every solver therefore plugs into `photonix.optim` unchanged: `value_and_grad`
of a circuit-level objective flows down into geometry.

---

## 10. Validation & benchmarking strategy

Credibility is the product. We add `tests/em/` plus a `benchmarks/` suite that is
run in CI (fast cases) and offline (heavy 3-D cross-checks):

1. **Analytic anchors** — slab modes, point-source Green's functions, single-mode
   ring transfer functions.
2. **Cross-solver consistency** — the *same* directional coupler must agree across
   FDFD, EME, varFDTD, and the analytic model within stated tolerances. This is
   our strongest internal correctness signal.
3. **Literature/measurement anchors** — published SOI strip n_eff/n_g, MMI
   imbalance, grating-coupler bandwidth; reproduce within a few percent.
4. **External reference (offline)** — selected cases vs Meep/Tidy3D, reported as a
   benchmark table in the docs (numbers, not adjectives).
5. **Gradient verification** — every differentiable output vs central finite
   differences on a small case, in the test suite.
6. **Convergence** — documented O(h²) / mode-count / PML-thickness studies.

---

## 11. Performance & scaling roadmap

- **Now (0.2):** `scipy.sparse` assembly; ARPACK shift-invert (FDE); sparse LU
  (FDFD 2-D); `jit`+`scan` FDTD on CPU/GPU. Wavelength sweeps via `vmap`.
- **Next (0.3):** GMRES + preconditioners for larger FDFD; GPU sparse via
  `jax.experimental.sparse`/CuPy; checkpointed adjoint FDTD.
- **Later (0.4+):** domain decomposition / multi-GPU FDTD; sparse KLU circuit
  solver for 10⁴⁺-element circuits; optional MKL/UMFPACK factorization backend.

Targets: 2-D FDE cross-section < ~0.5 s at engineering resolution; a planar MZI
broadband varFDTD run in seconds-to-minutes on a GPU.

---

## 12. Integration with existing photonix

- `photonix.components`: each model gains `method="analytic" | "fde" | "fdfd" |
  "eme" | "varfdtd"`. Analytic stays the default; rigorous solvers are opt-in and
  cached. A solved component can be **fit** to a compact model for fast reuse
  (model extraction).
- `photonix.modes`: the scalar solver is retained as `method="scalar"`; FDE is the
  new rigorous default. Same `n_eff`/`group_index` interface, so callers don't
  change.
- `photonix.circuit`: unchanged — it already consumes any `SDict`, including those
  produced by FDFD/EME/varFDTD.
- `photonix.pdk`: components can register a rigorous solver recipe alongside the
  compact model.
- `photonix.optim`: unchanged; now optimizes through real physics.

---

## 13. Dependencies & risks

- **Deps:** `scipy.sparse` (have it). Optional later: `jax.experimental.sparse`,
  `scikit-umfpack`/MKL, CuPy. `meep`/`tidy3d` are **dev-only** validation
  cross-checks, never runtime deps. No new hard runtime dependency.
- **Risks & mitigations:**
  - *Differentiable sparse eig is the trickiest piece* → build `em/eig.py` first,
    in isolation, validated against finite differences before any solver uses it.
  - *FDTD adjoint memory* → checkpointing; cap problem sizes in CI.
  - *PML correctness is a classic bug source* → dedicated reflection tests for the
    shared PML before solvers depend on it.
  - *Performance on CPU* → ship correct-first; GPU/iterative is a drop-in later
    because the API hides the factorization.

---

## 14. Milestones (proposed sequencing, each independently validated)

- **M0 — shared core:** `grid.py`, `operators.py`, `pml.py`, `materials.py`,
  `eig.py` (differentiable eig). Acceptance: curl operators reproduce analytic
  derivatives; PML reflection < −40 dB; eig gradient matches FD < 1e-4.
- **M1 — FDE:** full-vectorial straight + bent modes. Acceptance: slab analytic
  < 1e-4; SOI strip TE0/TM0/n_g within 1% of literature; `dn_eff/dwidth` vs FD.
  Wire FDE into `components.straight(method="fde")`.
- **M2 — FDFD + adjoint:** 2.5-D solver, S-params, adjoint topology-opt hook.
  Acceptance: coupler S-matrix vs analytic/EME; adjoint grad vs FD.
- **M3 — EME:** propagator reusing core S-matrix cascade. Acceptance: taper →
  unity, MMI split vs reference; mode-count convergence.
- **M4 — varFDTD:** effective-index collapse + 2-D FDTD + broadband S-params +
  checkpointed adjoint. Acceptance: ring/MZI/MMI vs analytic and vs offline 3-D.
- **M5 — integration & benchmarks:** `method=` switch across components/PDK;
  benchmark table in docs; cross-solver consistency suite in CI.

Each milestone is a reviewable PR with its own tests; the order respects hard
dependencies (everything needs M0; EME needs FDE; varFDTD needs the FDTD core).

---

## 15. Open questions for review

1. **Priority within the suite** — FDE first is non-negotiable (everything needs
   it). After that, do you want **FDFD+adjoint** next (best for inverse design) or
   **EME** next (best for tapers/MMIs and reuses the most existing code)?
2. **varFDTD vs full 3-D FDTD** — varFDTD covers planar PICs at low cost and is the
   right near-term target; a true 3-D FDTD is a much larger lift. Agreed to defer
   full 3-D?
3. **External validation bar** — is matching Meep/Tidy3D within a stated few-
   percent (documented in a benchmark table) the credibility target you want, or
   do you have specific measured devices/foundry data to anchor against?
4. **GPU timeline** — correctness-first on CPU now with a GPU drop-in later, or is
   GPU performance a hard requirement for the first rigorous release?


---

## 16. As-built status — implemented & validated

This section records what is actually implemented and validated in the codebase,
versus the proposal above. Every claim here is backed by a passing test
(`tests/test_em_*.py`) or a runnable example (`examples/`).

### 16.1 Mode solvers (`em/fde_vector.py`)

The FDE was built in an accuracy ladder, each rung correcting the previous and
validated against the canonical 500×220 nm SOI strip at 1.55 µm:

| solver | TE0 n_eff | what it adds | validation |
|---|---|---|---|
| scalar (`fde.py`) | 2.611 | baseline Helmholtz | <0.1 % vs analytic slab |
| **semivector** (`solve_modes_vector`) | 2.489 | interface `(1/eps) d(eps·)` weighting; quasi-TE/TM | non-symmetric adjoint vs FD to 4.5e-7 |
| **full-vector** (`solve_modes_fullvector`) | 2.449 | Yee `Omega = P@Q`, hybrid Ex/Ey, polarization fraction | literature ~2.44; adjoint vs FD to 8.5e-8 |

Both vector operators are **non-symmetric**, so the differentiable `n_eff(eps)`
adjoint uses the *left* eigenvector: `dλ/deps = (uᵀ dA v)/(uᵀ v)`, supplied via
`jax.custom_vjp` and the frozen-bilinear-form trick (`n_eff_eps_vector`,
`n_eff_eps_fullvector`). The full-vector eigenvalue is `λ = -n_eff²`. The
operator matvec is reproduced in `xp` for the adjoint and asserted to match the
scipy sparse operator to ~1e-13.

The full-vector fundamental is 98 % Ex (TE-like) and the second mode 95 % Ey
(TM-like) — a genuine hybrid decomposition the scalar/semivector solvers cannot
produce.

### 16.2 PML & bend loss (`bend_loss_fullvector`)

Stretched-coordinate PML makes `n_eff` complex (radiation loss). Validated:
a straight guide with PML keeps `Im(n_eff) ~ 1e-8` (non-perturbing); bend loss
from the conformal map `n -> n(1 + x/R)` rises monotonically as the radius
tightens (~1e-4 → ~6e-3 dB/90° over R = 1.5 → 1.0 µm), with the physical leaky
mode selected by overlap with the straight fundamental (rejecting spurious PML
modes). Two implementation lessons are baked in: the outer window must reach the
radiation caustic at `x_c ~ R(n_eff/n_clad - 1)`, and the inner boundary must
stay inside `x = -R` (where the conformal index would cross zero).
Status: validated as a *trend*, not yet pinned to measured dB.

### 16.3 Vectorial EME (`em/eme.py`)

EME is polarization-aware (`polarization="te"|"tm"`). TE is the original scalar
path. **TM** uses `Hy` modes from the generalized eigenproblem
`A Hy = β² B Hy` (`B = diag(1/eps)`) and the **1/eps-weighted power overlap**
`O_lk = integral (1/eps_B) Hy_l^B Hy_k^A dx`, since TM power flows as
`~(β/eps)|Hy|²`. Validated: transparent interface and reciprocity to ~1e-16,
energy conservation exact, and a smooth taper stays adiabatic with reflection at
the ~1e-5 floor. The correct TM overlap weight was *selected by adiabaticity*
(energy conservation alone is non-discriminating because the symmetric interface
construction is unitary for any overlap) and then **independently confirmed** by
the TM FDFD (below). A `beta -> 0` floor guards the cascade against
non-propagating modes; EME represents guided modes only (no radiation continuum).

### 16.4 TM FDFD (`em/fdfd.py`)

The FDFD gained a TM (`Hz`) polarization: `div((1/eps) grad Hz) + k0² Hz` with
face permittivity and a `1/eps`-weighted modal projection. A straight TM guide is
lossless to 1.0000. This is an *independent* full-wave check of the TM EME.

### 16.5 Cross-solver benchmark (`tests/test_em_benchmark.py`)

The first entry in validation-as-a-product: EME (modal) and FDFD (full-wave)
compute the transmission of the same planar width step and must agree.

| case | EME | FDFD | |ΔT| |
|---|---|---|---|
| TE width step | 0.9934 | 0.9927 | 7e-4 |
| TM width step | 0.9992 | 0.9967 | 2.5e-3 |

This independently retires the TM-overlap ambiguity that adiabatic
self-consistency alone left open.

### 16.6 2-D vectorial foundation (`fullvector_transverse_fields`, `power_overlap`)

Toward a 2-D hybrid EME: the full-vector transverse fields with the magnetic
field reconstructed as `H_t = Q @ e` are **power-orthonormal** under the Poynting
overlap `integral (Ex_a Hy_b - Ey_a Hx_b) dA = δ_ab`, validated to ~1e-14. This
bi-orthonormality is the prerequisite for vectorial mode-matching, and the
self-interface is transparent to machine precision.

**Open: the reciprocal 2-D cascade.** A junction built naively from these
overlaps is energy-bounded with a machine-precision transparent limit, but is
*not reciprocal* in a truncated mode basis (observed `|Tf - Tbᵀ| ~ 0.3`) and a
taper came out anti-adiabatic. The correct reciprocal vectorial-interface
formulation is the remaining work for a full 2-D hybrid EME; the modes and
overlap shipped here are correct and useful on their own.

### 16.7 Honest open edges

* bend-loss magnitudes are a validated trend, not benchmarked to measured dB;
* EME is guided-mode-only and 1-D-cross-section; the reciprocal 2-D hybrid
  cascade is unfinished (16.6);
* all cross-checks are *internal* (EME ↔ FDFD); an external reference
  (Meep / Tidy3D / measured device) is the next credibility rung — see the
  `benchmarks/` scaffold;
* GPU execution is untested (the JAX core is GPU-ready but CPU-validated here).

### 16.8 Test & example inventory

`tests/test_em_fde_vector.py` (semivector, full-vector, PML/bend, 2-D overlap),
`tests/test_em_eme.py` (TE + TM EME), `tests/test_em_benchmark.py` (EME↔FDFD,
TE + TM), plus the existing scalar-FDE / FDFD / EME suites. Runnable showcase:
`examples/vector_em_showcase.py` → `examples/outputs/vector_em_showcase.png`.

## 17. Meep backend — FDTD by delegation (`em/meep/`)

The original roadmap listed a from-scratch 3-D Maxwell solver (FDTD/FEM) as the
largest open item. Rather than re-implement a mature solver, photonix **delegates
every FDTD need to MIT Meep** (and its bundled mode solver MPB) through an optional
extension, `photonix.em.meep`. The design goal is that switching from photonix's
own frequency-domain solvers to FDTD is an *import change*, not a data-model
change: the backend speaks native photonix types in both directions.

### 17.1 Optionality & the unit bridge (`_guard.py`)

Meep is heavy and conda-only, so it is a strict optional dependency, with a
**fail-loud-but-localised** import contract: `import photonix.em` never requires
Meep (the backend is not imported eagerly — `em/__init__.py` exposes `meep` through
a module-level `__getattr__`), but the backend itself *does* require Meep at import.
So `from photonix.em import meep` (or any `em.meep…` access) raises a single,
install-hint `ImportError` the moment it is touched without Meep — the requirement
is surfaced immediately and unambiguously, rather than deferred to deep inside a
solver call or silently downgraded. The package `__init__` enforces this by calling
`require_meep()` at import; `require_meep()` / `require_mpb()` also guard the
individual solvers, and `HAS_MEEP` / `HAS_MPB` allow a cheap availability probe
(via `importlib.util.find_spec`) without triggering the raise. The unit bridge fixes
Meep's length scale at `a = 1 µm`, giving `f = 1/λ_µm` and `n_eff = k/f`;
`meep_frequency`, `n_eff_from_k`, `k_from_n_eff` are the only places those
conversions live.

### 17.2 Geometry translation (`materials.py`, `geometry.py`)

Any photonix permittivity grid maps onto a Meep `MaterialGrid` by the linear-in-ε
weight `w = (ε − ε_lo)/(ε_hi − ε_lo)` — lossless for a two-material structure,
piecewise-linear otherwise. The weight computation (`material_grid_weights`), the
cell sizing and the column/row → Meep-coordinate mapping (`cell_size`, `col_to_x`,
`row_to_y`, `DeviceGrid`) are **pure NumPy** (they import no Meep symbols at
runtime); only the thin `to_material_grid` / `build_block` wrappers touch Meep.
Because the backend now requires Meep to import (§17.1), these helpers are exercised
by the test suite on a Meep-equipped machine rather than in a Meep-free run. The one
orientation
subtlety — photonix `eps[iy, ix]` vs Meep's `[ix, iy]` weight indexing — is
localised to `to_material_grid` (a transpose).

### 17.3 MPB modes → `VectorModeData` (`modes.py`)

`meep.solve_modes(...)` solves cross-section modes with MPB's `find_k`
(target frequency → propagation constant) and returns a
`photonix.em.fde_vector.VectorModeData` — the same type the in-house full-vector
FDE produces — so an MPB result is a drop-in cross-check. It takes either explicit
rectangular-waveguide parameters (matching `rectangular_waveguide`) or an arbitrary
`eps` grid via `MaterialGrid`. `te_fraction` is computed from the MPB E-field when
available and degrades to `None` otherwise; `n_eff` is always returned.

### 17.4 Meep FDTD → `SDict` (`fdtd.py`)

`meep.waveguide_sparams(eps, *, dx, dy, wl, src_col, in_mon_col, out_mon_col, ...)`
mirrors the signature of the frequency-domain `em.fdfd.waveguide_sparams` exactly —
same grid, same column indexing, same `{("o1","o2"): …}` `SDict` out — so a caller
swaps FDFD for FDTD by changing the import. Method: a 2-D Meep simulation with an
`EigenModeSource`, mode monitors at the input/output planes, and
`get_eigenmode_coefficients` for the forward/backward modal amplitudes. Meep's
coefficients are power-normalised, so `|S21|² ` is the power transmission directly
(no `√β` rescaling, unlike the FDFD routine). The 2-D polarization label maps to
Meep z-parity in `parity_for` (`te → EVEN_Z`/H_z, `tm → ODD_Z`/E_z).

### 17.5 Benchmark credibility

`benchmarks/external/meep_adapter.py` (previously a stub) now computes the SOI-strip
TE0 `n_eff` (MPB) and the width-step TE transmission (Meep FDTD) through this same
backend, so `python benchmarks/run.py --external` produces a real external-solver
column — the "external reference rung" flagged as open in §16.7. It skips cleanly
(reported `ImportError`) where Meep is not installed.

### 17.6 Validation status & honest edges

`tests/test_em_meep.py` splits along the import contract (§17.1). Two **always-on
contract tests** run with or without Meep: `import photonix.em` succeeds, and the
Meep backend raises the install-hint `ImportError` when Meep is absent (both the
`from photonix.em import meep` and `em.meep` spellings). The remaining **13 backend
tests** — unit conversions, MaterialGrid weights, grid↔coordinate mapping, the
epsilon-lookup closure, plus the Meep-requiring MPB `n_eff` vs the FDE solver and a
straight-waveguide FDTD transmission ≈ 1 — are `skipif(not MEEP_PRESENT)`. In this
environment (Meep is conda-only and not installed) the contract tests pass and the
backend tests skip; the backend numbers are written against Meep's documented API
and **have not been executed here**, so they are asserted by construction and must
be confirmed on a Meep-equipped machine. That run, plus broadband (multi-frequency)
S-parameter extraction and a `meep.adjoint` hook into `photonix.optim` for
FDTD-grade inverse design, are the natural next steps on this foundation.
