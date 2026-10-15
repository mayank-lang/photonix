# Polygonal cross-sections (`photonix.xsection`)

`photonix.xsection` describes a waveguide cross-section once -- as painted
polygons of materials -- and turns it into whatever a solver needs:
permittivity for the FDE mode solvers, conductivities and capacities for heat or
conduction solvers, masks for heaters, contacts and doping. Using one geometry
object for all of them keeps the optical and thermal models of a device from
silently diverging.

It removes three limits of the rectangle-only builders
(`photonix.em.rectangular_waveguide`, `photonix.pdk.LayerStack.cross_section`):
any number of features per layer (coupler gaps, neighbouring waveguides),
lateral offsets (heaters, thermal crosstalk), and slanted sidewalls, undercuts,
ribs, disks or arbitrary simple polygons.

```python
import photonix.xsection as xs

cpl = xs.coupler(0.5, 0.2, 0.22, sidewall_deg=80.0, box=1.444)   # two trapezoidal cores
grid = cpl.uniform_grid(resolution=40)                           # cell-centred, like em.rectangular_waveguide
eps = cpl.eps(grid, wl=1.55, smoothing="tensor")                 # (3, ny, nx) Kottke-smoothed diagonal
modes = xs.solve_modes(cpl, num_modes=3)                         # full-vector, tensor smoothing
study = xs.n_eff_convergence(xs.strip(0.5, 0.22), (20, 30, 40))  # Richardson + GCI
cs = xs.strip(0.5, 0.22).to_em(resolution=40)                    # drop-in em.CrossSection
```

## API

| Object | Purpose |
|---|---|
| `Region`, `polygon`, `rect`, `band`, `trapezoid`, `rib`, `disk` | Simple polygons (µm, `x` lateral, `y` vertical) with a material, a painter's `order`, `tags` and optional per-region `properties` (e.g. `k_th`). `trapezoid(..., sidewall_deg, reference="top"/"middle"/"bottom")` sets the etch angle and the plane at which `width` is measured. `band` is a laterally unbounded layer. |
| `TensorGrid`, `uniform_grid`, `graded_grid` | Cell-edge grids; `uniform_grid` reproduces the `rectangular_waveguide` / `LayerStack` construction exactly; `graded_grid` refines boxes with a bounded growth ratio (for heat/conduction solvers; the FDE solvers need uniform grids). |
| `XSection(regions, background)` | The cross-section. `rasterize(grid, kernel)` returns a `Coverage`; `eps`, `coefficient`, `mask`, `normals`, `fractions`, `to_em` are wrappers. |
| `Coverage` | Exact visible fractions per sample, face coverages (box kernel) and kernel-weighted indicator gradients. Evaluate many wavelengths / temperatures / quantities on one rasterization. |
| `strip`, `rib_waveguide`, `coupler`, `heater_over_waveguide` | Library cross-sections (SOI defaults `n_Si = 3.4757`, `n_SiO2 = 1.444`). |
| `from_layer_stack(stack, features)` | Adapter from `photonix.pdk.LayerStack`: unpatterned levels become bands, each `(layer, x_center, width[, sidewall_deg])` feature a trapezoid, painted in stack order. Accepts the `{layer: width}` mapping of `LayerStack.cross_section`. |
| `solve_modes`, `n_eff`, `yee_permittivity` | Full-vector Yee-grid modes with per-component (staggered) tensor smoothing including `eps_xy`; anisotropic and lossy media; `n_eff` is differentiable through the geometry under JAX. |
| `n_eff_convergence` | Grid-refinement study (`photonix.em.estimate_convergence`) of `n_eff`, by default with the tent kernel. |
| `resolve_index`, `resolve_property` | Duck-typed materials: a number, a callable `n(wl)`, any object with `index(wl[, temperature])`, or a 3-tuple `(n_x, n_y, n_z)` for diagonal anisotropy. No material database is imported. Sign: see [Loss sign convention](#loss-sign-convention). |

## Loss sign convention

`photonix.xsection` and the `photonix.em` solvers use the engineering sign
(`exp(-j beta z)`): **`Im(n) < 0` is loss**, and a lossy mode has
`Im(n_eff) < 0`. `photonix.materials` returns the physics sign `n + i k`
(`k >= 0` is loss). `resolve_index` converts automatically:

1. an `index` method or callable with a `convention` keyword
   (`MaterialSpec.index`, i.e. everything from `get_material`, `doped_si`, ...)
   is called with `convention="engineering"`;
2. any other object or bound method defined in `photonix.materials` (e.g. a
   `LorentzFit`) is complex-conjugated;
3. numbers, plain callables and other objects are used as they are -- they
   must already follow the engineering sign (write `3.48 - 0.01j`, not
   `3.48 + 0.01j`, for an absorbing medium).

The rule applies to each entry of an anisotropic 3-tuple.

## Algorithms

**Exact rasterization.** A trapezoidal sweep cuts the window into horizontal
slabs at grid lines, vertices and edge crossings; winding numbers and the
painter's rank decide which region owns each trapezoid; the integral of each
owner-changing edge over each column is exact once the edge is split at the
column lines. Overlapping and abutting shapes need no polygon booleans, and the
result equals the exact rectangle rasterizers to rounding. A per-cell product
rule `f_i * prod(1 - f_j)` (sometimes proposed) is wrong wherever two abutting
shapes share a cut cell, so it is not used. Topology is decided in NumPy; the
arithmetic is replayed in `jax.numpy` when vertices are JAX values. The traced
derivative is the exact boundary (Hadamard) shape derivative: slab boundaries
are frozen, slanted edges move area through their position, and each
horizontal edge contributes `int V_y dx` with the velocity interpolated between
its two vertices -- so a face lying exactly on a grid line (as `graded_grid`
refine boxes and the tent kernel's dual lines produce) keeps its gradient, and
the two vertices of a horizontal edge each get their share. Where a face or
edge end lies exactly on a grid line the per-cell values are only one-sided
differentiable; the gradient is the mean of the two sides, which is what a
central finite difference sees. Forward values are unaffected.

**Smoothing.** `smoothing="arithmetic"` is the area average (legacy
convention). `smoothing="tensor"` is Kottke's anisotropic average
(Farjadpour et al., Opt. Lett. 31, 2972 (2006); Kottke, Farjadpour & Johnson,
Phys. Rev. E 77, 036611 (2008)) with the normal taken from the exactly
integrated, kernel-weighted gradient of the smoothed property; for isotropic media it is
`P <eps^-1>^-1 + (1 - P) <eps>`, `P = n n^T`. In the mode solver each field
component gets its own average on its Yee position and slanted interfaces add a
symmetric `eps_xy` coupling (essential: without it a 45-degree core converges at
first order). With the box kernel the normal is taken over a continuous
3 x 3-cell trapezoid kernel (1 on the cell, falling to 0 across its
neighbours), like the larger-than-a-pixel support of Kottke and Meep, rather
than over the cell itself: the cell-integrated gradient flips when a face
crosses a (Yee-shifted) grid line, which made `n_eff` jump by up to ~3e-5 for a
1e-9 µm geometry change at 20 px/µm. The smoothed `eps` and `n_eff` are now
continuous in the geometry with both kernels (they still have kinks where a
face crosses a grid line, as the area fractions do).

**Kernels.** `kernel="box"` averages each sample over its cell.
`kernel="tent"` uses the bilinear hat centred on the sample and reaching its
neighbours, computed exactly from per-cell moments `int (1, xi, eta, xi*eta) dA`
of the same sweep. The three-point second difference equals the tent average
of the second derivative, so a tent-averaged `eps` removes, at leading order,
the part of the interface error that depends on where a face cuts its cell;
the smooth `O(h^2)` stencil error remains. The box kernel keeps an `O(h^2)`
term that oscillates with that position. The argument is exact for field
components tangential to an interface and heuristic for normal ones. This is
our analysis of the standard scheme, checked numerically below, not a
published recipe.

## Choosing smoothing and kernel

| Goal | Use |
|---|---|
| Reproduce `em.rectangular_waveguide` / `solve_modes_fullvector` exactly | `smoothing="arithmetic"`, `kernel="box"` (`to_em`) |
| Best single-resolution `n_eff` | `smoothing="tensor"`, `kernel="box"` (default) |
| A converged `n_eff` with an error bar | `n_eff_convergence(..., kernel="tent")` (default) |
| Thermal / electrical coefficients | `coefficient(grid, "k_th", T, average="tensor")` (laminate: series across, parallel along) or `"harmonic"` / `"arithmetic"` |

Measured sensitivity to where the faces sit in the cells -- peak-to-peak
`n_eff` change of the TE0 mode of a 500 x 220 nm Si/SiO2 strip (vertical
sidewalls, tensor smoothing) over five sub-cell translations, and the mean over
those translations (the continuum value is about 2.4447):

| px/µm | box: vertical shift | box: lateral shift | box: mean | tent: vertical shift | tent: lateral shift | tent: mean |
|---|---|---|---|---|---|---|
| 20 | 1.8e-2 | 4.0e-3 | 2.4287 | 3.9e-3 | 4.1e-3 | 2.4181 |
| 30 | 5.1e-3 | 1.7e-3 | 2.4418 | 1.1e-3 | 1.4e-3 | 2.4330 |
| 40 | 5.4e-3 | 9.3e-4 | 2.4467 | 3.4e-4 | 6.6e-4 | 2.4383 |
| 60 | 3.5e-3 | 4.1e-4 | 2.4428 | 1.5e-4 | 1.2e-4 | 2.4421 |

The box error is smaller on average but its dependence on vertical alignment
barely decays between 30 and 60 px/µm; the tent error is a smooth, larger bias
whose alignment dependence falls quickly with resolution. Below ~20 px/µm both
kernels move by several 1e-3 under sub-cell shifts.

## Validation evidence (`tests/test_xsection.py`)

| Check | Evidence | Result |
|---|---|---|
| Area conservation | shoelace area, random non-convex polygons, rotated rectangles, slivers below one cell, 200-gon disk | to 1e-13 absolute |
| Per-cell areas, painter's order | independent Sutherland-Hodgman clipping per cell; visible area = area minus overlap; graded grid | to 1e-14 |
| Tent moments | per-dual-cell clipping + Green's-theorem polygon moments, graded grid, painted regions; reproduction of `int (1, x, y, xy) dA` (the box kernel fails the first moments) | to 1e-13 |
| Rectangle regression | `em.rectangular_waveguide(...).eps` at 20/33/40 px/µm; `LayerStack.cross_section` for a rib on the demo PDK (also through `from_layer_stack`) | to 1e-13 relative |
| Trapezoid at 90 deg | identical polygon and raster to the rectangle | exact |
| Laminate / normals | vertical interface: harmonic / arithmetic means (box: isotropic and anisotropic; tent: isotropic); slanted half-plane normals (both kernels) | to 1e-12 |
| Legacy solver | arithmetic path vs `em.solve_modes_fullvector` | `n_eff` to 1e-10 |
| Rotation invariance | 45-deg core vs axis-aligned Richardson limit 2.36506 (aligned grids, 40/80 px/µm) | tensor 2.5e-3 at 40 px/µm vs arithmetic 2.0e-2 |
| Convergence order | 80-deg trapezoid on face-aligned grids (20-60 px/µm) | tensor: observed order ~2 and a lower GCI than arithmetic; both reach a common limit |
| Tent extrapolation | `n_eff_convergence` on unaligned grids (30/40/60), 0- and 45-deg cores | within 2.2e-4 / 2.9e-4 of 2.36506; bounds 3.5e-3 / 4.2e-3 |
| Translation invariance (sub-cell shifts) | 80-deg, 220 nm trapezoid at 40 px/µm, vertical/lateral/diagonal shifts | tent: 4e-4 (GCI 7.5e-3); box: 5.9e-3 (vertical); box lateral shifts stay within its GCI |
| Physical ordering | rib > strip; shallower sidewall at fixed top width > vertical; fixed bottom width < vertical | ordering holds |
| Coupler splitting | even/odd TE splitting vs `SOI_STRIP_SPLITTING_FIT` (gaps 0.1-0.4 µm, 60 px/µm) | within 1.65 % (box, gap 0.2), 1.67 % (tent, gap 0.2); fit trusted to ~2 % |
| Symmetry | x <-> y transposition maps TE0 of `(n_x, n_y, n_z)` onto TM0 of `(n_y, n_x, n_z)` | to 1e-9 |
| Loss | metal heater: decaying complex index, larger when closer | `Im n_eff < 0`, ordering holds |
| Gradients (JAX) | per-sample `d eps / d width` and `d n_eff / d width` vs central differences, including corners exactly on grid lines / tent nodes | per sample 1e-5 (box) / 1e-7 (tent); `n_eff` 5e-6 relative |
| Shape derivative of horizontal edges | per-sample Jacobians of the exact fractions (both kernels) w.r.t. every vertex coordinate vs central differences: faces on grid lines (uniform and graded grids), independently moving vertices, coincident painted faces; total area vs shoelace; `d n_eff / d height` with faces on Yee / tent-dual grid lines | per sample 2e-6 relative; area to 1e-12; `n_eff` 2e-5 relative (formerly 0.04 vs 4.2) |
| Continuity of box tensor smoothing | Yee-staggered `eps` and `n_eff` for a 1e-9 µm width change with the side faces on grid lines (20 px/µm) | `eps` < 1e-6, `n_eff` < 1e-7 (formerly 1.6e-2 / 3e-5) |

Evidence levels (see [`VALIDATION.md`](VALIDATION.md)): exact identities and
laminate formulas are analytic (level 2); the mode results are convergence and
exact-symmetry checks (level 1); the coupler comparison is against an earlier
photonix calculation with the same discretization family, not an external
benchmark. Nothing here is calibrated to measurements.

## Limits

* The FDE mode solvers need uniform grids; `graded_grid` serves other solvers.
* `smoothing="tensor"` assumes dielectrics: averaging `1/eps` across a metal
  (`Re eps < 0`) interface is ill-posed -- use `"arithmetic"` with metals.
* With the default box kernel the `n_eff` error depends on where faces cut
  their cells (~5e-3 at 40 px/µm for a 220 nm core) and refinement sequences on
  ordinary grids are often not monotone; the tent kernel fixes that but has a
  ~2-3x larger systematic error at one resolution. Converge before quoting
  digits.
* `jax.grad` is supported through the geometry, `jax.jit` is not (the sweep
  topology is computed from concrete coordinates). The differentiable `n_eff`
  handles lossless (real) permittivity only. Gradients are first derivatives
  (the shape-derivative terms do not support higher orders); at grid-line
  coincidences they are the mean of the one-sided derivatives, so check them
  with central differences, or with steps that cross no grid line. The face
  coverages `Coverage.face_x` / `face_y` jump when an interface crosses a face;
  their derivatives are meaningful only away from that.
* Polygons must be simple; holes are made by painting (a higher-order region of
  the surrounding material). Bands are polygons of half-width 1e4 µm, so grids
  must lie inside `|x| < 1e4` µm.
* Material and property values are whatever the caller passes; the library
  defaults are the photonix SOI constants, and the thermal values in the
  example are illustrative bulk numbers, not process data.
