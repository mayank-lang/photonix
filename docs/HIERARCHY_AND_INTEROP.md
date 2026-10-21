# Hierarchy, multimode circuits and netlist interop

Three additions to the circuit layer that build on the
[circuit engine](CIRCUIT_ENGINE.md) without changing it:

1. **Hierarchical netlists** (`photonix.circuit.hierarchy`): parameterised
   subcircuits, safe parameter expressions, per-call caching and hierarchical
   layout extraction.
2. **Multimode and polarization-aware circuits** (`photonix.circuit.modal`,
   `photonix.components.polarization`, `photonix.components.multimode`,
   `photonix.analysis.polarization`): ports that carry several modes.
3. **Netlist and layout interop** (`photonix.circuit.netlist_io`,
   `photonix.layout.pcell`, `photonix.layout.schematic`): SAX/gdsfactory
   netlists, gdsfactory pic-YAML, schematic-driven layout and deterministic
   PCells.

## 1. Hierarchical netlists

```python
from photonix.circuit import Netlist, Subcircuit, HierarchicalNetlist, compile_hierarchical

arm = Netlist()
arm.add("wg", "straight", length="=2*pi*radius")           # expression over a parameter
arm.expose("o1", ("wg", "o1")).expose("o2", ("wg", "o2"))

top = Netlist(name="chain")
top.add("a", "arm", radius=5.0).add("b", "arm", radius="=2*r0")
top.connect(("a", "o2"), ("b", "o1"))
top.expose("in", ("a", "o1")).expose("out", ("b", "o2"))

hnl = HierarchicalNetlist(top, {"arm": Subcircuit(arm, {"radius": 10.0})}, parameters={"r0": 4.0})
f = compile_hierarchical(hnl)                              # mode="hierarchical" | "flatten"
s = f(wl=1.55, params={"r0": 4.5}, **{"a.wg": {"loss_db_cm": 3.0}})
flat = hnl.flatten()                                       # instances "a.wg", "b.wg"
text = hnl.to_json()                                       # {"format": "photonix.netlist", "version": 2, ...}
```

| API | Purpose |
|---|---|
| `Subcircuit(netlist, parameters, doc)` | A reusable block; its `netlist.ports` are the ports of every instance. |
| `HierarchicalNetlist(top, subcircuits, parameters, name)` | `validate(models=None)`, `flatten(params=None)`, `instance_paths()`, `used_subcircuits()`, `to_dict/from_dict/to_json/from_json`. |
| `compile_hierarchical(hnl, models=None, *, mode, backend, cache)` | Differentiable model `f(*, wl, params=None, **{path: settings})`. |
| `evaluate_expression(expr, scope)`, `is_expression(v)` | The safe expression evaluator. |
| `load_netlist(source)` | Loads a version-1 (`Netlist`) or version-2 (`HierarchicalNetlist`) document. |
| `photonix.layout.extract_hierarchical_netlist(cell)` | One subcircuit per distinct composite cell. |

**Semantics.** An instance whose model name is a subcircuit name is a
subcircuit instance (subcircuit names win over registry models). Scoping is
SPICE's (Nagel, SPICE2, UCB/ERL M520, 1975): instance settings override the
subcircuit's parameter defaults, which override the leaf model defaults; a
parent's instance settings are evaluated in the parent's scope, a subcircuit's
own settings in its scope; parameter defaults may be expressions over other
parameters (cycles are reported). Instances may only set declared parameters.
Call-time overrides address any instance path (`"mesh.mzi3.int"`) and are
checked against the hierarchy.

**Expressions.** Strings starting with `=` are parsed with `ast` and walked
against a whitelist: numbers, `+ - * / ** %`, unary signs, parentheses, scope
names, `pi`, `e` and the functions `sqrt sin cos tan exp log abs min max`.
Attribute access, subscripts, other calls, keyword arguments, dunder or
underscore names and non-numeric literals raise `ValueError`; `eval` is never
used. Integer powers are evaluated in floating point (so `10**10**10`
overflows instead of exhausting memory). Arithmetic stays in the backend, so
parameters may be JAX tracers.

**Evaluation.** `mode="hierarchical"` replaces every subcircuit instance by
the exposed-port S-matrix the engine computes for it (recursive block
elimination). Because the interconnection equations are linear, eliminating a
block's internal waves first is a Schur complement of the flat system, so the
result equals the flat solve up to round-off; `mode="flatten"` solves the flat
topology once. With `cache=True` a call evaluates each distinct leaf
`(model, settings)` and each distinct `(subcircuit, resolved parameters,
overrides inside it)` once. Values are compared by value for numbers and by
identity for JAX arrays/tracers, so under `jit`/`grad` only instances sharing
the same setting objects are merged and gradients reach every instance. The
cache lives for one call (wavelength arrays are not hashed across calls).

**Serialization.** Version 2 is the version-1 document of the top netlist plus
`"parameters"` and `"subcircuits"` (each with `"parameters"`, `"doc"` and the
version-1 netlist fields). Expressions are stored verbatim. `Netlist.from_dict`
still reads only version 1 and now points to `load_netlist` for version 2.

**Hierarchical extraction.** `extract_hierarchical_netlist(cell)` extracts each
composite cell with `extract_netlist` in its own coordinates (coincidence is
invariant under rigid reference transforms, so all placements of a cell share
its internal connections). A subcircuit's ports are its leaves' optical
terminals not connected inside it: a terminal on a declared cell port takes
that port's name, others are named `"<instance>_<port>"` (the parent may
connect to or expose them). Equal cells from separate generator calls share a
subcircuit; differing cells with one name get `name_1`, ...; a cell name that
equals a leaf model used in the design gets `_cell`. By construction
`extract_hierarchical_netlist(c).flatten() == extract_netlist(c)`.

### Validation (hierarchy)

| Test (`tests/test_circuit_hierarchy.py`) | Checks | Result |
|---|---|---|
| `test_lattice_of_stage_subcircuits_equals_builder` | Lattice of "stage" subcircuits (expressions for arm lengths) vs `circuit.lattice_filter`, both modes | <= 1e-12 |
| `test_mzi_chain_hand_flattened` | Two MZI subcircuits vs a hand-written flat netlist (`flatten()` equality and S-matrix) | equal; <= 1e-12 |
| `test_crow_of_ring_subcircuits_equals_builder` | CROW of ring subcircuits with a top-level `radius` parameter vs `circuit.crow` (2 and 3 rings, 801 wavelengths) | <= 1e-12 |
| `test_three_level_nesting_agrees_with_flat_solve` | 3-level nesting with a feedback loop through the block, all 3 backends x 2 modes | <= 1e-12 |
| `test_mesh_from_mzi_subcircuit_equals_builder` | Clements mesh rebuilt from an `mzi` subcircuit vs `circuit.mzi_mesh` (imperfect splitters, lossy phases) and vs the target unitary | <= 1e-12 |
| `test_mesh_16_evaluates_each_unique_block_once` | 16x16 mesh (120 MZIs): coupler model called once, phase model once per distinct phase, one engine solve per distinct (theta, phi) + top; 2 solves for identical MZIs; 121 without cache | exact counts |
| `test_json_round_trip_keeps_expressions`, `test_version_one_files_still_load` | v2 round trip to equality, expressions kept, v1 loads and simulates identically | pass |
| `test_expression_values`, `test_expression_injection_and_errors_raise_value_error` | `=2*pi*radius` (radius 10) = 62.83185307179586; 21 injection/invalid strings (`__import__('os')`, `().__class__`, `open(1)`, ...) raise `ValueError` | pass |
| `test_cycle_reports_path`, `test_unknown_subcircuit_port_lists_available_ports` | `A -> B -> C -> A` named; unknown port lists the available ones | pass |
| `test_gradient_wrt_subcircuit_parameter_matches_flatten_and_fd` | `jax.grad` w.r.t. a ring-subcircuit radius: hierarchical = flatten (1e-9 rel), = central FD (1e-5 rel); `jit` | pass |
| `test_extract_hierarchical_flattens_to_extract_netlist` | 7 layout fixtures (layout MZI, MZI chain, rotated pair, 3 levels with mirror/rotation/routes/pads, fibre array, electrical, leaf) | `flatten() == extract_netlist`; S <= 1e-12 |

Evidence level 3 (independent computation: flat solves and closed-form
builders).

## 2. Multimode and polarization-aware circuits

A port carrying several modes is represented by *modal terminals*
`"<port>:<mode>"` (`"o1:TE0"`, `"o2:TM0"`), ordinary SDict port names, so the
engine is unchanged. `split_modal_port` also accepts the Meep multiport names
(`"o1:m1"`).

```python
from photonix.circuit import ModalNetlist, Netlist, lift_model
import photonix.components as c

ring2 = lift_model(c.all_pass_ring, ("TE0", "TM0"),
                   per_mode={"TE0": {"neff": 2.45, "ng": 4.2}, "TM0": {"neff": 1.75, "ng": 3.8}})
nl = Netlist()
nl.add("psr", "polarization_splitter_rotator").add("wg", "straight", length=20.0)
nl.connect(("psr", "o2"), ("wg", "o1"))
nl.expose("in", ("psr", "o1")).expose("out", ("wg", "o2")).expose("x", ("psr", "o3"))
s = ModalNetlist(nl).compile()(wl=1.55)       # keys like ("in:TM0", "x:TE0")
```

* `lift_model(model, modes, *, per_mode, coupling, coupling_port)` evaluates a
  single-mode model once per mode (per-mode settings on top of the call's) and
  optionally applies a port-side mode mixer `C(wl)` as `S' = C_out S C_in^T`
  (reciprocal; lossless for unitary `C`). Without coupling the cross-mode
  entries are absent.
* `ModalNetlist(netlist, modes, default_modes=("TE0",), on_mismatch)` lifts
  single-mode instances to their declared modes, keeps natively modal ones,
  connects `(A, "p:m") <-> (B, "q:m")` for every common mode, and terminates
  (with one `ModeMismatchWarning` per compiled model) or rejects modes present
  on one side only. `compile()` returns the model, `expand()` an equivalent
  plain `Netlist` plus its models.
* `eme_to_modal(EMEResult, modes_in=..., modes_out=...)` names all EME blocks
  (`Tf`, `Tb`, `Rf`, `Rb`) as modal terminals; `meep_dataset_to_modal(ds,
  {"m1": "TE0"})` relabels a Meep multiport dataset (still callable as a
  model).

New compact models (registered in `components.MODELS`; Jones basis
`(TE0, TM0)`; all reciprocal and passive for every parameter value):

| Model | Ports | Physics | Evidence | Limits |
|---|---|---|---|---|
| `polarization_rotator` | o1, o2 x {TE0, TM0} | Retarder `R(-θ) diag(e^{-jΓ/2}, e^{jΓ/2}) R(θ)` | 2 (Mueller block orthogonal, det 1) | Wavelength-independent |
| `polarization_splitter_rotator` | o1:{TE0,TM0}, o2:TE0, o3:TE0 | PSR after Dai & Bowers, Opt. Express 19, 10940 (2011): loss, conversion loss, TE/TM crosstalk; scaled by `max(1, σmax)` | 0 (behavioural) | Wavelength-independent; calibrate to data |
| `polarization_beam_splitter` | o1, o2, o3 x {TE0, TM0} | TE -> o2, TM -> o3 with finite extinction | 0 | Wavelength-independent |
| `grating_coupler_2d` | o1:{TE0,TM0} fibre, o2:TE0, o3:TE0 | Gaussian spectrum of `grating_coupler`, PDL, unitary crosstalk mixer | 0 | No angle dependence |
| `jones_fiber` (`pmd_fiber_jones`) | o1, o2 x {TE0=x, TM0=y} | N random birefringent sections (Gordon & Kogelnik, PNAS 97, 4541, 2000) | 2 (Maxwellian mean DGD within 3 %, 2000 seeds) | Not differentiable (random emulator); common phase omitted |
| `multimode_straight` | o1, o2 x modes | One propagation constant per mode; optional uniform 2-mode coupling (CMT) | 2 | Coupling for exactly 2 modes |
| `asymmetric_mode_converter` | o1:TE0, o4:TE0 (bus), o2:TE1, o3:TE1 | Coupled-mode theory (Yariv 1973; Huang 1994): `P = κ²/(κ²+δ²) sin²(SL)` | 2 (closed form to 1e-12) | Mismatch wavelength-independent |
| `mode_mux` | o1..oN:TE0 -> o(N+1):{TE0..} | Loss + uniform crosstalk, normalized to σmax <= 1 | 0 | Behavioural |

`photonix.analysis.polarization`: `jones_from_sdict`, `pdl_db`,
`polarization_extinction_db`, `stokes`, `jones_to_mueller`, `dgd` (Heffner,
IEEE PTL 4, 1066, 1992: `DGD = |arg(ρ1/ρ2)|/Δω` from the eigenvalues of
`J(ω2) J(ω1)^-1`; needs `DGD·Δω < π`) and `principal_states`.

### Validation (multimode)

| Test (`tests/test_modal_circuits.py`) | Checks | Result |
|---|---|---|
| `test_single_mode_modal_netlist_reproduces_flat_circuit` | TE0-only modal netlists (ring, layout MZI, lattice) vs `compile_circuit`, auto-lifted and explicitly lifted | <= 1e-14 |
| `test_te_tm_ring_resonates_per_polarization` | Lifted ring: notch depth at the exact root of `m λ = n_mode(λ) L` per polarization equals the analytic all-pass value; TE<->TM leakage exactly 0 | 1e-12 / exact |
| `test_rotator_psr_chain_is_passive_and_unitary_when_lossless`, `test_new_models_are_reciprocal_and_passive_over_parameters` | `analyze_sparameters`: passive, reciprocal; lossless chain unitary to 1e-12 | pass |
| `test_mode_converter_full_and_detuned_conversion` | Full conversion at `L = π/2κ`; detuned peak `κ²/(κ²+δ²)`; `P(L)` curve | <= 1e-12 |
| `test_pdl_mueller_stokes` | PDL of `diag(a, b)` = `20 log10(|a|/|b|)`; rotator Mueller 3x3 block orthogonal; `stokes(JE) = M stokes(E)` | <= 1e-12 |
| `test_dgd_of_single_birefringent_element`, `test_pmd_emulator_mean_dgd_is_maxwellian` | DGD of one element = τ (1e-9 ps); PSPs = element axes; 2000-seed mean DGD = `D√L` within 3 %, `<τ²> = 3π/8 <τ>²` within 6 % | pass |
| `test_meep_style_dataset_cascades_with_multimode_straight` | Synthetic `o1:m1..o2:m2` dataset relabelled and cascaded with a coupled multimode straight = matrix product | <= 1e-12 |
| `test_eme_taper_cut_and_recombined_through_modal_netlist` | 12-section EME taper (4 modes, absorber) cut mid-section; two EME runs joined in a `ModalNetlist` vs one run | <= 1e-9 |
| `test_gradient_through_modal_netlist_matches_fd` | `jax.grad` of converted power w.r.t. κ vs central FD and `sin²(κL)` | 1e-6 rel / 1e-12 |

## 3. Netlist and layout interop

**SAX / gdsfactory netlists.** `to_sax_netlist(nl)` writes
`{"instances": {inst: {"component", "settings"}}, "connections": {"a,o2": "b,o1"},
"ports": {"in0": "a,o1"}}`; a `HierarchicalNetlist` becomes a *recursive* SAX
netlist (`{"top_level": ..., "<subcircuit>": ...}`). `from_sax_netlist` reads
both, plus gdsfactory's `nets` list form; `placements`/`routes` are ignored
there. Parameter declarations and documentation go into an `"info"` key that
SAX and gdsfactory ignore, so round trips are exact.

**PCells.** `@pcell` (or `pcell(fn, name=..., cache_size=...)`) names a cell
`f"{name}_{sha1(canonical_json(settings))[:8]}"` over all arguments with
defaults applied. Canonical JSON sorts keys, rounds floats to a 1e-9 grid,
writes integral floats as integers, lists tuples, and names callables and cells.
Names are identical across interpreters (unlike `hash()`); an LRU cache returns
the same `Cell` object for identical calls, so GDS export writes one structure
per setting with no `$n` suffixes. `pcell_info(cell)` returns
`{"component", "settings"}`. A renamed leaf without a `model` keeps its
original name as `model`, so extraction is unchanged.

**Schematic-driven layout.** `SchematicSpec(netlist, placements, routes)`
holds component instances (component names and *layout* settings),
`Placement(x, y, rotation, mirror, port, dx, dy)` and
`RouteSpec(links, settings, strategy)` (`route_bundle` (with `sort=False`, so
links are honoured), `route_manhattan`, `route_auto`, `route_astar`); it reads
and writes gdsfactory pic-YAML (`to_yaml`/`from_yaml`, PyYAML imported lazily,
install with `pip install pyyaml`). `build_layout(spec, pdk)` instantiates via
`pdk.get_layout` (or `photonix.layout.components`), applies placements
(anchoring a port at `(x, y)` when `port` is given), places the rest along
`connections` breadth-first (islands at the origin), rejects connections left
more than `tol` apart or not facing (conflicting cycles), routes (default
style from the PDK cross-section) and exposes ports. Route cells are named
`"<group>_<k>"`. `schematic_from_layout(cell)` recovers placements,
connections, route links (as one-link `route_manhattan` groups; the router is
not recoverable from geometry) and ports; component settings are recovered
only for PCell-made cells. Mirroring follows photonix (about the x axis, then
rotation); gdsfactory mirrors about the y axis.

### Validation (interop)

| Test | Checks | Result |
|---|---|---|
| `test_netlist_io.py::test_netlist_sax_round_trip` | MZI, lattice, CROW, 4x4 mesh netlists -> SAX -> netlist equality; bit-identical S | pass |
| `test_netlist_io.py::test_recursive_sax_netlist_round_trips_hierarchy` | Hierarchy with parameters and expressions -> recursive SAX -> equality | pass |
| `test_layout_schematic.py::test_build_layout_closes_schematic_and_simulates` | `build_layout(spec, demo_pdk())` -> `extract_netlist`: topology equals the schematic with route instances inserted (no warnings); `pdk.simulate(layout)` = the schematic with drawn route lengths | <= 1e-12 |
| `test_schematic_from_layout_recovers_placements_connections_and_routes` | Placements (incl. port-anchored) to 1 nm, connections, links, ports; rebuild reproduces placements | pass |
| `test_yaml_round_trip_is_idempotent_and_reads_gdsfactory_sample` | YAML -> spec -> YAML idempotent; gdsfactory-style sample parsed and built | pass |
| `test_pcell_identity_names_and_cache`, `test_pcell_names_are_stable_across_interpreters`, `test_gds_export_writes_one_structure_per_pcell_setting` | `is` identity; distinct names; names equal in two fresh interpreters with random hash seeds; 10 identical straights -> one GDS structure, no `$` | pass |

## Limits

* Expressions are numeric only; a literal string setting cannot start with `=`.
* The per-call cache does not persist across calls; under `jit` only
  identical setting objects are deduplicated.
* Subcircuit instances cannot be given `port_aliases`; the hierarchical model
  returns the top netlist's port names only.
* Polarization/multimode models are behavioural with wavelength-independent
  polarization parameters (except the 2D grating coupler's spectrum and the
  fibre birefringence). They are not calibrated to any process; validate
  against full-vector EM (`photonix.em`, Meep) or measurements first.
* `ModalNetlist.expand()` reads the modal port structure from one probe
  evaluation; models whose port set depends on settings need a re-expansion.
* No optional cross-checks against an installed `sax` or `gdsfactory`
  (neither is available in the test environment); interop follows their
  published dictionary/YAML schemas. LVS-style comparison is done with plain
  netlist equality here; a dedicated LVS-lite lives with the verification
  package.
