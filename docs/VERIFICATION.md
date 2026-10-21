# Layout verification: LVS-lite, DRC rule deck, density and fill

`photonix.verification` checks a drawn layout against itself, against its
schematic and against an open rule table, so most layout mistakes surface
seconds after the layout is generated, before the foundry's qualified decks run
(through `photonix.layout.run_drc` / `run_lvs` and KLayout).
`photonix.layout.fill` places dummy fill with an optics-aware keep-out.

None of this is sign-off. The rule deck is photonix's own table, not a foundry
deck, and the LVS is geometric (ports, polygons, netlists), not a
device-extraction LVS. All functions need the `gdstk` extra
(`pip install "photonix[layout]"`).

```python
import photonix.verification as ver
from photonix.layout import components as lc

report = ver.run_lvs_lite(top, schematic)        # LvsReport: connectivity + parameters + netlist compare
print(report)                                    # issues by kind, with locations
report.to_lyrdb("lvs.lyrdb", top=top.name)       # markers for KLayout's Marker Browser

deck = ver.RuleDeck.from_pdk(pdk)                # or RuleDeck([Rule(...), ...]); JSON round trip
deck.rules.append(ver.Rule("M.EN.V", "enclosure", (12, 0), (11, 0), 0.2))
drc = deck.run(top, waivers=[ver.Waiver("M.EN.V", region, "test structure")])
ver.write_lyrdb(drc, "drc.lyrdb", top=top.name)
deck.to_klayout_drc("photonix.drc")              # the same rules as a KLayout DRC script

dm = ver.density_map(top, (1, 0), window=100.0, step=50.0)
keep = ver.evanescent_keepout(neff=2.43, n_clad=1.444, target=1e-4)   # 0.58 um
res = fill(top, layer=(1, 0), keepout={(1, 0): max(keep, 1.0)}, target_density=0.25)
top.add_ref(res.cell)
```

`examples/layout_verification.py` runs all of it on a small chip with two
deliberate mistakes.

## LVS-lite

`extract_netlist` connects two ports when their centres coincide and is silent
about everything else: a 5 nm gap, two waveguides overlapping by 3 µm, or a
port 0.8 µm away from its device's geometry all extract without a word.
`run_lvs_lite(top, schematic=None)` runs four checks and returns an
`LvsReport` (`issues`, `mapping`, `clean`, `summary()`, `to_lyrdb()`).
Every `LvsIssue` has a `kind`, a `location` in the top cell, the `objects`
involved, a `detail` text and, where one exists, the measured `value`.

| Check | Function | Issue kinds |
|---|---|---|
| Ports sit on their cell's drawn face | `check_ports_on_geometry` | `port_off_geometry`, `port_width_mismatch` |
| Connectivity of the placed layout | `check_optical_connectivity` | `near_open`, `open`, `short`, `port_width_mismatch`, `angle_mismatch`, `ambiguous_connection` |
| Settings versus drawn geometry | `check_parameters` (`extract_parameters`) | `parameter_mismatch` |
| Layout versus schematic | `compare_netlists` | `missing_instance`, `extra_instance`, `model_mismatch`, `setting_mismatch`, `connection_mismatch`, `external_port_mismatch` |

**Ports on geometry.** For a port at `c` with outward normal `n`, the point
`c - 1e-3 n` must be inside the merged polygons of the port layer and
`c + 1e-3 n` outside (else `port_off_geometry`, with the distance to the
nearest geometry as value). The chord of the geometry through the inner probe,
perpendicular to `n`, must cover the port span: a face narrower than the port
is a `port_width_mismatch` unless the geometry reaches the port width within
one port width of the face (corrugated gratings ending on a narrow tooth); a
face wider than the port is accepted (MMI and star-coupler bodies) unless
`exact_width=True`. Ports on marker layers (the fibre marker `(203, 0)`) are
skipped.

**Connectivity.** Leaf instances are found with the rules and names of
`extract_netlist`, port-less leaves (rings, fill tiles) included.

- Unconnected ports within `near` (1 µm) of each other: one `near_open` per
  pair, valued by their distance. An unconnected port whose face abuts another
  instance's polygon is also a `near_open`, unless the two instances short.
- When the top cell declares optical ports, every other unconnected port is an
  `open`. Without top ports the external ports are unknown and no opens are
  reported.
- Same-layer overlap of two instances is a `short`, valued by its area, except
  inside a disc of radius `max(width)` around each of their declared
  connections. This is the SiEPIC waveguide-overlap (DevRec) rule
  (Chrostowski & Hochberg, *Silicon Photonics Design*, CUP 2015, ch. 10).
  The booleans run on a 1e-6 µm grid.

**Parameter re-extraction.** `extract_parameters(leaf)` measures the drawn
polygons, independent of the generator:

- `length_geo = area / width` of a two-port ribbon or linear taper. A band of
  half-width `h` around a C2 curve of curvature radius above `h` has area
  `2 h L` (Steiner's tube formula). A polygonal arc of `N` segments of angle
  `dθ` has drawn length `R N sin dθ`, a relative deficit of `dθ²/6`.
- `gap_geo`: the exact minimum edge-to-edge distance between two islands
  (vertex-to-segment both ways). A ring drawn with vertices every `dθ`
  exceeds the circle's gap by at most `(R + w/2)(1 − cos(dθ/2))`.
- `radius_geo`, `length_x_geo`: Kasa algebraic circle fits (I. Kåsa, IEEE
  Trans. Instrum. Meas. IM-25, 8, 1976) of the outer and inner ring
  boundaries, averaged; for a racetrack, of its two end arcs. Bends are also
  fitted.

`check_parameters(top, rtol=2e-3, atol=2e-3)` compares these with each leaf's
`model_settings` (bends through their implied path length `R|θ|`, Euler bends
`R|θ|(1 + p)`).

**Netlist comparison.** Instances whose model is transparent (routes, by
default) are collapsed into direct connections first, so a routed layout
equals a schematic that joins the devices directly. Instances are then
matched deterministically, most specific evidence first: equal names with
equal models, the instances behind equally named external ports, propagation
along connections (port names and models must agree), and Weisfeiler-Lehman
colour refinement (Shervashidze et al., JMLR 12, 2539, 2011) on the union of
both port-level graphs with a greedy, consistency-checked assignment inside
each colour class. A swapped connection is therefore reported at the
connections concerned, not as "not isomorphic". Numeric settings compare with
`rtol`/`atol`.

## DRC rule deck

A `Rule(name, kind, layer, other=None, value, metric="euclidean", window=100, description="")`
has one of these kinds:

| Kind | Layers | Violation region |
|---|---|---|
| `width`, `space`, `area` | `layer` | as `photonix.layout.run_drc_precheck` (`space` also with the `square`/`projection` metrics) |
| `enclosure` | inner `layer`, outer `other` | `inner NOT erode(outer, e)`, including inner shapes outside `outer` |
| `separation` | `layer`, `other` | `grow(a, s/2) AND grow(b, s/2)`; overlap counts as distance 0 |
| `extension` | `a = layer`, `b = other` | `(grow(a AND b, x) NOT b) NOT grow(a NOT b, x)`: `b` must extend `x` past `a` wherever `a` lies inside `b`, except where `a` leaves `b` (a slab around a rib that ends where the rib becomes a strip) |
| `not_overlap`, `inside` | `layer`, `other` | `a AND b`; `a NOT b` |
| `min_angle` | `layer` | interior angle below the value at a vertex of a merged outline or hole (a small triangle marker) |
| `grid` | `layer` | each drawn vertex off the grid |
| `max_width` | `layer` | the whole feature when it survives an erosion by `w/2` (metal slotting) |
| `hole_area` | `layer` | enclosed holes smaller than the value |
| `min_density`, `max_density` | `layer` | windows of side `window` outside the bound |

Metrics follow KLayout. `euclidean` uses round-joined offsets (a disc), so it
catches corner-to-corner gaps. `square` uses mitred offsets. `projection`
uses Minkowski sums with axis-parallel segments, so only edges whose
projections overlap interact; it is exact for Manhattan edges, and oblique
edges are measured along the axes. All checks run on merged layers on the
1e-4 µm grid of the pre-check. Offsets are shrunk by 2e-4 µm, so a feature
drawn exactly at the rule passes, and violation pieces thinner than 3e-4 µm
are discarded as numerical slivers. A violation of two grid units (2 nm) is
therefore found.

`RuleDeck.run(cell, waivers=...)` returns a `DeckReport`, which is the shared
`DrcReport` record plus a `waived` list. A `Waiver(rule, region, reason)`
accepts violations of `rule` (`"*"` for all rules) that lie entirely inside
`region`. Decks round-trip through JSON (`to_json`/`from_json`).
`RuleDeck.from_pdk(pdk)` turns a PDK's `add_design_rules` table into
`<LAYER>.W/.S/.A` rules.

**Markers.** `write_lyrdb(report, path, top=...)` writes KLayout's
report-database XML, with one category per rule and one `polygon:` item per
violation in µm. The category description carries the rule's layer and value,
so `read_lyrdb` restores identical polygons. `read_lyrdb` also reads the
`box`, `edge` and `edge-pair` values written by KLayout's own DRC.

**KLayout cross-check.** `RuleDeck.to_klayout_drc(path)` writes an equivalent
KLayout DRC script: `width`/`space`/`separation`/`enclosing` with the
metric, `with_area`, `ongrid`, `holes`, sizing for maximum width, and booleans
for extension, overlap and inside. It reads `$input` and reports to `$report`,
the variables `run_klayout_deck` supplies. Densities and `min_angle` are
emitted as comments only, because their KLayout forms were not verified.
`compare_with_klayout(cell, deck, workdir=...)` runs both engines on the same
GDS and compares violation presence and total marker area per rule. Areas only
mean the same thing for rules whose markers are the violating region (width,
area, overlap, inside, hole area, maximum width): KLayout's space, separation
and enclosure markers span edge pairs rather than photonix's grown overlaps.

## Density and fill

`density_map(cell, layer, window, step, region)` gives the exact covered
fraction of every window. The layer is clipped to the region and fractured
along a tile grid (`gdstk.slice`, 1e-9 µm precision). Pieces are binned by
centroid, and window sums are box sums of a summed-area table (F. C. Crow,
SIGGRAPH 1984). The tile is the greatest common divisor of window and step, so
every window is an exact union of tiles. A window that reaches past the region
is normalised by its area inside the region. `check_density` and the density
rules report the windows outside the bounds.

`evanescent_keepout(neff, n_clad, wl, target)` returns
`d = ln(1/target) / (2γ)` with `γ = k0 √(neff² − n_clad²)`. This is the
distance at which the evanescent intensity has fallen to `target` times its
core-edge value. It is differentiable in all three inputs with the JAX
backend. In a 2-D cross-section the far field decays as `K0(γ r)`, which
carries an extra `1/√r`, so the estimate is conservative.

`fill(cell, layer=, region=, tile_size=, tile_space=, keepout={layer: d}, target_density=, window=, stagger=)`
returns a `FillResult` and does not modify the cell. The result's `cell` holds
tile references to add to the top cell. It also carries the tile `sites`, the
density maps before and after, and the `residual` windows still below target.

- **Keep-out.** Keep-out zones are round-joined dilations whose polygonal arcs
  are pushed outward by `1/cos(π/256)`, so no tile comes closer than requested.
  The fill layer itself keeps `tile_space` unless it is listed.
- **Site classification.** Lattice sites are classified in bulk against the
  allowed set eroded by the tile's circumradius (sure fits) and mitre-eroded by
  half a tile (sure misses). Only the thin band between them gets an exact
  boolean test.
- **Target density.** Tiles are added window by window, largest deficit first,
  in a spatially spread bit-reversed order. Top-up passes run after
  re-measuring, because a tile that straddles a window edge feeds the
  neighbouring window. This is a greedy heuristic for the min-variation fill
  problem (A. B. Kahng et al., IEEE Trans. CAD 18(4), 445, 1999), not its
  linear program.

## Validation

| Test | Checks | Result |
|---|---|---|
| 5 nm gap between two straights | exactly one `near_open`, value 0.005 µm | exact (1e-12) |
| Straights overlapping by 3 µm; 90° crossing | one `short` each, area 1.5 µm² and `w²` = 0.25 µm² | 1e-6 |
| Pre-fix `grating_coupler` (no taper) | `port_off_geometry` on `o1`, 0.785 µm | 1e-9; current generator clean |
| Every `layout.components` generator at defaults | `run_lvs_lite` clean | 20/20 |
| Rotation (90°, 217°) and mirroring of a defect layout | issue kinds, objects and values unchanged | to 1e-5 |
| `lc.mzi()` vs a hand-written netlist (same and renamed instances) | clean; one changed setting gives one `setting_mismatch`; swapped splitter outputs give exactly 4 `connection_mismatch` | pass |
| Route between two MMIs vs a direct schematic connection | clean (routes transparent) | pass |
| `directional_coupler`/`coupler` gap | recovered | 1e-9 |
| `ring_single` radius and racetrack `length_x` | recovered; gap within the sagitta bound | 1e-4 |
| Circular bend, `npts` 20/60/200 | `length_geo = R N sin dθ` | 1e-9 relative |
| Manhattan routes (circular, Euler) | `length_geo` within 2e-3 of the setting, never longer | pass |
| Enclosure, separation, space, extension at `value ∓ 2 nm` | fail / pass, all three metrics | pass |
| Diagonal squares 0.6 s and 0.8 s apart | Euclidean, square and projection flags as predicted | pass |
| 30° triangle, square, 135° chamfer, hole corners | only the 30° tip below 45° | pass |
| Layout shifted by 0.5 nm | every vertex off a 1 nm grid | pass |
| 10 µm × 40 µm metal vs 5 µm maximum width | one marker of area 400 µm², 5.002 µm fails, 4.998 µm passes | 1e-6 |
| Rule deck under rotation and mirroring | per-rule counts and areas unchanged | 1e-3 µm² |
| `.lyrdb` write/read | identical polygons, layers, values | exact |
| Stripes `a/p`, checkerboard | densities `a/p`, 0.5 in every window | 1e-12 |
| 50 random windows (rotated rectangles, ring) | box sums equal brute-force booleans | 1e-9 |
| Fill of an empty `L × L` region | `(floor((L − t)/(t + s)) + 1)²` tiles | exact |
| Fill around straights, bend and ring | min tile-to-waveguide distance ≥ keep-out (exact distance); filled layout LVS-clean | pass |
| Fill to 20 % in 50 µm windows | every window ≥ 0.2, below 0.3 | pass |
| `evanescent_keepout` gradient (JAX) | `d/dneff`, `d/dn_clad`, `d/dwl` vs finite differences | 1e-6 relative |
| Full-vector FDE: 0.2 µm Si line at 0.4–1.0 µm beside a 500 × 220 nm strip | slope of `ln(Δn_eff · d)` equals `−2γ` | within 5 % (about 3 %) |
| Same, at the keep-out distance (0.58 µm for `target = 1e-4`) | `Δn_eff < 1e-5` | pass |

Evidence levels (see [Validation](VALIDATION.md)): the geometric checks are
level 2, exact against constructed geometry with known answers. The keep-out
decay rate is level 3, a cross-check against an independent full-vector mode
solver. The KLayout cross-check (`compare_with_klayout`) is implemented but
was not run in CI (no KLayout executable), so the generated deck is
**unverified against KLayout**.

## Limits

- The checks are geometric. There is no device recognition from polygons
  (LVS-lite compares the netlist extracted from the cells' declared models),
  no electrical connectivity through vias, and no antenna, latch-up or
  context-dependent rules.
- Shorts are same-layer overlaps of distinct leaf instances. Overlaps inside a
  single leaf (a self-intersecting route) are not reported; use the DRC deck.
- Opens need declared top-cell ports. Port-on-geometry checks optical ports
  only, by default.
- `projection` is exact for Manhattan edges only. `separation` counts
  overlapping shapes as violations, unlike KLayout's edge-based check.
  `max_width` reports whole wide features, not only the excess.
- The generated KLayout deck omits density and angle rules, and its
  `max_width` uses KLayout's default sizing mode. Treat it as a cross-check,
  not as a deck.
- Fill to a target density is greedy, with windows equal to the step. It does
  not minimise density variation across overlapping windows. Tiles are
  axis-aligned squares on one layer per call.
- The keep-out estimate uses one decay constant. Multimode, slot or
  plasmonic guides and metal fill need their own mode solve: place the
  perturbation in an `XSection` and solve with `photonix.xsection.n_eff`, as
  the validation test does.
