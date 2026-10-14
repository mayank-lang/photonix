# Routing

`photonix.layout` routes waveguides between ports with bends of a guaranteed
minimum radius, and *verifies* every geometric promise on the drawn polygon.
Every router returns ordinary single-polygon cells with `model="route"` and the
exact centre-line `length` and `n_bends`, so `extract_netlist` simulates what
was drawn (`components.waveguide_route`).

| Function | Connects | Topologies |
|---|---|---|
| `route` | 1 pair | sharp-cornered dogleg (schematic) |
| `route_manhattan` | 1 pair, axis-aligned | straight, L, Z, U, or explicit corner `waypoints` |
| `route_sbend` | 1 pair facing each other (any angle) | straight + S-bend (`euler`, `circular`, `cosine`) + straight |
| `route_auto` | 1 pair, any orientations | straight, S-bend, L, Z, U, 3-bend, 4-bend loops; non-Manhattan ports |
| `route_bundle` | N pairs, axis-aligned groups | L, Z/S, U, or a Manhattan backbone (`waypoints`) |
| `fanout` | N parallel ports | S-bends to a new pitch |
| `route_astar` | 1 pair, axis-aligned, around obstacles | any Manhattan path on a grid |
| `route_length_matched` | 1 pair | shortest route + trombones / shallow bumps to an exact length |

All routers accept `xs=` — any object with `width`, `layer` and `radius`
(for example a `photonix.pdk.CrossSection`) — as defaults for those three
settings; explicit keyword arguments win.

```python
import photonix.layout as lay

a = lay.Port("a", (0, 0), 0)
lay.route_auto(a, lay.Port("b", (30, 3), 180), radius=10.0)      # S-bend
lay.route_auto(a, lay.Port("b", (-20, 0), 180), radius=5.0)      # 4-bend loop: port2 behind port1
ins  = [lay.Port(f"i{k}", (0, 2 * k), 0) for k in range(8)]
outs = [lay.Port(f"o{k}", (100 - 2 * k, 80), 270) for k in range(8)]
bus  = lay.route_bundle(ins, outs, radius=10.0)                  # 8 nested L routes at 2 um pitch
```

`examples/routing_showcase.py` draws all of them on one chip, runs the DRC
pre-check, extracts the netlist and simulates it.

## Algorithms

**S-bends.** For an axial span `L` and lateral offset `D`:

* cosine `y = (D/2)(1 - cos(pi x/L))`: peak curvature `pi^2 |D| / (2 L^2)` at
  the ends, so a radius `R` needs `L >= pi sqrt(R |D| / 2)`; length
  `(2L/pi) E(-A^2)`, `A = pi D / (2L)` (complete elliptic integral);
* two opposite bends of angle `theta` meet the offset when
  `tan(theta/2) = |D|/L`, for any bend profile, because back-to-back bends with
  tangent length `T` advance `2T(1 + cos theta)` and offset `2T sin theta`.
  Circular arcs then have `R = (L^2 + D^2)/(4|D|)`; Euler bends
  `R = L / (2 T_1(theta)(1 + cos theta))` with the unit-radius clothoid tangent
  `T_1`, and length `2 R theta (1 + p)`.

`sbend_arc_length` and `sbend_min_radius` implement these with the backend
array module and are differentiable under JAX (trapezoid rule over one period
for the cosine — exponentially convergent — and 32-point Gauss–Legendre for the
clothoid). `sbend_min_span` inverts the radius bound. The Euler S-bend is the
routing default because its curvature is continuous; the cosine and circular
shapes jump from zero to their peak curvature at the joints.

**route_auto.** For each turn sequence (`L`/`R` per corner, up to `max_bends`)
the segment directions are fixed, so the segment lengths `l_i >= m_i` (each run
must hold the tangent lengths of its two bends) satisfy one linear equation per
axis. The minimum total is closed form: every run at its minimum plus the
signed slack on the runs pointing the needed way (split equally, which
reproduces `route_manhattan`'s centred Z jog). Candidates are scored by
`routed length + bend_penalty * n_bends` (default `bend_penalty = 2R(2 - pi/2)`)
and the cheapest one that never comes back within `width + min_space` of itself
is drawn. Ports at other angles first turn onto the nearest Manhattan heading
through a bend of exactly that angle; ports that share an orientation modulo
90 degrees are routed Manhattan in their own rotated frame.

**route_bundle** (river routing; Leiserson & Pinter, *SIAM J. Comput.* 12, 1983).
Give each segment of direction `u` the coordinate `lambda = n . x` of its line
along the left normal `n`. Manhattan routes sharing their segment directions
are planar exactly when they keep one `lambda` order on every segment, so the
two port groups are sorted by `lambda` along their exit/arrival headings and
paired rank by rank (`sort=True`); `sort=False` keeps the given pairing and
rejects a crossing one. L bundles turn at the intersection of their port lines
(for pitch `p` the lengths obey `L_k = L_0 - 2kp`); Z bundles jog in columns
one `pitch` apart, the route nearest the direction of travel first, with
offsets smaller than two corner offsets drawn as S-bends in their own slots; U
bundles nest, outermost (longest lateral run) in the farthest column; a
`waypoints` backbone offsets route `k` by `(k - (N-1)/2) pitch`. Each route is
drawn by `route_manhattan` (or the S-bend router).

**route_astar** (Hart, Nilsson & Raphael 1968; Lee 1961). States are
`(node, heading)` on a grid that also contains both port axes and the lines
hugging every obstacle's bounding box. A turn costs `bend_penalty` and must be
followed by a run of `2d`, so every run holds both fillets. Cost =
corner-polyline length + `bend_penalty * n_bends`; heuristic = Manhattan
distance + `bend_penalty` times the fewest turns still needed (0–4), which is
admissible, so A* returns Dijkstra's cost (`heuristic=False`). Obstacles are
rasterised by exact Euclidean distance at the nodes: straight steps need
`keepout + width/2 + grid/2`, turns need every node within `grid/sqrt(2)` of
the drawn fillet curve at `keepout + width/2 + grid/sqrt(2)` — enough, because
distance is 1-Lipschitz and every point of the curve has a node that near, to
bound every point of the drawn path (corner area the fillet never covers does
not block a turn). The straight leads at the ports (length
`max(d, keepout + width/2 + grid/sqrt(2))`) and the straight route between
collinear facing ports are checked exactly against every obstacle except the
polygons touching or containing that port (within `width/2` of its centre,
i.e. the device it belongs to); a blocked lead raises `RoutingError`.

**route_length_matched.** A trombone of height `h` adds `2h` of polyline and
four corners that each shorten the filleted path by `2d - L_bend`, so
`dL = 2h - 4(2d - L_bend)` and `h = (dL/n + 4(2d - L_bend))/2` for `n` equal
bumps (`h >= 2d`; `max_height` adds bumps; bumps spread over several runs when
one run is too short). Extra lengths below the smallest trombone use one
trapezoid of four bends of angle `theta`, with
`dL(theta) = 4T(1 - cos theta) + 4(L_bend(theta) - 2T)` rising continuously
from 0, solved by bisection. The base route is the shortest of: the S-bend
`route_auto` draws for facing ports (spanning the whole axial distance, when
its radius is at least `radius`), a minimum-radius S-bend followed by a
straight (lateral offset below `2d` and enough axial room for both bends),
and the `route_auto` Manhattan plan. A target equal to `route_auto`'s length
therefore succeeds; longer targets use the shortest base whose straight runs
can hold the meanders (the full-span S-bend has none), and the error for a
too-short target reports the true minimum. Targets between the full-span
S-bend and the next base route fail with an error naming that route's length.

## Validation (`tests/test_routing_advanced.py`)

Every check runs on the drawn polygon, with the centre line recovered from its
two offset sides, against an independent reference.

| Claim | Reference | Tolerance | Level |
|---|---|---|---|
| Cosine S-bend length | `scipy.integrate.quad` and `scipy.special.ellipe` | 1e-9 µm, 1e-13 rel | 2 |
| Clothoid end point (Gauss–Legendre) | Fresnel integrals `scipy.special.fresnel` | 1e-13 | 2 |
| Circular/Euler S-bend radius and length | closed forms; `sbend_min_span` inverts `sbend_min_radius` | 1e-9 rel | 2 |
| Minimum radius respected and reached | finite-differenced headings of the drawn curve | `≤ (1+1e-3)/R`, `≥ 0.98/R` | 1 |
| Constant width; length = area/width | polygon rungs; shoelace area, 2nd-order convergence in `npts` | 1e-9; < 1e-4 at 96 points/90° | 1 |
| JAX gradients of S-bend length/radius | central finite differences | 1e-7 rel | 2 |
| `route_auto` = `route_manhattan` on L/Z/U | `route_manhattan` | identical polygons | 3 |
| `route_auto` topology search optimal | exhaustive grid Dijkstra (`route_astar`, no obstacles) | 1e-9 µm | 3 |
| Every placement routes validly or raises | 210-case sweep of positions × orientations | > 85 % routed | 1 |
| Bundle: disjoint, spaced, `L_k = L_0 - 2kp`, connected | `gdstk.boolean`, `run_drc_precheck(min_space = p - w)`, `extract_netlist` + circuit | exact / 1e-9 / 1e-12 | 3 |
| Bundle clearance prefilter never misses | exhaustive O(N²) exact segment distances | identical verdicts | 3 |
| A* detour around a rectangle | analytic optimum `60 + 2(10 + keepout + w/2 + grid/2) - 4(2d - L_bend)` | 1e-9 µm | 2 |
| A* optimal | Dijkstra on the same graph | 1e-9 µm | 3 |
| A* clearance | `gdstk.offset(obstacles, keepout)` ∩ route = ∅; no vertex inside | exact | 3 |
| Length matching | target; area/width; equal extracted phases | 1 nm; 1e-4; 1e-10 | 2 |
| Trombone height | `h = (dL/n + 4(2d - L_bend))/2` | 1e-9 µm | 2 |

## Limits

* Geometry only: routes simulate as `waveguide_route` (straight of the drawn
  length, `n_bends * bend_loss_db`). An S-bend counts as two bends whatever its
  angle; `bend_loss_db` is a per-bend input, not computed from curvature.
* `_geometry.euler_curve` integrates the clothoid numerically; its end point is
  accurate to about `4e-8` relative (see the Fresnel test), so a drawn Euler
  S-bend's recorded length differs from the exact clothoid value by ~1e-7
  relative.
* `route_bundle` needs each port group to share one axis-aligned orientation
  and one topology; an L bundle whose port lines cross too close raises rather
  than adding corners (pass a `waypoints` backbone). Mixed jog/S-bend groups are
  laid out sequentially, which needs more axial room than pure jogs.
* `route_astar` needs axis-aligned ports, keeps up to `grid/sqrt(2)` extra
  clearance from curved or oblique obstacles, and exempts only the polygons a
  port touches (its own device) from the check of that port's straight lead.
  A grid search cannot see its own
  path: if the cheapest grid route would cross itself it falls back to the
  closed-form topologies (correct, not optimal around obstacles). The search is
  pure Python: a 400 × 300 µm window at 1 µm grid takes a fraction of a second
  with the heuristic, seconds without.
* `route_length_matched` checks clearance against itself only, not against
  neighbouring routes or devices; route groups need separate checks (e.g.
  `run_drc_precheck`).
* None of this replaces foundry DRC or LVS sign-off.
